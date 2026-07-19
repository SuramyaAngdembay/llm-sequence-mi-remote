#!/usr/bin/env python3
"""Re-rank token delta-SAE features using only discovery-user positives.

The SAE itself is trained on benign token deltas and is untouched; only the
feature ranking (`delta_sae_top_features.csv`, which drives top-k / control
set selection in the causal and necessity evals) depends on positive labels.
This script recomputes that ranking with positive rows restricted to a
discovery user list, so that downstream effects can be confirmed on held-out
users without selection contamination.

Writes a new frontier-style config dir containing a copy of the SAE checkpoint
plus the re-ranked `delta_sae_top_features.csv`, usable as --frontier-dir by
the eval scripts.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from sae_core import TopKSAE, evaluate_features, choose_feature_sets


def load_rows(
    extract_dir: Path,
    layer: int,
    *,
    keep_positive_users: set[str],
    user_by_example: dict[int, str],
    y_by_example: dict[int, int],
    benign_sample_prob: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    layer_dir = extract_dir / f"layer_{layer}"
    chunk_paths = sorted(layer_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks for layer {layer} in {layer_dir}")
    rng = np.random.default_rng(seed)
    vec_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    n_pos_dropped = 0
    n_pos_kept = 0
    for path in chunk_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        vecs = np.asarray(obj["delta"], dtype=np.float32)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        y = np.asarray([y_by_example.get(int(i), 0) for i in idx], dtype=np.int64)
        users = np.asarray([user_by_example.get(int(i), "") for i in idx])
        pos_mask = y > 0
        pos_keep = pos_mask & np.isin(users, list(keep_positive_users))
        n_pos_dropped += int((pos_mask & ~pos_keep).sum())
        n_pos_kept += int(pos_keep.sum())
        benign_mask = ~pos_mask
        if benign_sample_prob < 1.0:
            benign_mask = benign_mask & (rng.random(len(idx)) < benign_sample_prob)
        keep = pos_keep | benign_mask
        vec_parts.append(vecs[keep])
        y_parts.append(y[keep])
    x = np.concatenate(vec_parts, axis=0)
    yy = np.concatenate(y_parts, axis=0)
    stats = {
        "n_rows": int(len(yy)),
        "n_positive_rows_kept": n_pos_kept,
        "n_positive_rows_dropped_confirmation": n_pos_dropped,
        "n_benign_rows": int((yy == 0).sum()),
    }
    return x, yy, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True, help="source frontier root")
    ap.add_argument("--out-frontier-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--discovery-user-file", type=Path, required=True)
    ap.add_argument("--benign-sample-prob", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg_rel = Path(f"layer_{args.layer}") / f"m{args.latent_mult:02d}_k{args.k:02d}"
    src_cfg = args.frontier_dir / cfg_rel
    dst_cfg = args.out_frontier_dir / cfg_rel
    dst_cfg.mkdir(parents=True, exist_ok=True)

    discovery_users = {
        ln.strip() for ln in args.discovery_user_file.read_text().splitlines() if ln.strip()
    }
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet")
    meta = pd.read_csv(args.data_dir / "example_metadata.csv")
    if "example_idx" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "example_idx"})
    user_by_example = dict(zip(meta["example_idx"].astype(int), meta["user_id"].astype(str)))
    y_by_example = dict(zip(scores["example_idx"].astype(int), scores["y"].astype(int)))

    bundle = torch.load(src_cfg / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32)
    x_std = np.asarray(bundle["x_std"], dtype=np.float32)

    x, y, stats = load_rows(
        args.extract_dir,
        args.layer,
        keep_positive_users=discovery_users,
        user_by_example=user_by_example,
        y_by_example=y_by_example,
        benign_sample_prob=args.benign_sample_prob,
        seed=args.seed,
    )
    x -= x_mean
    x /= x_std

    feature_df, eval_stats = evaluate_features(model, x, y, device=device, batch_size=args.batch_size)
    feature_sets = choose_feature_sets(feature_df)

    shutil.copy2(src_cfg / "delta_sae_model.pt", dst_cfg / "delta_sae_model.pt")
    feature_df.to_csv(dst_cfg / "delta_sae_top_features.csv", index=False)
    summary = {
        "source_frontier": str(src_cfg),
        "discovery_user_file": str(args.discovery_user_file),
        "n_discovery_users": len(discovery_users),
        "benign_sample_prob": args.benign_sample_prob,
        "row_stats": stats,
        "eval_stats": {k: float(v) for k, v in eval_stats.items()},
        "feature_sets": {k: [int(i) for i in v] for k, v in feature_sets.items()},
    }
    (dst_cfg / "reselect_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
