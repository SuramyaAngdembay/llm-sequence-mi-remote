#!/usr/bin/env python3
"""Rank SAE features by behavioral (SES-token) association only.

The standard feature ranking uses all token positions, which lets static
profile tokens (DAY/PSY) dominate. This script recomputes the ranking with
rows restricted to SES-class tokens, additionally computes each feature's
profile-mass share, excludes profile-dominated features from the ranking
CSV, and writes a frontier-style config dir consumable by the causal and
necessity eval scripts.

Supports single discovery-file mode and LOUO mode (like
reselect_token_sae_features.py) so behavioral selection keeps the same
held-out discipline as the profile-feature analyses.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from sae_core import TopKSAE, evaluate_features, choose_feature_sets
from eval_token_delta_sae_causal import read_jsonl
from feature_token_attribution import token_classes_for_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--out-frontier-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--discovery-user-file", type=Path, default=None)
    ap.add_argument("--louo-splits-dir", type=Path, default=None)
    ap.add_argument("--profile-mass-max", type=float, default=0.5,
                    help="exclude features with PSY+DAY activation-mass share above this")
    ap.add_argument("--benign-sample-prob", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.discovery_user_file is None and args.louo_splits_dir is None:
        raise SystemExit("Provide --discovery-user-file or --louo-splits-dir")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg_rel = Path(f"layer_{args.layer}") / f"m{args.latent_mult:02d}_k{args.k:02d}"
    src_cfg = args.frontier_dir / cfg_rel
    bundle = torch.load(src_cfg / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    model = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32)
    x_std = np.asarray(bundle["x_std"], dtype=np.float32)
    d_latent = int(bundle["d_latent"])

    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet")
    user_by_example = dict(zip(scores["example_idx"].astype(int), scores["user_id"].astype(str)))
    y_by_example = dict(zip(scores["example_idx"].astype(int), scores["y"].astype(int)))
    texts = {i: ex["text"] for i, ex in enumerate(read_jsonl(args.data_dir / "eval.jsonl"))}
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), use_fast=True)

    class_cache: Dict[int, list] = {}

    def cls_for(e: int) -> list:
        if e not in class_cache:
            class_cache[e] = token_classes_for_text(texts[e], tokenizer, args.max_seq_len)[0]
        return class_cache[e]

    # Load rows once: all positives + benign sample; keep class per row.
    layer_dir = args.extract_dir / f"layer_{args.layer}"
    rng = np.random.default_rng(args.seed)
    vec_parts, y_parts, user_parts, ses_parts = [], [], [], []
    profile_mass = np.zeros(d_latent, dtype=np.float64)
    total_mass = np.zeros(d_latent, dtype=np.float64)
    for path in sorted(layer_dir.glob("chunk_*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        vecs = np.asarray(obj["delta"], dtype=np.float32)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        pos = np.asarray(obj["position"], dtype=np.int64)
        y = np.asarray([y_by_example.get(int(i), 0) for i in idx], dtype=np.int64)
        users = np.asarray([user_by_example.get(int(i), "") for i in idx])
        keep = (y > 0) | (rng.random(len(idx)) < args.benign_sample_prob)
        vecs, idx, pos, y, users = vecs[keep], idx[keep], pos[keep], y[keep], users[keep]
        if len(idx) == 0:
            continue
        is_ses = np.zeros(len(idx), dtype=bool)
        is_profile = np.zeros(len(idx), dtype=bool)
        for r in range(len(idx)):
            cl = cls_for(int(idx[r]))
            p = int(pos[r])
            c = cl[p] if p < len(cl) else "SES"
            is_ses[r] = c == "SES"
            is_profile[r] = c in ("DAY", "PSY")
        # accumulate profile-mass shares over the whole dictionary (positives only)
        pos_mask = y > 0
        if pos_mask.any():
            xb = torch.from_numpy((vecs[pos_mask] - x_mean) / x_std).to(device)
            with torch.no_grad():
                _, z = model(xb)
            z = z.cpu().numpy()
            total_mass += z.sum(axis=0)
            prof_rows = is_profile[pos_mask]
            if prof_rows.any():
                profile_mass += z[prof_rows].sum(axis=0)
        vec_parts.append(vecs); y_parts.append(y); user_parts.append(users); ses_parts.append(is_ses)

    x_all = np.concatenate(vec_parts); y_all = np.concatenate(y_parts)
    users_all = np.concatenate(user_parts); ses_all = np.concatenate(ses_parts)
    profile_share = profile_mass / np.maximum(total_mass, 1e-9)
    print(f"[base] rows={len(y_all)} ses_rows={int(ses_all.sum())} positives={int((y_all>0).sum())}", flush=True)

    if args.louo_splits_dir is not None:
        folds = sorted(args.louo_splits_dir.glob("louo_*_discovery_users.txt"))
    else:
        folds = [args.discovery_user_file]

    for fold_file in folds:
        discovery = {ln.strip() for ln in fold_file.read_text().splitlines() if ln.strip()}
        pos_mask = y_all > 0
        keep = ses_all & ((~pos_mask) | np.isin(users_all, list(discovery)))
        x = (x_all[keep] - x_mean) / x_std
        y = y_all[keep]
        feature_df, eval_stats = evaluate_features(model, x, y, device=device, batch_size=args.batch_size)
        feature_df["profile_mass_share"] = profile_share[feature_df["feature_id"].astype(int).to_numpy()]
        behavioral = feature_df[feature_df["profile_mass_share"] <= args.profile_mass_max].copy()
        if args.louo_splits_dir is not None:
            held = fold_file.name.replace("louo_", "").replace("_discovery_users.txt", "")
            dst_cfg = args.out_frontier_dir / f"louo_{held}" / cfg_rel
        else:
            dst_cfg = args.out_frontier_dir / cfg_rel
        dst_cfg.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_cfg / "delta_sae_model.pt", dst_cfg / "delta_sae_model.pt")
        behavioral.to_csv(dst_cfg / "delta_sae_top_features.csv", index=False)
        sets = choose_feature_sets(behavioral)
        summary = {
            "fold": fold_file.name,
            "n_features_total": int(len(feature_df)),
            "n_behavioral": int(len(behavioral)),
            "profile_mass_max": args.profile_mass_max,
            "eval_stats": {k: float(v) for k, v in eval_stats.items()},
            "feature_sets": {k: [int(i) for i in v] for k, v in sets.items()},
            "top5_profile_mass": [round(float(profile_share[i]), 4) for i in sets["top5"]],
        }
        (dst_cfg / "behavioral_rank_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[fold {fold_file.name}] behavioral top5={sets['top5']} profile_mass={summary['top5_profile_mass']}", flush=True)


if __name__ == "__main__":
    main()
