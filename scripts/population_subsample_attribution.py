#!/usr/bin/env python3
"""Test the population-structure hypothesis by malicious-user subsampling.

Holds the model, benchmark, SAE, and benign population fixed; varies only the
positive population used for feature selection. For each subsample size K and
repetition, re-ranks features by row_gap (positive-mean minus benign-mean
activation, mirroring sae_core.evaluate_features) restricted to the sampled
malicious users, takes the top-5, and reports their profile-mass share.

Efficient design: token deltas are encoded through the SAE once; per-user
activation sums for positives and global sums for the benign sample are
precomputed, so each repetition is O(d_latent) arithmetic.

Hypothesis prediction: small K -> profile-bound top features dominate;
large K -> behavioral features dominate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from sae_core import TopKSAE
from eval_token_delta_sae_causal import read_jsonl
from feature_token_attribution import token_classes_for_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sizes", default="4,10,20,30,60")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--benign-sample-prob", type=float, default=0.10)
    ap.add_argument("--profile-mass-threshold", type=float, default=0.5)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--sae-batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    model = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    x_mean = torch.from_numpy(np.asarray(bundle["x_mean"], dtype=np.float32)).to(device)
    x_std = torch.from_numpy(np.asarray(bundle["x_std"], dtype=np.float32)).to(device)
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

    rng = np.random.default_rng(args.seed)
    # accumulators
    ben_sum = np.zeros(d_latent, dtype=np.float64); ben_rows = 0
    user_sum: Dict[str, np.ndarray] = {}
    user_rows: Dict[str, int] = {}
    profile_mass = np.zeros(d_latent, dtype=np.float64)
    total_mass = np.zeros(d_latent, dtype=np.float64)

    layer_dir = args.extract_dir / f"layer_{args.layer}"
    for path in sorted(layer_dir.glob("chunk_*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        vecs = np.asarray(obj["delta"], dtype=np.float32)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        pos = np.asarray(obj["position"], dtype=np.int64)
        y = np.asarray([y_by_example.get(int(i), 0) for i in idx], dtype=np.int64)
        keep = (y > 0) | (rng.random(len(idx)) < args.benign_sample_prob)
        vecs, idx, pos, y = vecs[keep], idx[keep], pos[keep], y[keep]
        if len(idx) == 0:
            continue
        for start in range(0, len(idx), args.sae_batch_size):
            sl = slice(start, start + args.sae_batch_size)
            xb = torch.from_numpy(vecs[sl]).to(device)
            xb = (xb - x_mean) / x_std
            with torch.no_grad():
                _, z = model(xb)
            z = z.cpu().numpy().astype(np.float64)
            yb = y[sl]; ib = idx[sl]; pb = pos[sl]
            ben_mask = yb == 0
            if ben_mask.any():
                ben_sum += z[ben_mask].sum(axis=0); ben_rows += int(ben_mask.sum())
            pos_mask = ~ben_mask
            if pos_mask.any():
                zp = z[pos_mask]
                total_mass += zp.sum(axis=0)
                for r in np.nonzero(pos_mask)[0]:
                    e = int(ib[r]); u = user_by_example.get(e, "")
                    if u not in user_sum:
                        user_sum[u] = np.zeros(d_latent, dtype=np.float64); user_rows[u] = 0
                    user_sum[u] += z[r]; user_rows[u] += 1
                    cl = cls_for(e); p = int(pb[r])
                    if p < len(cl) and cl[p] in ("DAY", "PSY"):
                        profile_mass += z[r]
    ben_mean = ben_sum / max(ben_rows, 1)
    profile_share = profile_mass / np.maximum(total_mass, 1e-9)
    users = sorted(user_sum)
    print(f"[base] users={len(users)} benign_rows={ben_rows}", flush=True)

    sizes = [int(x) for x in args.sizes.split(",")]
    out_rows: List[dict] = []
    rep_rng = np.random.default_rng(args.seed + 1)
    for K in sizes:
        n_reps = 1 if K >= len(users) else args.reps
        for rep in range(n_reps):
            chosen = users if K >= len(users) else list(rep_rng.choice(users, size=K, replace=False))
            s = np.zeros(d_latent, dtype=np.float64); n = 0
            for u in chosen:
                s += user_sum[u]; n += user_rows[u]
            gap = s / max(n, 1) - ben_mean
            top5 = np.argsort(-gap)[:5]
            shares = profile_share[top5]
            out_rows.append({
                "K": K, "rep": rep,
                "top5": [int(i) for i in top5],
                "top5_profile_shares": [round(float(v), 4) for v in shares],
                "mean_profile_share": round(float(shares.mean()), 4),
                "n_profile_features": int((shares > args.profile_mass_threshold).sum()),
                "n_positive_rows": n,
            })
            print(f"K={K} rep={rep} profile_features={out_rows[-1]['n_profile_features']}/5 "
                  f"mean_share={out_rows[-1]['mean_profile_share']}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "subsample_attribution.json").write_text(json.dumps(out_rows, indent=2))
    df = pd.DataFrame([{k: r[k] for k in ("K", "rep", "mean_profile_share", "n_profile_features")} for r in out_rows])
    agg = df.groupby("K").agg(mean_profile_share=("mean_profile_share", "mean"),
                              mean_n_profile=("n_profile_features", "mean"),
                              reps=("rep", "count")).reset_index()
    agg.to_csv(args.out_dir / "subsample_summary.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
