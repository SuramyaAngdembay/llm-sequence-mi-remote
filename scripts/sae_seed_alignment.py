#!/usr/bin/env python3
"""Decoder-space alignment between SAE seeds and benchmarks.

All token delta-SAEs decode into the same d_model=4096 residual space of the
same base model, so decoder directions are directly comparable across seeds
and across benchmarks. For each source SAE's top-k features (by row_gap), we
report the best-match |cosine| against every feature of a target SAE, plus a
whole-dictionary baseline (median best-match over all source features).

Reads frontier-style cfg dirs (delta_sae_model.pt + delta_sae_top_features.csv).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_sae(cfg_dir: Path) -> tuple[np.ndarray, list[int]]:
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    dec = bundle["state_dict"]["decoder.weight"].numpy()  # (d_in, d_latent)
    dec = dec / (np.linalg.norm(dec, axis=0, keepdims=True) + 1e-8)
    feats = pd.read_csv(cfg_dir / "delta_sae_top_features.csv")
    top5 = [int(x) for x in feats.sort_values("row_gap", ascending=False).head(5)["feature_id"]]
    return dec.astype(np.float32), top5


def best_match_stats(dec_a: np.ndarray, ids_a: list[int], dec_b: np.ndarray) -> dict:
    sims = np.abs(dec_a.T @ dec_b)  # (latent_a, latent_b)
    best = sims.max(axis=1)
    top = best[ids_a]
    return {
        "top5_best_match_mean": float(top.mean()),
        "top5_best_match_min": float(top.min()),
        "top5_best_matches": [round(float(v), 4) for v in top],
        "all_features_best_match_median": float(np.median(best)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="semicolon-separated label_a=path_a,label_b=path_b pairs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cache: dict[str, tuple[np.ndarray, list[int]]] = {}

    def get(label_path: str):
        label, path = label_path.split("=", 1)
        if label not in cache:
            cache[label] = load_sae(Path(path))
        return label, cache[label]

    rows = []
    for pair in args.pairs.split(";"):
        a_spec, b_spec = pair.split(",")
        la, (dec_a, top_a) = get(a_spec)
        lb, (dec_b, _) = get(b_spec)
        stats = best_match_stats(dec_a, top_a, dec_b)
        rows.append({"source": la, "target": lb, **stats})
        print(f"{la} -> {lb}: top5 mean |cos| = {stats['top5_best_match_mean']:.4f} "
              f"(min {stats['top5_best_match_min']:.4f}; dict median "
              f"{stats['all_features_best_match_median']:.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
