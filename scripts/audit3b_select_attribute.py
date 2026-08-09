#!/usr/bin/env python3
"""3B mini-audit driver: benign-dictionary feature selection + attribution.

For one benchmark: runs reselect_token_sae_features (full positive pool as
the discovery set) for each extracted layer, picks the layer whose top-5 has
the largest mean row gap, then runs feature_token_attribution there. Prints
a compact profile-vs-behavioral verdict per selected feature.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layers", default="12,18,24")
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--benign-sample-prob", type=float, default=0.25)
    args = ap.parse_args()

    scripts = Path(__file__).resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet")
    pos_users = sorted(scores.loc[scores["y"] == 1, "user_id"].astype(str).unique())
    user_file = args.out_dir / "all_positive_users.txt"
    user_file.write_text("\n".join(pos_users) + "\n")
    print(f"[driver] {len(pos_users)} positive users", flush=True)

    best = None
    for layer in [int(x) for x in args.layers.split(",")]:
        out_frontier = args.out_dir / f"reselect_l{layer}"
        cmd = [sys.executable, str(scripts / "reselect_token_sae_features.py"),
               "--extract-dir", str(args.extract_dir), "--data-dir", str(args.data_dir),
               "--frontier-dir", str(args.frontier_dir), "--out-frontier-dir", str(out_frontier),
               "--layer", str(layer), "--latent-mult", str(args.latent_mult), "--k", str(args.k),
               "--discovery-user-file", str(user_file),
               "--benign-sample-prob", str(args.benign_sample_prob), "--device", "cuda"]
        subprocess.run(cmd, check=True)
        cfg = out_frontier / f"layer_{layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
        feats = pd.read_csv(cfg / "delta_sae_top_features.csv")
        gapcol = "row_gap" if "row_gap" in feats.columns else feats.columns[1]
        gap = float(feats.head(5)[gapcol].mean())
        print(f"[driver] layer={layer} top5_gap={gap:.6f} ids={feats.head(5)['feature_id'].tolist()}", flush=True)
        if best is None or gap > best[1]:
            best = (layer, gap, out_frontier)

    layer, gap, out_frontier = best
    print(f"[driver] BEST layer={layer} gap={gap:.6f}", flush=True)
    attr_dir = args.out_dir / f"attribution_l{layer}"
    cmd = [sys.executable, str(scripts / "feature_token_attribution.py"),
           "--extract-dir", str(args.extract_dir), "--data-dir", str(args.data_dir),
           "--frontier-dir", str(out_frontier), "--adapter-dir", str(args.adapter_dir),
           "--layer", str(layer), "--latent-mult", str(args.latent_mult), "--k", str(args.k),
           "--out-dir", str(attr_dir), "--device", "cuda"]
    subprocess.run(cmd, check=True)

    d = json.load(open(attr_dir / "feature_token_attribution.json"))
    print(f"[verdict] best_layer={layer} top5={d['top5']}")
    for f in d["features"]:
        prof = f["positive_DAY_mass_frac"] + f["positive_PSY_mass_frac"]
        print(f"[verdict] feat {f['feature_id']}: DAY={f['positive_DAY_mass_frac']:.3f} "
              f"PSY={f['positive_PSY_mass_frac']:.3f} SES={f['positive_SES_mass_frac']:.3f} "
              f"-> profile_mass={prof:.3f}", flush=True)


if __name__ == "__main__":
    main()
