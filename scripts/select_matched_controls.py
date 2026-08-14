#!/usr/bin/env python3
"""Select perturbation-matched active control features.

Reviewer concern: control5_active features are matched on activity but not on
intervention dose, so top-feature edits are 3-10x larger in residual-space
norm. This selects a control set matched to the top-5 on the quantities that
determine edit size:
  - activation frequency (fraction of pool token rows with the feature active)
  - mean activation magnitude when active
  - decoder column norm
Candidates must still be gap-neutral (|row_gap| below a quantile cap), and
top-5 features are excluded. Selection greedily pairs each top feature with
its nearest unused candidate in z-scored stat space.

Outputs matched_controls.json ({"control5_matched": [...]}) for the eval
scripts' --control-feature-file, plus a stats report comparing the predicted
per-token edit-size proxy (mean_active_mag * decoder_norm) across sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_core import TopKSAE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gap-quantile-cap", type=float, default=0.5,
                    help="candidates must have |row_gap| below this quantile of nonzero |gap|")
    ap.add_argument("--min-active-frac", type=float, default=0.002)
    ap.add_argument("--sample-prob", type=float, default=0.25,
                    help="row subsample for stats (all rows kept if <=0 or >=1)")
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
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32).reshape(1, -1)
    x_std = np.asarray(bundle["x_std"], dtype=np.float32).reshape(1, -1)
    d_latent = int(bundle["d_latent"])

    feature_df = pd.read_csv(cfg_dir / "delta_sae_top_features.csv")
    gapcol = "row_gap" if "row_gap" in feature_df.columns else feature_df.columns[1]
    top5 = [int(x) for x in feature_df.head(5)["feature_id"]]
    gaps = feature_df.set_index("feature_id")[gapcol].astype(float)

    # Streaming per-feature stats over the pool.
    rng = np.random.default_rng(args.seed)
    n_rows = 0
    active_count = np.zeros(d_latent, dtype=np.float64)
    act_sum = np.zeros(d_latent, dtype=np.float64)
    layer_dir = args.extract_dir / f"layer_{args.layer}"
    with torch.no_grad():
        for path in sorted(layer_dir.glob("chunk_*.pt")):
            obj = torch.load(path, map_location="cpu", weights_only=False)
            vecs = np.asarray(obj["delta"], dtype=np.float32)
            if 0 < args.sample_prob < 1:
                keep = rng.random(len(vecs)) < args.sample_prob
                vecs = vecs[keep]
            for s in range(0, len(vecs), args.sae_batch_size):
                xb = torch.from_numpy((vecs[s:s+args.sae_batch_size] - x_mean) / x_std).to(device)
                _, z = model(xb)
                z = z.cpu().numpy()
                active_count += (z > 0).sum(axis=0)
                act_sum += z.sum(axis=0)
                n_rows += z.shape[0]
    freq = active_count / max(n_rows, 1)
    mean_act = act_sum / np.maximum(active_count, 1)
    dec_norm = np.linalg.norm(model.decoder.weight.detach().cpu().numpy(), axis=0)
    edit_proxy = mean_act * dec_norm

    gap_arr = np.zeros(d_latent)
    for fid, g in gaps.items():
        if 0 <= int(fid) < d_latent:
            gap_arr[int(fid)] = g
    nz = np.abs(gap_arr[np.abs(gap_arr) > 0])
    gap_cap = float(np.quantile(nz, args.gap_quantile_cap)) if len(nz) else 0.0

    eligible = np.flatnonzero((freq >= args.min_active_frac) & (np.abs(gap_arr) <= gap_cap))
    eligible = np.array([f for f in eligible if f not in set(top5)])
    if len(eligible) < 5:
        raise SystemExit(f"only {len(eligible)} eligible candidates; relax caps")

    stats = np.stack([np.log10(freq + 1e-9), np.log10(mean_act + 1e-9), dec_norm], axis=1)
    mu, sd = stats[eligible].mean(0), stats[eligible].std(0) + 1e-9
    z_all = (stats - mu) / sd
    chosen: list[int] = []
    for f in top5:
        d = np.linalg.norm(z_all[eligible] - z_all[f], axis=1)
        for idx in np.argsort(d):
            cand = int(eligible[idx])
            if cand not in chosen:
                chosen.append(cand)
                break

    def summ(ids):
        ids = np.array(ids, dtype=int)
        return {"freq": [round(float(x), 5) for x in freq[ids]],
                "mean_act": [round(float(x), 3) for x in mean_act[ids]],
                "dec_norm": [round(float(x), 3) for x in dec_norm[ids]],
                "edit_proxy": [round(float(x), 3) for x in edit_proxy[ids]],
                "abs_gap": [round(float(abs(gap_arr[i])), 5) for i in ids]}

    report = {"n_rows_stats": int(n_rows), "gap_cap": gap_cap,
              "top5": {"ids": top5, **summ(top5)},
              "control5_matched": {"ids": chosen, **summ(chosen)},
              "edit_proxy_ratio_top_over_matched": round(float(edit_proxy[top5].mean() / max(edit_proxy[chosen].mean(), 1e-9)), 3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"control5_matched": chosen}))
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
