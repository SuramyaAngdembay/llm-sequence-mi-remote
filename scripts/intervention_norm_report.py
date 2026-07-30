#!/usr/bin/env python3
"""Perturbation-norm comparability check for causal patching arms.

Reviewer concern: if top-feature interventions inject larger residual-stream
perturbations than active-control interventions, larger score changes could
reflect perturbation size rather than feature content. This script recomputes,
for every (receiver, donor, feature set, alpha) row actually evaluated in a
causal run's candidate CSV, the norm of the arm-differing component of the
injected shift, entirely in closed form (no model scoring):

  shift_t = [decode(z'_t) - decode(z_t)] + [decode(z_t) - delta_t]

The second term (reconstruction substitution) is applied at every token in
both arms and is receiver-specific but arm-common; we report it per receiver.
The first term is the feature edit: with a linear decoder,
  edit_t = (W_dec[:, F] @ (z'_{t,F} - z_{t,F})) * x_std,
nonzero only at the receiver's F-active positions with
z'_{t,F} = (1-alpha) z_{t,F} + alpha p_F (donor prototype), exactly as in
eval_token_delta_sae_causal.build_token_patch_shift.

Outputs per-row edit norms and per-(feature_set, donor_type) aggregates over
all candidate rows and over per-receiver best rows (most negative delta), the
subset that defines the reported estimand.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from sae_core import TopKSAE


def load_needed_tokens(extract_dir: Path, layer: int, needed_idx: set[int]):
    vec_parts: List[np.ndarray] = []
    idx_parts: List[np.ndarray] = []
    layer_dir = extract_dir / f"layer_{layer}"
    for path in sorted(layer_dir.glob("chunk_*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        keep = np.isin(idx, list(needed_idx))
        if not keep.any():
            continue
        vec_parts.append(np.asarray(obj["delta"], dtype=np.float32)[keep])
        idx_parts.append(idx[keep])
    if not vec_parts:
        raise SystemExit("no token rows found for requested examples")
    return np.concatenate(vec_parts), np.concatenate(idx_parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-csv", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--context-modes", default=None,
                    help="comma list; default: all in the CSV")
    ap.add_argument("--sae-batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    model = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32).reshape(1, -1)
    x_std = np.asarray(bundle["x_std"], dtype=np.float32).reshape(1, -1)

    cand = pd.read_csv(args.candidate_csv)
    if args.context_modes:
        cand = cand[cand["context_mode"].isin(args.context_modes.split(","))]
    cand = cand.copy()
    cand["features"] = cand["selected_features"].map(ast.literal_eval)

    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet")
    idx_by_id = dict(zip(scores["example_id"].astype(str), scores["example_idx"].astype(int)))
    needed_ids = set(cand["receiver_example_id"]) | set(cand["donor_example_id"])
    needed_idx = {idx_by_id[e] for e in needed_ids if e in idx_by_id}
    missing = [e for e in needed_ids if e not in idx_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} example ids missing from scores parquet, e.g. {missing[:3]}")

    vecs, idx = load_needed_tokens(args.extract_dir, args.layer, needed_idx)
    print(f"[load] token rows={len(idx)} examples={len(needed_idx)}", flush=True)

    all_feats = sorted({f for feats in cand["features"] for f in feats})
    col_of = {f: j for j, f in enumerate(all_feats)}
    W_sub = model.decoder.weight.detach().cpu().numpy()[:, all_feats].astype(np.float32)  # d_in x n_feats

    # Encode all needed tokens once; keep only the involved feature columns
    # plus the per-token reconstruction-substitution norm (arm-common term).
    z_cols = np.zeros((len(idx), len(all_feats)), dtype=np.float32)
    recon_norm = np.zeros(len(idx), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(idx), args.sae_batch_size):
            sl = slice(start, start + args.sae_batch_size)
            xb = torch.from_numpy((vecs[sl] - x_mean) / x_std).to(device)
            recon, z = model(xb)
            z_cols[sl] = z[:, all_feats].cpu().numpy()
            err = ((recon - xb).cpu().numpy() * x_std)
            recon_norm[sl] = np.linalg.norm(err, axis=1)
    del vecs

    tok_slices: Dict[int, np.ndarray] = {e: np.flatnonzero(idx == e) for e in needed_idx}

    rows_out: List[dict] = []
    for r in cand.itertuples(index=False):
        e_recv = idx_by_id[r.receiver_example_id]
        e_don = idx_by_id[r.donor_example_id]
        fcols = np.array([col_of[f] for f in r.features], dtype=np.int64)
        z_recv = z_cols[tok_slices[e_recv]][:, fcols]
        z_don = z_cols[tok_slices[e_don]][:, fcols]
        active = z_recv.sum(axis=1) > 0.0
        n_active = int(active.sum())
        if n_active == 0:
            edit_fro = 0.0
            edit_mean = 0.0
        else:
            don_active = z_don.sum(axis=1) > 0.0
            proto = z_don[don_active].mean(axis=0) if don_active.any() else z_don.mean(axis=0)
            dz = float(r.alpha) * (proto[None, :] - z_recv[active])  # n_active x n_feats
            edit = dz @ W_sub[:, fcols].T                            # n_active x d_in
            edit *= x_std
            norms = np.linalg.norm(edit, axis=1)
            edit_fro = float(np.sqrt((norms ** 2).sum()))
            edit_mean = float(norms.mean())
        rows_out.append({
            "context_mode": r.context_mode, "feature_set": r.feature_set,
            "donor_type": r.donor_type, "alpha": float(r.alpha),
            "receiver_example_id": r.receiver_example_id,
            "donor_example_id": r.donor_example_id, "delta": float(r.delta),
            "n_active_tokens": n_active, "edit_norm_fro": edit_fro,
            "edit_norm_mean_token": edit_mean,
            "recon_norm_mean_token": float(recon_norm[tok_slices[e_recv]].mean()),
        })

    out = pd.DataFrame(rows_out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "intervention_norms_rows.csv", index=False)

    def agg(df: pd.DataFrame) -> pd.DataFrame:
        return (df.groupby(["context_mode", "feature_set", "donor_type"])
                  .agg(n=("edit_norm_fro", "size"),
                       edit_fro_mean=("edit_norm_fro", "mean"),
                       edit_fro_median=("edit_norm_fro", "median"),
                       edit_tok_mean=("edit_norm_mean_token", "mean"),
                       active_tokens_mean=("n_active_tokens", "mean"),
                       recon_tok_mean=("recon_norm_mean_token", "mean"))
                  .reset_index())

    best = (out.sort_values("delta")
               .groupby(["context_mode", "feature_set", "donor_type", "receiver_example_id"], as_index=False)
               .first())
    summary = {"all_rows": agg(out).to_dict(orient="records"),
               "best_rows": agg(best).to_dict(orient="records")}
    (args.out_dir / "intervention_norms_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["best_rows"], indent=2))


if __name__ == "__main__":
    main()
