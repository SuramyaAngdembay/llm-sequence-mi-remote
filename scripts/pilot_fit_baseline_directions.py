#!/usr/bin/env python3
"""H8 (EXPLORATORY, 2026-09-25): fit the simpler decompositions of the same deltas.

Streams the layer-26 token-delta cache once (CPU) and writes, in the SAE's
standardized coordinates (delta - x_mean) / x_std:

  U        principal directions (d x n_pcs) of a deterministic stride sample of
           ALL eval rows, which is the population the SAE itself was trained on;
  evals    their eigenvalues (variance explained);
  d        the mean-difference direction mean_pos - mean_neg over the ranking
           population used for the SAE feature ranking: positive rows of
           discovery users only (confirmation users' positive rows excluded),
           benign rows of users who are not confirmation users;
  gap      U^T d, the malicious-minus-benign gap of each principal direction.

Provenance (row and user counts, chunk names, output sha256) goes to a JSON
sidecar. Nothing here looks at any receiver's label or score.

  python scripts/pilot_fit_baseline_directions.py --extract-dir D --layer 26 \
      --sae F/layer_26/m02_k04/delta_sae_model.pt --confirmation-users C \
      --discovery-users R --out directions.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--sae", type=Path, required=True, help="delta_sae_model.pt (for x_mean and x_std)")
    ap.add_argument("--confirmation-users", type=Path, required=True)
    ap.add_argument("--discovery-users", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=4, help="PCA uses every stride-th token row")
    ap.add_argument("--n-pcs", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    bundle = torch.load(args.sae, map_location="cpu", weights_only=False)
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32).reshape(-1)     # stored as (1, d)
    x_std = np.asarray(bundle["x_std"], dtype=np.float32).reshape(-1)
    d = int(x_mean.shape[0])
    conf = {l.strip() for l in args.confirmation_users.read_text().splitlines() if l.strip()}
    disc = {l.strip() for l in args.discovery_users.read_text().splitlines() if l.strip()}
    if conf & disc:
        raise SystemExit("confirmation and discovery users overlap")
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").set_index("example_idx")
    user = scores["user_id"].to_dict()
    y = scores["y"].astype(int).to_dict()
    man = pd.read_csv(args.extract_dir / "chunk_manifest.csv")
    man = man[man["layer"] == args.layer].sort_values("chunk_id")
    paths = [args.extract_dir / f"layer_{args.layer}" / Path(p).name for p in man["path"]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing chunks: {missing[:3]}")

    n_pca = 0
    s_pca = np.zeros(d, dtype=np.float64)
    xtx = np.zeros((d, d), dtype=np.float64)
    n_pos = n_neg = 0
    s_pos = np.zeros(d, dtype=np.float64)
    s_neg = np.zeros(d, dtype=np.float64)
    users_pos, users_neg, users_pca = set(), set(), set()
    row_offset = 0
    for k, path in enumerate(paths):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        delta = np.asarray(obj["delta"])                       # float16 (rows, d)
        ex = np.asarray(obj["example_idx"], dtype=np.int64)
        n = len(ex)
        # PCA sample: every stride-th row of the whole pool (the SAE's training exposure)
        sel = np.arange((-row_offset) % args.stride, n, args.stride)      # one global stride across chunks
        blk = (delta[sel].astype(np.float32) - x_mean) / x_std
        n_pca += len(sel)
        s_pca += blk.sum(axis=0, dtype=np.float64)
        xtx += (blk.T @ blk).astype(np.float64)
        users_pca.update(user[int(i)] for i in np.unique(ex[sel]))
        # ranking population for the mean-difference direction (all rows, no stride)
        ex_user = np.array([user[int(i)] for i in ex])
        ex_y = np.array([y[int(i)] for i in ex])
        pos = (ex_y == 1) & np.isin(ex_user, list(disc))
        neg = (ex_y == 0) & ~np.isin(ex_user, list(conf))
        for mask, acc, cnt_name in ((pos, s_pos, "pos"), (neg, s_neg, "neg")):
            if mask.any():
                blk = (delta[mask].astype(np.float32) - x_mean) / x_std
                acc += blk.sum(axis=0, dtype=np.float64)
        n_pos += int(pos.sum()); n_neg += int(neg.sum())
        users_pos.update(np.unique(ex_user[pos]).tolist()); users_neg.update(np.unique(ex_user[neg]).tolist())
        row_offset += n
        del obj, delta
        print(f"[fit] chunk {k + 1}/{len(paths)} rows {n} ({time.time() - t0:.0f}s)", flush=True)

    mean_pca = s_pca / n_pca
    cov = xtx / n_pca - np.outer(mean_pca, mean_pca)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][: args.n_pcs]
    U = evecs[:, order].astype(np.float32)
    evals = evals[order].astype(np.float32)
    mean_pos, mean_neg = s_pos / n_pos, s_neg / n_neg
    dvec = (mean_pos - mean_neg).astype(np.float32)
    gap = (U.T @ dvec).astype(np.float32)
    np.savez(args.out, U=U, evals=evals, d=dvec, gap=gap, mean_pca=mean_pca.astype(np.float32),
             mean_pos=mean_pos.astype(np.float32), mean_neg=mean_neg.astype(np.float32),
             total_variance=np.float32(np.trace(cov)))
    prov = {
        "layer": args.layer, "sae": str(args.sae), "stride": args.stride, "n_pcs": args.n_pcs,
        "pca_rows": int(n_pca), "pca_users": len(users_pca), "pca_population": "every stride-th row of all eval token rows",
        "ranking_population": "positive rows of discovery users; benign rows of non-confirmation users",
        "pos_rows": int(n_pos), "pos_users": len(users_pos), "neg_rows": int(n_neg), "neg_users": len(users_neg),
        "confirmation_users_in_pos": sorted(users_pos & conf), "confirmation_users_in_neg": sorted(users_neg & conf),
        "variance_explained_top_pcs": [float(v) for v in (evals / np.trace(cov))[:10]],
        "gap_top_pcs": [float(v) for v in gap[:10]], "mean_diff_norm": float(np.linalg.norm(dvec)),
        "chunks": [p.name for p in paths], "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "seconds": round(time.time() - t0),
    }
    args.out.with_suffix(".json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps({k: prov[k] for k in ("pca_rows", "pos_rows", "neg_rows", "pos_users", "neg_users",
                                             "confirmation_users_in_pos", "out_sha256")}), flush=True)


if __name__ == "__main__":
    main()
