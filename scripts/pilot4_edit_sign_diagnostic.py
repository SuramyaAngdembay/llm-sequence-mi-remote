#!/usr/bin/env python3
"""Post-hoc diagnostic for Pilot 4 (EXPLORATORY, 2026-09-26, after its results).

For each baseline direction set U, the Pilot 4 edit at a union-support token
moves the token's projection p_t = U^T z~_t to the donor prototype p. This
script reports, separately for malicious and matched benign receivers, how
far and in which direction that move goes before the edit is rescaled to the
SAE edit's size: the coefficient change c_t = p - p_t, its sign (for the
one-dimensional mean-difference direction), and the rescaling gain. It is
needed to read the malicious-minus-benign contrast E8c: a group difference in
the sign of the edit would produce a group difference in its effect without
any anomaly-specific information in the direction.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from pilot_intervention_specificity import load_delta_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--delta-cache", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--populations", type=Path, required=True)
    ap.add_argument("--directions", type=Path, required=True)
    ap.add_argument("--pilot4-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from sae_core import TopKSAE
    from eval_token_delta_sae_causal import build_example_slices, build_sparse_cache_for_examples, load_eval_examples

    man = json.loads(args.pilot4_manifest.read_text())
    S = man["selected"]
    pops = json.loads(args.populations.read_text())
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    dept_of = lambda i: dict(kv.split("=", 1) for kv in meta.loc[i, "text"].split("\n", 1)[0][4:].split())["dept"]
    excluded = set(man["excluded_users"])
    pairs = [(a, b) for a, b in pops["receiver_pairs"] if meta.loc[a, "user_id"] not in excluded]
    kind = {a: "malicious" for a, _ in pairs} | {b: "benign" for _, b in pairs}
    receivers = sorted(kind)

    d = np.load(args.directions)
    U_all = d["U"].astype(np.float32)
    sets = {name: U_all[:, cols] for name, cols in man["pc_sets"].items()}
    sets["meanDiff"] = (d["d"] / np.linalg.norm(d["d"]))[:, None].astype(np.float32)

    cfg_dir = args.frontier_dir / "layer_26" / "m02_k04"
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"]))
    sae.load_state_dict(bundle["state_dict"]); sae.eval()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32).reshape(-1)
    x_std = np.asarray(bundle["x_std"], dtype=np.float32).reshape(-1)
    x, idx = load_delta_cache(args.delta_cache, set(receivers))
    slices = build_example_slices(idx.astype(np.int64, copy=False))
    with torch.no_grad():
        Z = build_sparse_cache_for_examples(sae, x, slices, receivers, x_mean=x_mean, x_std=x_std,
                                            device=torch.device("cpu"), batch_size=4096)

    acc = defaultdict(lambda: defaultdict(list))            # (set, kind) -> stat -> values
    per_user = defaultdict(lambda: defaultdict(list))        # (set, kind, user) -> signs
    for i in receivers:
        rows = Z[i][:, S].sum(axis=1) > 0
        if not rows.any():
            continue
        zt = (x[slices[i]][rows] - x_mean) / x_std
        for name, U in sets.items():
            p = np.asarray(man["prototypes"][dept_of(i)][name], dtype=np.float32)
            c = p[None, :] - zt @ U                           # coefficient change per token
            nat = np.linalg.norm((c @ U.T) * x_std, axis=1)
            acc[(name, kind[i])]["c_norm"].extend(np.linalg.norm(c, axis=1).tolist())
            acc[(name, kind[i])]["natural_edit_norm"].extend(nat.tolist())
            if U.shape[1] == 1:
                acc[(name, kind[i])]["c"].extend(c[:, 0].tolist())
                acc[(name, kind[i])]["projection"].extend((zt @ U)[:, 0].tolist())
                per_user[(name, kind[i])][meta.loc[i, "user_id"]].extend((c[:, 0] < 0).tolist())
    out = {}
    for (name, k), st in sorted(acc.items()):
        rec = {"n_tokens": len(st["c_norm"]), "coef_change_norm_median": float(np.median(st["c_norm"])),
               "natural_edit_norm_median": float(np.median(st["natural_edit_norm"]))}
        if st["c"]:
            c = np.asarray(st["c"])
            rec.update({"share_tokens_projection_decreased": float(np.mean(c < 0)),
                        "coef_change_mean": float(c.mean()), "projection_mean": float(np.mean(st["projection"])),
                        "prototype_mean_over_depts": float(np.mean([man["prototypes"][dp]["meanDiff"][0] for dp in man["prototypes"]])),
                        "share_decreased_per_user_median": float(np.median([np.mean(v) for v in per_user[(name, k)].values()]))})
        out[f"{name}|{k}"] = rec
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
