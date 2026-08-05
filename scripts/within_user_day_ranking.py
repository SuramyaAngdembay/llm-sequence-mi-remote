#!/usr/bin/env python3
"""Within-user day-ranking: does the score separate a malicious user's
positive days from that same user's benign days?

Under a pure user-novelty account the score is driven by who the user is,
which is constant within user, so within-user AUC should sit at chance; a
behavioral signal ranks a user's malicious days above their own benign days.
For each malicious user with at least one positive and one benign day we
compute the within-user ROC-AUC of adapted_nll (positive vs own-benign
days), then summarize: per-user AUCs, their mean/median, the fraction of
users above 0.5, and a user-level cluster bootstrap CI for the mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    order = np.argsort(neg, kind="mergesort")
    s = neg[order]
    lo = np.searchsorted(s, pos, side="left")
    hi = np.searchsorted(s, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-parquet", type=Path, required=True)
    ap.add_argument("--score-col", default="adapted_nll")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.scores_parquet)[["user_id", "y", args.score_col]]
    per_user = []
    for u, g in df.groupby("user_id"):
        pos = g.loc[g["y"] == 1, args.score_col].to_numpy(dtype=float)
        neg = g.loc[g["y"] == 0, args.score_col].to_numpy(dtype=float)
        if len(pos) == 0 or len(neg) == 0:
            continue
        per_user.append({"user_id": u, "n_pos": int(len(pos)), "n_ben": int(len(neg)),
                         "auc": auc(pos, neg)})
    aucs = np.array([r["auc"] for r in per_user])
    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.draws)
    for i in range(args.draws):
        boot[i] = aucs[rng.integers(0, len(aucs), size=len(aucs))].mean()
    report = {
        "score_col": args.score_col,
        "n_users": len(per_user),
        "mean_auc": float(aucs.mean()),
        "mean_auc_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "median_auc": float(np.median(aucs)),
        "frac_users_above_0.5": float((aucs > 0.5).mean()),
        "per_user": sorted(per_user, key=lambda r: -r["n_pos"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_user"}, indent=2))
    for r in report["per_user"][:8]:
        print(f"  {r['user_id']}: auc={r['auc']:.3f} (pos={r['n_pos']}, ben={r['n_ben']})")


if __name__ == "__main__":
    main()
