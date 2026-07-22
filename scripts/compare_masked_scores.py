#!/usr/bin/env python3
"""Compare detector behavior between original and profile-masked scorings.

Joins example_scores.parquet files (original + variants) on example_id and
reports day-level ROC/PR-AUC and user-level (max-aggregated) ROC/PR-AUC for
adapted_nll on the shared population, plus score-shift diagnostics by class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def auc_roc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    df = pd.DataFrame({"s": s, "r": ranks}); ranks = df.groupby("s")["r"].transform("mean").to_numpy()
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_pr(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(int(y.sum()), 1)
    # step-wise integration over recall
    d_rec = np.diff(np.concatenate([[0.0], rec]))
    return float((prec * d_rec).sum())


def metrics(df: pd.DataFrame, score_col: str) -> dict:
    y = df["y"].to_numpy(dtype=int); s = df[score_col].to_numpy(dtype=float)
    day = {"day_roc": auc_roc(y, s), "day_pr": auc_pr(y, s)}
    ug = df.groupby("user_id").agg(y=("y", "max"), s=(score_col, "max"))
    day["user_roc"] = auc_roc(ug["y"].to_numpy(dtype=int), ug["s"].to_numpy(dtype=float))
    day["user_pr"] = auc_pr(ug["y"].to_numpy(dtype=int), ug["s"].to_numpy(dtype=float))
    return day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--variant", action="append", required=True,
                    help="name=path/to/example_scores.parquet (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = pd.read_parquet(args.original)
    report = {"population": {"n": int(len(base)), "n_pos": int((base["y"] == 1).sum()),
                             "n_users": int(base["user_id"].nunique())},
              "original": metrics(base, "adapted_nll")}
    for spec in args.variant:
        name, path = spec.split("=", 1)
        v = pd.read_parquet(path)
        merged = base[["example_id", "user_id", "y", "adapted_nll"]].merge(
            v[["example_id", "adapted_nll"]], on="example_id", suffixes=("_orig", "_masked"))
        m = metrics(merged.rename(columns={"adapted_nll_masked": "adapted_nll"}), "adapted_nll")
        shift = merged["adapted_nll_masked"] - merged["adapted_nll_orig"]
        m["nll_shift_pos_mean"] = float(shift[merged["y"] == 1].mean())
        m["nll_shift_ben_mean"] = float(shift[merged["y"] == 0].mean())
        report[name] = m
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
