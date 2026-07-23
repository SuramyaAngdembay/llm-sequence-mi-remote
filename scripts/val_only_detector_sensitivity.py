#!/usr/bin/env python3
"""Detector sensitivity to benign train/test overlap.

The session LM's training corpus covers ~90% of benign users, so most benign
test users in the fold-aligned benchmark were seen during adaptation. This
script recomputes day- and user-level ROC/PR for adapted_nll on the cached
full-population scores under two benign populations:
  (a) all benign users (the fold-aligned convention), and
  (b) only validation-split benign users (never seen in training).
If metrics are similar, the overlap is not driving the detector numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compare_masked_scores import auc_roc, auc_pr, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-parquet", type=Path, required=True)
    ap.add_argument("--all-jsonl", type=Path, default=None,
                    help="needed only if the parquet lacks a split column")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.scores_parquet)
    if "split" not in df.columns:
        if args.all_jsonl is None:
            raise SystemExit("parquet lacks split column; provide --all-jsonl")
        import json as _json
        split_map = {}
        with args.all_jsonl.open() as f:
            for line in f:
                ex = _json.loads(line)
                split_map[ex["example_id"]] = ex["split"]
        df["split"] = df["example_id"].map(split_map)

    pos = df[df["y"] == 1]
    report = {"n_rows": int(len(df)), "n_pos": int(len(pos)),
              "n_benign_users_all": int(df.loc[df["y"] == 0, "user_id"].nunique()),
              "n_benign_users_val": int(df.loc[(df["y"] == 0) & (df["split"] == "val"), "user_id"].nunique())}

    all_pop = df[(df["y"] == 1) | (df["y"] == 0)]
    report["all_benign"] = metrics(all_pop, "adapted_nll")
    val_pop = pd.concat([pos, df[(df["y"] == 0) & (df["split"] == "val")]])
    report["val_only_benign"] = metrics(val_pop, "adapted_nll")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
