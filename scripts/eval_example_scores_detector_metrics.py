#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auc(y: np.ndarray, score: np.ndarray, kind: str) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    if kind == "pr":
        return float(average_precision_score(y, score))
    if kind == "roc":
        return float(roc_auc_score(y, score))
    raise ValueError(f"unsupported auc kind: {kind}")


def top1pct_recall(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) == 0 or int(y.sum()) == 0:
        return float("nan")
    k = max(1, int(round(0.01 * len(y))))
    idx = np.argsort(score)[::-1][:k]
    return float(y[idx].sum() / y.sum())


def summarize_score(scores: pd.DataFrame, score_name: str, score: np.ndarray, run_name: str) -> Dict[str, object]:
    y = scores["y"].to_numpy(dtype=int)
    row: Dict[str, object] = {
        "run_name": run_name,
        "score_name": score_name,
        "n_eval": int(len(scores)),
        "n_pos_days": int(y.sum()),
        "day_pr_auc": safe_auc(y, score, "pr"),
        "day_roc_auc": safe_auc(y, score, "roc"),
        "day_top1pct_recall": top1pct_recall(y, score),
    }

    user_agg = (
        scores.assign(score=score)
        .groupby("user_id", as_index=False)
        .agg(y=("y", "max"), score=("score", "max"))
    )
    uy = user_agg["y"].to_numpy(dtype=int)
    us = user_agg["score"].to_numpy(dtype=float)
    ranks = user_agg.sort_values("score", ascending=False).reset_index(drop=True)
    pos_ranks = (ranks.index[ranks["y"].to_numpy(dtype=int) == 1] + 1).tolist()
    row.update(
        {
            "n_eval_users": int(len(user_agg)),
            "n_pos_users": int(uy.sum()),
            "user_pr_auc": safe_auc(uy, us, "pr"),
            "user_roc_auc": safe_auc(uy, us, "roc"),
            "user_top1pct_recall": top1pct_recall(uy, us),
            "heldout_user_rank_first_positive": int(pos_ranks[0]) if pos_ranks else None,
        }
    )
    return row


def build_rows(scores: pd.DataFrame, run_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    score_specs = [
        ("adapted_nll", scores["adapted_nll"].to_numpy(dtype=float)),
        ("base_nll", scores["base_nll"].to_numpy(dtype=float)),
        ("neg_delta_nll", -scores["delta_nll"].to_numpy(dtype=float)),
    ]
    for score_name, score in score_specs:
        rows.append(summarize_score(scores, score_name, score, run_name))
    return rows


def write_markdown(df: pd.DataFrame, out_path: Path, *, score_path: Path, split: str) -> None:
    lines = [
        "# Example Scores Detector Metrics",
        "",
        f"- score source: `{score_path}`",
        f"- split filter: `{split}`",
        "",
        df.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-parquet", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    scores = pd.read_parquet(args.scores_parquet)
    if args.split:
        scores = scores.loc[scores["split"] == args.split].copy()
    if scores.empty:
        raise ValueError(f"no rows left after split filter {args.split!r}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(build_rows(scores, args.run_name))
    df.to_csv(out_dir / "detector_metrics.csv", index=False)
    (out_dir / "detector_metrics.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    write_markdown(df, out_dir / "DETECTOR_METRICS.md", score_path=args.scores_parquet, split=args.split)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
