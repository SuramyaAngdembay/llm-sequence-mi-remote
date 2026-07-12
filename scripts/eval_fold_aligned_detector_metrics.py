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


def top1_recall(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) == 0 or int(y.sum()) == 0:
        return float("nan")
    k = max(1, int(round(0.01 * len(y))))
    idx = np.argsort(score)[::-1][:k]
    return float(y[idx].sum() / y.sum())


def make_folds(df: pd.DataFrame, seed: int, benign_test_users: int) -> List[Dict[str, object]]:
    pos_users = sorted(df.loc[df["y"] == 1, "user_id"].unique().tolist())
    all_users = np.asarray(sorted(df["user_id"].unique().tolist()))
    benign_users = np.asarray(sorted(set(all_users) - set(pos_users)))
    folds: List[Dict[str, object]] = []
    for i, heldout_pos in enumerate(pos_users):
        local_rng = np.random.default_rng(seed + 1000 + i)
        n_test = min(benign_test_users, len(benign_users))
        test_benign = local_rng.choice(benign_users, size=n_test, replace=False)
        test_users = set(test_benign.tolist()) | {heldout_pos}
        folds.append(
            {
                "fold": i,
                "heldout_pos_user": heldout_pos,
                "test_users": test_users,
            }
        )
    return folds


def summarize_scores(test_df: pd.DataFrame, score: np.ndarray, score_name: str, fold: Dict[str, object], run_name: str) -> Dict[str, object]:
    y = test_df["y"].to_numpy(dtype=int)
    row = {
        "run_name": run_name,
        "score_name": score_name,
        "fold": int(fold["fold"]),
        "heldout_pos_user": str(fold["heldout_pos_user"]),
        "n_test_rows": int(len(test_df)),
        "n_test_pos_rows": int(y.sum()),
        "day_pr_auc": safe_auc(y, score, "pr"),
        "day_roc_auc": safe_auc(y, score, "roc"),
        "day_top1_recall": top1_recall(y, score),
    }
    user_df = pd.DataFrame({"user_id": test_df["user_id"].to_numpy(), "y": y, "score": score})
    user_agg = user_df.groupby("user_id").agg(y=("y", "max"), score=("score", "max")).reset_index()
    uy = user_agg["y"].to_numpy(dtype=int)
    us = user_agg["score"].to_numpy(dtype=float)
    ranks = user_agg.sort_values("score", ascending=False).reset_index(drop=True)
    pos_ranks = (ranks.index[ranks["y"].to_numpy(dtype=int) == 1] + 1).tolist()
    row.update(
        {
            "n_test_users": int(len(user_agg)),
            "n_test_pos_users": int(uy.sum()),
            "user_pr_auc": safe_auc(uy, us, "pr"),
            "user_roc_auc": safe_auc(uy, us, "roc"),
            "user_top1pct_recall": top1_recall(uy, us),
            "heldout_user_rank": int(pos_ranks[0]) if pos_ranks else None,
        }
    )
    return row


def write_markdown(summary: pd.DataFrame, details: pd.DataFrame, out_path: Path, *, score_path: Path, run_name: str, seed: int, benign_test_users: int) -> None:
    lines = [
        "# Fold-Aligned Remote Detector Metrics",
        "",
        f"- score source: `{score_path}`",
        f"- run name: `{run_name}`",
        f"- fold seed: `{seed}`",
        f"- benign test users per fold: `{benign_test_users}`",
        "- protocol: fixed benign-trained remote model, evaluated on the same leave-one-malicious-user-out test-user construction as the local detector baselines",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Fold Rows",
        "",
        details.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-parquet", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benign-test-users", type=int, default=800)
    ap.add_argument("--fold-limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    scores = pd.read_parquet(args.scores_parquet)
    required = {"user_id", "y", "adapted_nll", "base_nll", "delta_nll"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"missing required score columns: {missing}")

    folds = make_folds(scores[["user_id", "y"]].drop_duplicates(), args.seed, args.benign_test_users)
    if args.fold_limit > 0:
        folds = folds[: args.fold_limit]
    if not folds:
        raise ValueError("no folds constructed")

    detail_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    for fold in folds:
        test_users = sorted(str(x) for x in fold["test_users"])
        test_df = scores.loc[scores["user_id"].isin(test_users)].copy()
        for user_id in test_users:
            manifest_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "heldout_pos_user": str(fold["heldout_pos_user"]),
                    "test_user_id": user_id,
                    "is_heldout_positive_user": int(user_id == str(fold["heldout_pos_user"])),
                }
            )
        score_specs = [
            ("adapted_nll", test_df["adapted_nll"].to_numpy(dtype=float)),
            ("base_nll", test_df["base_nll"].to_numpy(dtype=float)),
            ("neg_delta_nll", -test_df["delta_nll"].to_numpy(dtype=float)),
        ]
        for score_name, score in score_specs:
            detail_rows.append(summarize_scores(test_df, score, score_name, fold, args.run_name))

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby("score_name")
        .agg(
            folds=("fold", "count"),
            day_pr_auc_mean=("day_pr_auc", "mean"),
            day_roc_auc_mean=("day_roc_auc", "mean"),
            day_top1_recall_mean=("day_top1_recall", "mean"),
            user_pr_auc_mean=("user_pr_auc", "mean"),
            user_roc_auc_mean=("user_roc_auc", "mean"),
            user_top1pct_recall_mean=("user_top1pct_recall", "mean"),
            heldout_user_rank_mean=("heldout_user_rank", "mean"),
            heldout_user_rank_median=("heldout_user_rank", "median"),
        )
        .reset_index()
        .sort_values(["user_pr_auc_mean", "day_pr_auc_mean"], ascending=False)
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(out_dir / "fold_aligned_detector_rows.csv", index=False)
    summary_df.to_csv(out_dir / "fold_aligned_detector_summary.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "fold_aligned_test_users.csv", index=False)
    (out_dir / "fold_aligned_detector_summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    write_markdown(
        summary_df,
        detail_df,
        out_dir / "FOLD_ALIGNED_DETECTOR_REPORT.md",
        score_path=args.scores_parquet,
        run_name=args.run_name,
        seed=args.seed,
        benign_test_users=args.benign_test_users,
    )

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
