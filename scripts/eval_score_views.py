#!/usr/bin/env python3
"""Evaluate frozen score views from a token-class decomposition cache.

Views (definitions imported from `token_class_decomposition`, never redefined
here) are evaluated on *identical* examples, the same frozen adapter and the
same full input context. Profiles remain in context for every view; a
behaviour-only view removes the direct profile contribution to the score, not
the influence of profile context on behavioural predictions.

Populations reuse `eval_fold_aligned_detector_metrics.make_folds`, so the fold
construction (seed, held-out malicious user, benign cohort) is bit-identical to
the published protocol.

Reported per view: day-level ROC/AP, user-level ROC/AP after max-aggregation,
held-out user rank, within-user ROC for the held-out malicious user, day recall
at a prespecified false-positive budget, population sizes and prevalence.
Uncertainty is a paired cluster bootstrap over the malicious users; with four
of them it is descriptive and conditional on these adapters and this benign
cohort, and it says nothing about training-seed variability.

Also reports the exact mean-score-gap decomposition

    E_pos[s] - E_neg[s] = sum_c ( E_pos[w_c s_c] - E_neg[w_c s_c] ),
    w_c s_c = loss_sum_c / n_targets  (per example),

which is an accounting identity for the mean gap. It is NOT an AUC
decomposition and not a causal claim: class contributions are signed and can
cancel, so a single class's share can exceed the net gap or be negative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_fold_aligned_detector_metrics import make_folds
from token_class_decomposition import CERT_VIEWS, LANL_VIEWS

VIEW_SETS = {"cert": CERT_VIEWS, "lanl": LANL_VIEWS}
# Prespecified before looking at any result.
FPR_BUDGETS = (0.001, 0.01)
PRIMARY_VIEWS = ("full", "profile_only", "behavior_only")


def view_series(df: pd.DataFrame, classes: Sequence[str]) -> pd.Series:
    loss = sum(df[f"loss_sum_{c}"] for c in classes if f"loss_sum_{c}" in df.columns)
    n = sum(df[f"n_{c}"] for c in classes if f"n_{c}" in df.columns)
    return (loss / n.replace(0, np.nan)).astype(float)


def safe_auc(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s) if kind == "pr" else roc_auc_score(y, s))


def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """Recall when the alert threshold admits `fpr` of the negatives."""
    m = np.isfinite(s)
    y, s = y[m], s[m]
    neg = s[y == 0]
    if neg.size == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    return float((s[y == 1] > thr).mean())


def fold_metrics(test: pd.DataFrame, s: np.ndarray, heldout_user: str) -> Dict[str, float]:
    y = test["y"].to_numpy(dtype=int)
    row: Dict[str, float] = {
        "n_test_rows": int(len(test)),
        "n_test_pos_rows": int(y.sum()),
        "day_prevalence": float(y.mean()),
        "day_roc_auc": safe_auc(y, s, "roc"),
        "day_pr_auc": safe_auc(y, s, "pr"),
    }
    for b in FPR_BUDGETS:
        row[f"day_recall_at_fpr_{b}"] = recall_at_fpr(y, s, b)

    u = pd.DataFrame({"user_id": test["user_id"].to_numpy(), "y": y, "score": s})
    agg = u.groupby("user_id").agg(y=("y", "max"), score=("score", "max")).reset_index()
    uy = agg["y"].to_numpy(dtype=int)
    us = agg["score"].to_numpy(dtype=float)
    ranks = agg.sort_values("score", ascending=False).reset_index(drop=True)
    pos_rank = (ranks.index[ranks["y"].to_numpy(dtype=int) == 1] + 1).tolist()
    k = max(1, int(round(0.01 * len(agg))))
    row.update(
        {
            "n_test_users": int(len(agg)),
            "user_roc_auc": safe_auc(uy, us, "roc"),
            "user_pr_auc": safe_auc(uy, us, "pr"),
            "heldout_user_rank": int(pos_rank[0]) if pos_rank else -1,
            "user_top1pct_recall": float(1.0 if pos_rank and pos_rank[0] <= k else 0.0),
            "user_top1pct_k": int(k),
        }
    )
    own = test.loc[test["user_id"] == heldout_user]
    oy = own["y"].to_numpy(dtype=int)
    os_ = s[test["user_id"].to_numpy() == heldout_user]
    row["within_user_roc"] = safe_auc(oy, os_, "roc")
    row["within_user_n_pos"] = int(oy.sum())
    row["within_user_n_days"] = int(len(own))
    return row


def cluster_bootstrap(per_fold: Dict[str, List[float]], n_draws: int, seed: int):
    """Returns (per-view CI dict, the shared draw indices, the per-view arrays)."""
    names = sorted(per_fold)
    n = len(next(iter(per_fold.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    out: Dict[str, Dict[str, float]] = {}
    arrs = {k: np.asarray(v, dtype=float) for k, v in per_fold.items()}
    for k in names:
        draws = np.nanmean(arrs[k][idx], axis=1)
        out[k] = {
            "mean": float(np.nanmean(arrs[k])),
            "lo": float(np.nanpercentile(draws, 2.5)),
            "hi": float(np.nanpercentile(draws, 97.5)),
        }
    return out, idx, arrs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-scores", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--schema", choices=sorted(VIEW_SETS), default="cert")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benign-test-users", type=int, default=800)
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    ap.add_argument("--reference-scores", type=Path, default=None,
                    help="cached example_scores.parquet; adds a 'full_cached' view to reproduce the published baseline")
    args = ap.parse_args()

    views = VIEW_SETS[args.schema]
    df = pd.read_parquet(args.class_scores)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge the cached baseline BEFORE building any score array, so every array
    # is taken from one dataframe in one row order
    if args.reference_scores is not None and Path(args.reference_scores).exists():
        ref = pd.read_parquet(args.reference_scores)[["example_idx", "adapted_nll"]]
        n_before = len(df)
        df = df.merge(
            ref.rename(columns={"adapted_nll": "adapted_nll_cached"}), on="example_idx", how="left"
        )
        if len(df) != n_before:
            raise RuntimeError(f"reference merge changed row count {n_before} -> {len(df)}")
        views = dict(views)
        views["full_cached"] = ()
    df = df.reset_index(drop=True)

    score_cols: Dict[str, np.ndarray] = {}
    for name, classes in views.items():
        if name == "full_cached":
            score_cols[name] = df["adapted_nll_cached"].to_numpy(dtype=float)
            continue
        df[f"view_{name}"] = view_series(df, classes)
        score_cols[name] = df[f"view_{name}"].to_numpy(dtype=float)

    folds = make_folds(df[["user_id", "y"]].drop_duplicates(), args.seed, args.benign_test_users)

    rows: List[Dict[str, object]] = []
    for fold in folds:
        test_users = sorted(str(x) for x in fold["test_users"])
        mask = df["user_id"].isin(test_users).to_numpy()
        test = df.loc[mask].copy()
        for name in views:
            s = score_cols[name][mask]
            r = fold_metrics(test, s, str(fold["heldout_pos_user"]))
            r.update(
                {
                    "run_name": args.run_name,
                    "view": name,
                    "fold": int(fold["fold"]),
                    "heldout_pos_user": str(fold["heldout_pos_user"]),
                }
            )
            rows.append(r)
    per_fold_df = pd.DataFrame(rows)
    per_fold_df.to_csv(out_dir / "score_view_folds.csv", index=False)

    metrics = [
        "day_roc_auc", "day_pr_auc", "user_roc_auc", "user_pr_auc",
        "heldout_user_rank", "within_user_roc", "user_top1pct_recall",
    ] + [f"day_recall_at_fpr_{b}" for b in FPR_BUDGETS]

    summary_rows: List[Dict[str, object]] = []
    boot_store: Dict[str, Dict[str, np.ndarray]] = {}
    for metric in metrics:
        per_view = {
            name: per_fold_df.loc[per_fold_df["view"] == name].sort_values("fold")[metric].tolist()
            for name in views
        }
        ci, idx, arrs = cluster_bootstrap(per_view, args.bootstrap_draws, args.seed)
        boot_store[metric] = {"idx": idx, "arrs": arrs}
        for name in views:
            summary_rows.append(
                {"view": name, "metric": metric, "mean": ci[name]["mean"],
                 "ci_lo": ci[name]["lo"], "ci_hi": ci[name]["hi"]}
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "score_view_summary.csv", index=False)

    # paired contrasts against the full view
    contrasts: List[Dict[str, object]] = []
    for metric in metrics:
        idx = boot_store[metric]["idx"]
        arrs = boot_store[metric]["arrs"]
        base = arrs["full"]
        for name in views:
            if name == "full":
                continue
            diff = np.nanmean(arrs[name][idx], axis=1) - np.nanmean(base[idx], axis=1)
            point = float(np.nanmean(arrs[name]) - np.nanmean(base))
            contrasts.append(
                {
                    "metric": metric,
                    "view": name,
                    "vs": "full",
                    "delta": point,
                    "ci_lo": float(np.nanpercentile(diff, 2.5)),
                    "ci_hi": float(np.nanpercentile(diff, 97.5)),
                    "tail_frac_not_better": float(np.mean(diff <= 0)),
                    "per_fold_delta": [
                        float(a - b) for a, b in zip(arrs[name], base)
                    ],
                }
            )
    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(out_dir / "score_view_contrasts.csv", index=False)

    # ---- exact mean-score-gap decomposition on the union of fold test rows
    class_cols = [c[len("loss_sum_"):] for c in df.columns
                  if c.startswith("loss_sum_") and not c.startswith("loss_sum_total")
                  and not c.startswith("base_")]
    gap_rows: List[Dict[str, object]] = []
    for fold in folds:
        test_users = sorted(str(x) for x in fold["test_users"])
        test = df.loc[df["user_id"].isin(test_users)]
        pos = test.loc[test["y"] == 1]
        neg = test.loc[test["y"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        total_gap = float(pos["view_full"].mean() - neg["view_full"].mean())
        for c in class_cols:
            contrib = test[f"loss_sum_{c}"] / test["n_targets"]
            gp = float(contrib[test["y"] == 1].mean())
            gn = float(contrib[test["y"] == 0].mean())
            gap_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "heldout_pos_user": str(fold["heldout_pos_user"]),
                    "class": c,
                    "pos_mean_weighted_contrib": gp,
                    "neg_mean_weighted_contrib": gn,
                    "delta_contrib": gp - gn,
                    "total_mean_gap": total_gap,
                    "share_of_total_gap": (gp - gn) / total_gap if total_gap != 0 else float("nan"),
                }
            )
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(out_dir / "mean_gap_decomposition.csv", index=False)
    recon_err = float("nan")
    if len(gap_df):
        chk = gap_df.groupby("fold").agg(s=("delta_contrib", "sum"), t=("total_mean_gap", "first"))
        recon_err = float((chk["s"] - chk["t"]).abs().max())

    meta = {
        "run_name": args.run_name,
        "class_scores": str(args.class_scores),
        "schema": args.schema,
        "views": {k: list(v) for k, v in views.items()},
        "fold_seed": args.seed,
        "benign_test_users_requested": args.benign_test_users,
        "n_folds": len(folds),
        "n_examples": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_positive_users": int(df.loc[df["y"] == 1, "user_id"].nunique()),
        "n_positive_rows": int((df["y"] == 1).sum()),
        "split_counts": df["split"].value_counts().to_dict(),
        "fpr_budgets": list(FPR_BUDGETS),
        "bootstrap": "paired cluster bootstrap over malicious users (folds); descriptive at n=4; conditional on these adapters and this benign cohort; not training-seed replication",
        "mean_gap_decomposition_max_recon_err": recon_err,
        "population_note": "every user in this pool was excluded from adapter training (positives by construction; benign comparison users are validation-split). This is a user-disjoint comparison, not a seen-user detection test.",
    }
    (out_dir / "score_view_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    piv = summary_df.loc[summary_df["metric"].isin(
        ["day_roc_auc", "day_pr_auc", "user_roc_auc", "within_user_roc", "heldout_user_rank"]
    )].pivot(index="view", columns="metric", values="mean")
    order = [v for v in PRIMARY_VIEWS if v in piv.index] + [v for v in piv.index if v not in PRIMARY_VIEWS]
    print(f"\n=== {args.run_name}: mean over {len(folds)} folds ===")
    print(piv.loc[order].to_string(float_format=lambda x: f"{x:8.4f}"))
    print(f"\nmean-gap decomposition max reconstruction error: {recon_err:.3e}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
