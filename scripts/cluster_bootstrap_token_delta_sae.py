#!/usr/bin/env python3
"""User-level cluster bootstrap for token delta-SAE causal / necessity rows.

The receiver-level bootstrap treats positive user-days as independent units.
Days from the same user share metadata, scenario, and style, so those CIs can
be anti-conservative — especially on r6.2 (70 days from 4 users). This script
recomputes uncertainty with the malicious user as the resampling unit and adds
per-user and leave-one-user-out breakdowns.

Inputs are the existing *_best_rows.csv artifacts; no GPU work is redone.

Estimand per unit (identical to the repaired pooled scripts):
- causal:    (top_anomalous - top_benign) - (control_anomalous - control_benign)
             per complete receiver
- necessity: (top_benign - top_positive) - (control_benign - control_positive)
             per complete pair

Outputs per (layer, latent_mult, k, context_mode, target):
- pooled estimate (must match the existing bootstrap estimate)
- user-cluster bootstrap CI (resample users with replacement, pooled mean over
  the concatenated rows of sampled users)
- mean-of-user-means estimate and its cluster bootstrap CI
- per-user estimates and day counts
- leave-one-user-out estimates
- n_users, min/median/max days per user, sign agreement across users
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

KEY_COLS = ["layer", "latent_mult", "k", "context_mode"]


def table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def user_from_example_id(example_id: str) -> str:
    return str(example_id).split(":", 1)[0]


def unit_metrics(best_df: pd.DataFrame, *, mode: str, top_set: str, control_set: str) -> pd.DataFrame:
    """Return one row per complete unit: KEY_COLS + user + metric."""
    if mode == "causal":
        index_col = "receiver_row_idx"
        col_fields = ["feature_set", "donor_type"]
        arms = ("anomalous", "benign")
    else:
        index_col = "pair_idx"
        col_fields = ["feature_set", "receiver_type"]
        arms = ("benign", "positive")

    out: List[pd.DataFrame] = []
    for key, sub in best_df.groupby(KEY_COLS, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        # unit -> user map
        if mode == "causal":
            user_map = (
                sub.groupby(index_col)["receiver_example_id"].first().map(user_from_example_id)
            )
        else:
            pos = sub[sub["receiver_type"] == "positive"]
            user_map = (
                pos.groupby(index_col)["receiver_example_id"].first().map(user_from_example_id)
            )
        work = sub.pivot_table(index=[index_col], columns=col_fields, values="delta", aggfunc="first")
        work.columns = [f"{a}_{b}" for a, b in work.columns]
        needed = [f"{s}_{a}" for s in (top_set, control_set) for a in arms]
        for col in needed:
            if col not in work.columns:
                work[col] = float("nan")
        metric = (work[f"{top_set}_{arms[0]}"] - work[f"{top_set}_{arms[1]}"]) - (
            work[f"{control_set}_{arms[0]}"] - work[f"{control_set}_{arms[1]}"]
        )
        df = pd.DataFrame({"metric": metric}).dropna()
        df["user_id"] = user_map.reindex(df.index)
        df = df.dropna(subset=["user_id"]).reset_index(drop=True)
        for col_name, value in zip(KEY_COLS, key):
            df[col_name] = value
        out.append(df)
    if not out:
        return pd.DataFrame(columns=["metric", "user_id", *KEY_COLS])
    return pd.concat(out, ignore_index=True)


def cluster_bootstrap(values_by_user: Dict[str, np.ndarray], *, n_bootstrap: int, seed: int) -> Dict[str, Any]:
    users = sorted(values_by_user)
    n_users = len(users)
    all_vals = np.concatenate([values_by_user[u] for u in users])
    user_means = np.array([values_by_user[u].mean() for u in users])
    result: Dict[str, Any] = {
        "n_users": n_users,
        "n_units": int(all_vals.size),
        "pooled_estimate": float(all_vals.mean()),
        "user_mean_estimate": float(user_means.mean()),
        "n_users_positive": int((user_means > 0).sum()),
    }
    if n_users < 2:
        for k in ("cluster_ci_low", "cluster_ci_high", "usermean_ci_low", "usermean_ci_high"):
            result[k] = float("nan")
        return result
    rng = np.random.default_rng(seed)
    pooled_draws = np.empty(n_bootstrap)
    usermean_draws = np.empty(n_bootstrap)
    arrs = [values_by_user[u] for u in users]
    for i in range(n_bootstrap):
        pick = rng.integers(0, n_users, size=n_users)
        sample = np.concatenate([arrs[j] for j in pick])
        pooled_draws[i] = sample.mean()
        usermean_draws[i] = user_means[pick].mean()
    result["cluster_ci_low"] = float(np.quantile(pooled_draws, 0.025))
    result["cluster_ci_high"] = float(np.quantile(pooled_draws, 0.975))
    result["usermean_ci_low"] = float(np.quantile(usermean_draws, 0.025))
    result["usermean_ci_high"] = float(np.quantile(usermean_draws, 0.975))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-rows-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["causal", "necessity"], required=True)
    ap.add_argument("--top-sets", default="top5")
    ap.add_argument("--control-set", default="control5_active")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    best_df = pd.read_csv(args.best_rows_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_sets = [x.strip() for x in args.top_sets.split(",") if x.strip()]
    control_set = args.control_set.strip()

    summary_rows: List[Dict[str, Any]] = []
    per_user_rows: List[Dict[str, Any]] = []
    louo_rows: List[Dict[str, Any]] = []

    for top_set in top_sets:
        units = unit_metrics(best_df, mode=args.mode, top_set=top_set, control_set=control_set)
        for key, sub in units.groupby(KEY_COLS, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(KEY_COLS, [int(key[0]), int(key[1]), int(key[2]), str(key[3])]))
            base["target"] = top_set
            values_by_user = {
                str(u): g["metric"].to_numpy(dtype=float) for u, g in sub.groupby("user_id")
            }
            boot = cluster_bootstrap(
                values_by_user, n_bootstrap=args.n_bootstrap, seed=args.seed + hash(str(key)) % 100000
            )
            day_counts = [len(v) for v in values_by_user.values()]
            summary_rows.append(
                {
                    **base,
                    **boot,
                    "days_per_user_min": int(min(day_counts)) if day_counts else 0,
                    "days_per_user_median": float(np.median(day_counts)) if day_counts else 0,
                    "days_per_user_max": int(max(day_counts)) if day_counts else 0,
                }
            )
            for u, vals in sorted(values_by_user.items()):
                per_user_rows.append(
                    {**base, "user_id": u, "n_units": int(vals.size), "user_estimate": float(vals.mean())}
                )
            users = sorted(values_by_user)
            for held_out in users:
                rest = np.concatenate([values_by_user[u] for u in users if u != held_out])
                louo_rows.append(
                    {
                        **base,
                        "held_out_user": held_out,
                        "estimate_without_user": float(rest.mean()) if rest.size else float("nan"),
                        "held_out_user_estimate": float(values_by_user[held_out].mean()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["pooled_estimate"], ascending=False
    ).reset_index(drop=True)
    per_user_df = pd.DataFrame(per_user_rows)
    louo_df = pd.DataFrame(louo_rows)

    summary_df.to_csv(out_dir / "cluster_bootstrap_summary.csv", index=False)
    per_user_df.to_csv(out_dir / "cluster_bootstrap_per_user.csv", index=False)
    louo_df.to_csv(out_dir / "cluster_bootstrap_louo.csv", index=False)

    small = summary_df["n_users"].min() if not summary_df.empty else 0
    lines = [
        f"# User-Level Cluster Bootstrap ({args.mode})",
        "",
        f"Source rows: `{args.best_rows_csv}`",
        f"Control set: `{control_set}`. Bootstrap draws: `{args.n_bootstrap}` (user-level resampling).",
        "",
        "The `pooled_estimate` column must match the receiver/pair-level bootstrap",
        "estimate for the same configuration; the cluster CIs replace the",
        "day-level CIs as the honest uncertainty statement.",
        "",
    ]
    if small and small < 8:
        lines += [
            f"NOTE: only {int(small)} malicious users in at least one configuration.",
            "Percentile CIs from so few clusters are unstable; per-user estimates,",
            "sign agreement (`n_users_positive` / `n_users`), and leave-one-user-out",
            "results are the more informative robustness statements.",
            "",
        ]
    lines += ["## Summary", "", table_text(summary_df), "", "## Per-user estimates", "", table_text(per_user_df), "", "## Leave-one-user-out", "", table_text(louo_df), ""]
    (out_dir / "CLUSTER_BOOTSTRAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(table_text(summary_df))


if __name__ == "__main__":
    main()
