#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def summarize_best(best_df: pd.DataFrame, top_sets: List[str], control_set: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if best_df.empty:
        return pd.DataFrame()
    key_cols = ["layer", "latent_mult", "k", "context_mode"]
    for key, sub in best_df.groupby(key_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        layer, latent_mult, k, context_mode = key
        work = sub.pivot_table(
            index=["pair_idx"],
            columns=["feature_set", "receiver_type"],
            values="delta",
            aggfunc="first",
        )
        work.columns = [f"{a}_{b}" for a, b in work.columns]
        work = work.reset_index()
        for top_set in top_sets:
            needed = [f"{top_set}_positive", f"{top_set}_benign", f"{control_set}_positive", f"{control_set}_benign"]
            row = {
                "layer": int(layer),
                "latent_mult": int(latent_mult),
                "k": int(k),
                "context_mode": str(context_mode),
                "target": str(top_set),
                "n_pairs": int(len(work)),
            }
            for col in needed:
                if col not in work.columns:
                    work[col] = float("nan")
            complete = work.dropna(subset=needed)
            row["n_complete_pairs"] = int(len(complete))
            row["top_positive_mean_best_delta"] = float(complete[f"{top_set}_positive"].mean())
            row["top_benign_mean_best_delta"] = float(complete[f"{top_set}_benign"].mean())
            row["control_positive_mean_best_delta"] = float(complete[f"{control_set}_positive"].mean())
            row["control_benign_mean_best_delta"] = float(complete[f"{control_set}_benign"].mean())
            row["top_necessity_advantage"] = float((complete[f"{top_set}_benign"] - complete[f"{top_set}_positive"]).mean())
            row["control_necessity_advantage"] = float(
                (complete[f"{control_set}_benign"] - complete[f"{control_set}_positive"]).mean()
            )
            row["top_minus_control_necessity"] = float(
                (
                    (complete[f"{top_set}_benign"] - complete[f"{top_set}_positive"])
                    - (complete[f"{control_set}_benign"] - complete[f"{control_set}_positive"])
                ).mean()
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["top_minus_control_necessity", "top_necessity_advantage", "context_mode", "target"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def bootstrap_metric(
    work: pd.DataFrame,
    *,
    top_set: str,
    control_set: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    needed = [f"{top_set}_positive", f"{top_set}_benign", f"{control_set}_positive", f"{control_set}_benign"]
    for col in needed:
        if col not in work.columns:
            work[col] = float("nan")
    metric = (
        (work[f"{top_set}_benign"] - work[f"{top_set}_positive"])
        - (work[f"{control_set}_benign"] - work[f"{control_set}_positive"])
    ).to_numpy(dtype=float)
    metric = metric[np.isfinite(metric)]
    if metric.size == 0:
        return {"n_complete_pairs": 0, "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    n = metric.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        draws[i] = float(metric[idx].mean())
    return {
        "n_complete_pairs": int(metric.size),
        "estimate": float(metric.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def write_report(summary_df: pd.DataFrame, out_path: Path, *, control_set: str, n_bootstrap: int) -> None:
    lines = [
        "# Token Delta SAE Necessity Bootstrap Stats",
        "",
        f"Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `{control_set}`. Bootstrap draws: `{n_bootstrap}`.",
        "",
        "## Summary",
        "",
        table_text(summary_df),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-rows-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-sets", default="top1,top3,top5")
    ap.add_argument("--control-set", default="control5_active")
    ap.add_argument("--n-bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    best_df = pd.read_csv(args.best_rows_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_sets = [x.strip() for x in args.top_sets.split(",") if x.strip()]
    control_set = args.control_set.strip()

    raw_summary = summarize_best(best_df, top_sets=top_sets, control_set=control_set)
    rows: List[Dict[str, Any]] = []
    for key, sub in best_df.groupby(["layer", "latent_mult", "k", "context_mode"], sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        layer, latent_mult, k, context_mode = key
        work = sub.pivot_table(
            index=["pair_idx"],
            columns=["feature_set", "receiver_type"],
            values="delta",
            aggfunc="first",
        )
        work.columns = [f"{a}_{b}" for a, b in work.columns]
        work = work.reset_index()
        for top_set in top_sets:
            boot = bootstrap_metric(
                work.copy(),
                top_set=top_set,
                control_set=control_set,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + int(layer) * 1000 + int(latent_mult) * 100 + int(k) * 10,
            )
            match = raw_summary[
                (raw_summary["layer"] == int(layer))
                & (raw_summary["latent_mult"] == int(latent_mult))
                & (raw_summary["k"] == int(k))
                & (raw_summary["context_mode"] == str(context_mode))
                & (raw_summary["target"] == str(top_set))
            ]
            row = {
                "layer": int(layer),
                "latent_mult": int(latent_mult),
                "k": int(k),
                "context_mode": str(context_mode),
                "target": str(top_set),
                "n_pairs": int(len(work)),
                **boot,
            }
            if not match.empty:
                src = match.iloc[0].to_dict()
                row.update(
                    {
                        "top_positive_mean_best_delta": float(src["top_positive_mean_best_delta"]),
                        "top_benign_mean_best_delta": float(src["top_benign_mean_best_delta"]),
                        "control_positive_mean_best_delta": float(src["control_positive_mean_best_delta"]),
                        "control_benign_mean_best_delta": float(src["control_benign_mean_best_delta"]),
                        "top_necessity_advantage": float(src["top_necessity_advantage"]),
                        "control_necessity_advantage": float(src["control_necessity_advantage"]),
                    }
                )
            rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        ["estimate", "ci_low", "context_mode", "target"], ascending=[False, False, True, True]
    ).reset_index(drop=True)
    summary_df.to_csv(out_dir / "token_delta_sae_necessity_bootstrap_summary.csv", index=False)
    write_report(
        summary_df,
        out_dir / "TOKEN_DELTA_SAE_NECESSITY_BOOTSTRAP_REPORT.md",
        control_set=control_set,
        n_bootstrap=args.n_bootstrap,
    )
    print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
