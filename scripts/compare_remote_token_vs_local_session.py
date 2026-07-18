#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch


def table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def build_fold_map(feature_path: Path, run_root: Path) -> pd.DataFrame:
    meta_all = pd.read_parquet(feature_path, columns=["user_id", "day_index"])
    rows: List[pd.DataFrame] = []
    bundle_paths = sorted(run_root.glob("fold*/model_bundle.pt"))
    if not bundle_paths:
        bundle_paths = sorted(run_root.glob("*/*/model_bundle.pt"))
    for bundle_path in bundle_paths:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        fold = bundle["fold"]
        test_users = set(fold["test_users"])
        test_df = meta_all.loc[meta_all["user_id"].isin(test_users), ["user_id", "day_index"]].copy().reset_index(drop=True)
        test_df["receiver_row_idx"] = test_df.index.astype(int)
        test_df["method"] = str(bundle.get("method", "plain"))
        test_df["fold"] = int(fold["fold"])
        test_df["heldout_pos_user"] = fold["heldout_pos_user"]
        rows.append(test_df)
    return pd.concat(rows, ignore_index=True)


def summarize_local_daylevel(
    best_path: Path,
    fold_map: pd.DataFrame,
    *,
    adaptive: bool,
    top_intervention: str,
    control_intervention: str,
) -> pd.DataFrame:
    best = pd.read_csv(best_path)
    merged = best.merge(
        fold_map,
        on=["method", "fold", "heldout_pos_user", "receiver_row_idx"],
        how="left",
        validate="many_to_one",
    )
    cfg_cols = ["method", "fold", "heldout_pos_user", "context_mode", "target", "intervention", "donor_type", "user_id", "day_index"]
    agg_index = ["method", "context_mode", "target"]
    if adaptive:
        cfg_cols = [
            "method",
            "fold",
            "heldout_pos_user",
            "context_mode",
            "target",
            "rank_metric",
            "adaptive_k",
            "intervention",
            "donor_type",
            "user_id",
            "day_index",
        ]
        agg_index = ["method", "context_mode", "target", "rank_metric", "adaptive_k"]
    day = merged.groupby(cfg_cols, as_index=False).agg(
        mean_day_delta=("delta", "mean"),
        n_sessions=("delta", "size"),
    )
    sumdf = day.groupby(agg_index + ["intervention", "donor_type"], as_index=False).agg(
        n_receivers=("mean_day_delta", "size"),
        mean_best_delta=("mean_day_delta", "mean"),
        mean_sessions_per_receiver=("n_sessions", "mean"),
    )
    wide = sumdf.pivot_table(index=agg_index, columns=["intervention", "donor_type"], values="mean_best_delta")
    wide.columns = [f"{a}_{b}_mean_best_delta" for a, b in wide.columns]
    wide = wide.reset_index()
    nwide = sumdf.pivot_table(index=agg_index, columns=["intervention", "donor_type"], values="n_receivers")
    nwide.columns = [f"{a}_{b}_n_receivers" for a, b in nwide.columns]
    nwide = nwide.reset_index()
    out = wide.merge(nwide, on=agg_index, how="left")
    top_anom = f"{top_intervention}_anomalous_mean_best_delta"
    top_ben = f"{top_intervention}_benign_mean_best_delta"
    ctrl_anom = f"{control_intervention}_anomalous_mean_best_delta"
    ctrl_ben = f"{control_intervention}_benign_mean_best_delta"
    missing = [col for col in [top_anom, top_ben, ctrl_anom, ctrl_ben] if col not in out.columns]
    if missing:
        raise ValueError(
            f"Missing expected local intervention columns in {best_path}: {missing}. "
            f"Available mean-delta columns: {[c for c in out.columns if c.endswith('_mean_best_delta')]}"
        )
    out["top_repair_advantage"] = out[top_anom] - out[top_ben]
    out["control_repair_advantage"] = out[ctrl_anom] - out[ctrl_ben]
    out["top_minus_control_advantage"] = out["top_repair_advantage"] - out["control_repair_advantage"]
    return out.sort_values("top_minus_control_advantage", ascending=False).reset_index(drop=True)


def write_report(
    out_path: Path,
    *,
    adaptive_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    remote_df: pd.DataFrame,
    local_top_intervention: str,
    local_control_intervention: str,
) -> None:
    top_local = adaptive_df.head(10).copy()
    top_resid = residual_df.head(10).copy()
    top_remote = remote_df.head(10).copy()
    lines = [
        "# Remote Token vs Local Session Day-Level Comparison",
        "",
        "Local session repair rows are aggregated to one value per `(user_id, day_index)` receiver by averaging the session-row best deltas within each positive day.",
        "This puts the local session branch on the same receiver unit as the remote token QLoRA branch.",
        f"Local summaries use `intervention={local_top_intervention}` as the target set and `intervention={local_control_intervention}` as the control set.",
        "Remote summaries are read from the provided remote summary CSV and may use a different control-set construction (for example `control5_active`).",
        "Remote summaries may also reflect stricter receiver/donor matching rules such as same-user exclusion, so `n_receivers` can be smaller than the full positive-day pool.",
        "",
        "## Best Local Adaptive Day-Level Rows",
        "",
        table_text(top_local),
        "",
        "## Best Local Residual Day-Level Rows",
        "",
        table_text(top_resid),
        "",
        "## Remote Token Summary",
        "",
        table_text(top_remote),
        "",
    ]
    if not top_local.empty and not top_remote.empty:
        local_best = top_local.iloc[0]
        remote_best = top_remote.iloc[0]
        lines += [
            "## Read",
            "",
            f"- best local adaptive day-level advantage: `{float(local_best['top_minus_control_advantage']):.6f}`",
            f"- best remote token advantage: `{float(remote_best['top_minus_control_advantage']):.6f}`",
            f"- best local residual day-level advantage: `{float(top_resid.iloc[0]['top_minus_control_advantage']):.6f}`" if not top_resid.empty else "- best local residual day-level advantage: `(missing)`",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-feature-path", required=True)
    ap.add_argument("--local-run-root", required=True)
    ap.add_argument("--local-adaptive-best-rows", required=True)
    ap.add_argument("--local-residual-best-rows", required=True)
    ap.add_argument("--remote-summary-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--local-top-intervention", default="top5")
    ap.add_argument("--local-control-intervention", default="control3")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_map = build_fold_map(Path(args.session_feature_path), Path(args.local_run_root))
    adaptive_df = summarize_local_daylevel(
        Path(args.local_adaptive_best_rows),
        fold_map,
        adaptive=True,
        top_intervention=args.local_top_intervention,
        control_intervention=args.local_control_intervention,
    )
    residual_df = summarize_local_daylevel(
        Path(args.local_residual_best_rows),
        fold_map,
        adaptive=False,
        top_intervention=args.local_top_intervention,
        control_intervention=args.local_control_intervention,
    )
    remote_df = pd.read_csv(args.remote_summary_csv).sort_values("top_minus_control_advantage", ascending=False).reset_index(drop=True)

    adaptive_df.to_csv(out_dir / "local_adaptive_daylevel_summary.csv", index=False)
    residual_df.to_csv(out_dir / "local_residual_daylevel_summary.csv", index=False)
    write_report(
        out_dir / "REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md",
        adaptive_df=adaptive_df,
        residual_df=residual_df,
        remote_df=remote_df,
        local_top_intervention=args.local_top_intervention,
        local_control_intervention=args.local_control_intervention,
    )

    print("LOCAL ADAPTIVE TOP ROWS")
    print(adaptive_df.head(10).to_string(index=False))
    print("\nLOCAL RESIDUAL TOP ROWS")
    print(residual_df.head(10).to_string(index=False))
    print("\nREMOTE TOKEN TOP ROWS")
    print(remote_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
