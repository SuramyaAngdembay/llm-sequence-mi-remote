#!/usr/bin/env python3
"""Evaluate frozen score views on LANL, under LANL's own protocol.

CERT's leave-one-malicious-user-out folds do not apply here: LANL is scored as
window-level ranking inside two pools that `scripts/lanl/lanl_split.py` marks
in `eval.jsonl` itself —

  seen   : windows from users in md5 folds 0-3 (whose benign windows fed training)
  unseen : windows from users in fold 4 (leave-users-out)

so the seen/unseen membership is read from the data, never recomputed, and
matches the published `results/lanl_auth_replication/ap_*.json` numbers.

Views come from `token_class_decomposition.LANL_VIEWS`, which is field-level,
not line-level: `su`/`du` -> ID_USER, `sc`/`dc` -> ID_HOST, `at`/`lt`/`or`/`res`
-> BEHAV, a bare `t<hh>` -> HOUR, and the `" | "` event separators -> OTHER
(scored as non-identity by the behaviour view).

As on CERT, every view is scored on identical windows with the same frozen
adapter and the full input context: identity fields stay in the context, and a
behaviour-only view removes only their direct contribution to the score.

Uncertainty is a cluster bootstrap over **users** within each pool, with the
same draws reused across views so contrasts are paired.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from token_class_decomposition import LANL_VIEWS

FPR_BUDGETS = (0.001, 0.01)


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
    m = np.isfinite(s)
    y, s = y[m], s[m]
    neg = s[y == 0]
    if neg.size == 0 or (y == 1).sum() == 0:
        return float("nan")
    return float((s[y == 1] > np.quantile(neg, 1.0 - fpr)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-scores", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="dir holding eval.jsonl with fold/seen fields")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reference-scores", type=Path, default=None)
    ap.add_argument("--published-ap-json", type=Path, default=None)
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.class_scores)

    # seen/unseen straight from the data, joined on example_id
    marks: List[Dict[str, object]] = []
    with (args.data_dir / "eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            marks.append({"example_id": r["example_id"], "seen": int(r["seen"]), "fold": int(r["fold"])})
    mark_df = pd.DataFrame(marks)
    n_before = len(df)
    df = df.merge(mark_df, on="example_id", how="left")
    if len(df) != n_before:
        raise RuntimeError(f"seen/fold merge changed row count {n_before} -> {len(df)}")
    if df["seen"].isna().any():
        raise RuntimeError(f"{int(df['seen'].isna().sum())} scored windows carry no seen/fold mark")

    views = dict(LANL_VIEWS)
    if args.reference_scores is not None and Path(args.reference_scores).exists():
        ref = pd.read_parquet(args.reference_scores)[["example_idx", "adapted_nll"]]
        df = df.merge(ref.rename(columns={"adapted_nll": "adapted_nll_cached"}), on="example_idx", how="left")
        views["full_cached"] = ()
    df = df.reset_index(drop=True)

    score_cols: Dict[str, np.ndarray] = {}
    for name, classes in views.items():
        if name == "full_cached":
            score_cols[name] = df["adapted_nll_cached"].to_numpy(dtype=float)
            continue
        df[f"view_{name}"] = view_series(df, classes)
        score_cols[name] = df[f"view_{name}"].to_numpy(dtype=float)

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []
    boot: Dict[str, Dict[str, np.ndarray]] = {}
    for pool_name, pool_mask in (("seen", df["seen"] == 1), ("unseen", df["seen"] == 0)):
        sub = df.loc[pool_mask]
        idx = np.flatnonzero(pool_mask.to_numpy())
        y = sub["y"].to_numpy(dtype=int)
        users = sub["user_id"].to_numpy()
        uniq = np.unique(users)
        # one shared set of user resamples per pool, reused for every view
        draws = rng.integers(0, len(uniq), size=(args.bootstrap_draws, len(uniq)))
        by_user = {u: np.flatnonzero(users == u) for u in uniq}
        for name in views:
            s = score_cols[name][idx]
            row = {
                "run_name": args.run_name,
                "pool": pool_name,
                "view": name,
                "n_windows": int(len(sub)),
                "n_positive": int(y.sum()),
                "prevalence": float(y.mean()),
                "n_users": int(len(uniq)),
                "auc": safe_auc(y, s, "roc"),
                "ap": safe_auc(y, s, "pr"),
            }
            for b in FPR_BUDGETS:
                row[f"recall_at_fpr_{b}"] = recall_at_fpr(y, s, b)
            aucs = np.empty(args.bootstrap_draws)
            for d in range(args.bootstrap_draws):
                sel = np.concatenate([by_user[uniq[j]] for j in draws[d]])
                aucs[d] = safe_auc(y[sel], s[sel], "roc")
            row["auc_ci_lo"] = float(np.nanpercentile(aucs, 2.5))
            row["auc_ci_hi"] = float(np.nanpercentile(aucs, 97.5))
            boot.setdefault(pool_name, {})[name] = aucs
            rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "lanl_score_view_summary.csv", index=False)

    contrasts: List[Dict[str, object]] = []
    for pool_name in ("seen", "unseen"):
        base = boot[pool_name]["full"]
        base_pt = float(res.loc[(res["pool"] == pool_name) & (res["view"] == "full"), "auc"].iloc[0])
        for name in views:
            if name == "full":
                continue
            pt = float(res.loc[(res["pool"] == pool_name) & (res["view"] == name), "auc"].iloc[0])
            d = boot[pool_name][name] - base
            contrasts.append(
                {
                    "pool": pool_name,
                    "view": name,
                    "vs": "full",
                    "delta_auc": pt - base_pt,
                    "ci_lo": float(np.nanpercentile(d, 2.5)),
                    "ci_hi": float(np.nanpercentile(d, 97.5)),
                    "tail_frac_not_better": float(np.mean(d <= 0)),
                }
            )
    pd.DataFrame(contrasts).to_csv(out_dir / "lanl_score_view_contrasts.csv", index=False)

    meta: Dict[str, object] = {
        "run_name": args.run_name,
        "class_scores": str(args.class_scores),
        "data_dir": str(args.data_dir),
        "views": {k: list(v) for k, v in views.items()},
        "n_windows": int(len(df)),
        "seen_windows": int((df["seen"] == 1).sum()),
        "unseen_windows": int((df["seen"] == 0).sum()),
        "bootstrap": "cluster bootstrap over users within each pool; identical draws across views",
        "protocol_note": "seen/unseen membership read from eval.jsonl (md5 leave-users-out, fold 4 unseen); never recomputed",
        "scope_note": "identity fields remain in the context for every view; the behaviour view removes only their direct score contribution",
    }
    if args.published_ap_json is not None and Path(args.published_ap_json).exists():
        pub = json.loads(Path(args.published_ap_json).read_text())
        meta["published"] = pub
        got = res.loc[res["view"] == "full"].set_index("pool")
        meta["reproduces_published_full"] = {
            "seen_auc": [pub.get("seen_auc"), round(float(got.loc["seen", "auc"]), 4)],
            "unseen_auc": [pub.get("unseen_auc"), round(float(got.loc["unseen", "auc"]), 4)],
            "seen_ap": [pub.get("seen_ap"), round(float(got.loc["seen", "ap"]), 4)],
            "unseen_ap": [pub.get("unseen_ap"), round(float(got.loc["unseen", "ap"]), 4)],
        }
    (out_dir / "lanl_score_view_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    piv = res.pivot(index="view", columns="pool", values=["auc", "ap"])
    print(f"\n=== {args.run_name}: LANL window-level, seen vs unseen ===")
    print(piv.to_string(float_format=lambda x: f"{x:8.4f}"))
    if "reproduces_published_full" in meta:
        print("\npublished vs recomputed (full view):", json.dumps(meta["reproduces_published_full"]))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
