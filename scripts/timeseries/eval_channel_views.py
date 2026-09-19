#!/usr/bin/env python3
"""Evaluate channel-class score views for one or more numerical detectors,
through the shared metric core, on ONE common set of eligible examples.

Why several models in a single invocation: the corrected forecaster cannot
score a user's first observation (no strictly earlier record), so it is
eligible on fewer rows than the reconstruction models. Comparing models scored
on different rows is exactly what produced the earlier mismatch. This script
therefore intersects every model's eligibility, evaluates all of them on that
one set, and records the set and its provenance.

Populations (`--population`):
  matched            all days of the answer-key users + the validation benign
                     cohort — the population the language-model evaluator uses.
                     Default.
  positive_days_only HISTORICAL. Malicious side restricted to attack days.
                     Retained only to reproduce the superseded numbers; it is
                     not comparable to the language-model results.

Pooled and equal-malicious-user fold summaries are both reported, because they
answer different questions and were previously mixed in prose.

A view's score is the exact conditional mean over its channel classes (sum of
class errors / sum of class counts); see `eval_metrics_core.weighted_view` for
why `s_full - s_profile` is not that quantity.

The superseded evaluator is kept as `eval_channel_views_HISTORICAL.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_metrics_core import (  # noqa: E402
    TIE_POLICY, cluster_bootstrap, dedup_by_id, paired_contrasts,
    pooled_metrics, user_max, weighted_view, within_user_roc,
)

VIEWS = {"full": ("PROFILE", "BEHAV"), "profile_only": ("PROFILE",), "behavior_only": ("BEHAV",)}
FPR_BUDGETS = (0.001, 0.01)


def load_model(path: Path) -> Dict[str, object]:
    z = np.load(path, allow_pickle=True)
    counts = {c: int(z[f"n_{c}"]) for c in ("PROFILE", "BEHAV")}
    sums = {c: z[f"err_sum_{c}"].astype(float) for c in ("PROFILE", "BEHAV")}
    total = z["err_sum_total"].astype(float)
    part = float(np.abs(sums["PROFILE"] + sums["BEHAV"] - total).max())
    if part > 1e-6:
        raise RuntimeError(f"{path}: class partition does not sum to the total ({part:.3e})")
    has_hist = z["has_history"].astype(bool) if "has_history" in z.files else np.ones(len(total), bool)
    return {
        "user_id": z["user_id"].astype(str), "y": z["y"].astype(int),
        "split": z["split"].astype(str), "has_history": has_hist,
        "views": {n: weighted_view(sums, counts, cls) for n, cls in VIEWS.items()},
        "class_sums": sums, "n_total": int(z["n_total"]), "partition_err": part,
    }


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--models", nargs="+", required=True, help="name=path/to/channel_scores.npz")
    ap_.add_argument("--matrix", type=Path, required=True, help="user-day matrix (for stable ids)")
    ap_.add_argument("--out-dir", type=Path, required=True)
    ap_.add_argument("--population", choices=("matched", "positive_days_only"), default="matched")
    ap_.add_argument("--bootstrap-draws", type=int, default=2000)
    ap_.add_argument("--seed", type=int, default=42)
    args = ap_.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = np.load(args.matrix, allow_pickle=True)
    uid = mat["user_id"].astype(str)
    day = mat["day_index"]
    y = mat["y"].astype(int)
    split = mat["split"].astype(str)
    example_id = np.array([f"{u}:{d}" for u, d in zip(uid, day)])

    paths = dict(s.split("=", 1) for s in args.models)
    models: Dict[str, Dict[str, object]] = {}
    for name, path in paths.items():
        m = load_model(Path(path))
        if len(m["y"]) != len(y) or not np.array_equal(m["user_id"], uid) or not np.array_equal(m["y"], y):
            raise RuntimeError(f"{name}: score rows do not align with the matrix")
        models[name] = m

    keep_idx = dedup_by_id(example_id, {"y": y, **{f"{n}_full": m["views"]["full"] for n, m in models.items()}})
    dedup_mask = np.zeros(len(y), bool)
    dedup_mask[keep_idx] = True

    if args.population == "matched":
        pop = (split == "eval") | (split == "val")
    else:
        pop = (y == 1) | ((y == 0) & (split == "val"))

    # ONE eligibility set: population AND dedup AND every model's history mask
    hist = np.ones(len(y), bool)
    for m in models.values():
        hist &= m["has_history"]
    eligible = np.flatnonzero(pop & dedup_mask & hist)
    pop_idx = np.flatnonzero(pop)
    pos_users = sorted(set(uid[pop_idx][y[pop_idx] == 1]))

    prov = {
        "population": args.population,
        "population_rows": int(pop.sum()),
        "dropped_duplicates": int((~dedup_mask).sum()),
        "dropped_no_history": int((pop & dedup_mask & ~hist).sum()),
        "common_eligible_rows": int(len(eligible)),
        "per_model_own_history_rows": {n: int((pop & m["has_history"]).sum()) for n, m in models.items()},
        "malicious_users": pos_users,
        "tie_policy": TIE_POLICY,
        "models": paths,
        "note": ("every model is evaluated on this identical eligible set; the forecaster's "
                 "missing-history rows are removed from ALL models so the comparison is like-for-like"),
    }
    pd.DataFrame({"example_id": example_id[eligible]}).to_csv(out_dir / "common_eligible_ids.csv", index=False)

    rows: List[Dict[str, object]] = []
    maxday: List[Dict[str, object]] = []
    for name, m in models.items():
        for view, s in m["views"].items():
            r = pooled_metrics(uid, y, s, eligible, FPR_BUDGETS)
            wu, n_wu, _ = within_user_roc(uid, y, s, eligible, pos_users)
            r.update({"model": name, "view": view, "within_user_roc": wu, "within_user_n": n_wu})
            rows.append(r)
            u = user_max(uid, y, s, eligible)
            for _, rec in u.loc[u["user_id"].isin(pos_users)].iterrows():
                j = int(rec["row"])
                maxday.append({"model": name, "view": view, "user_id": rec["user_id"],
                               "top_example_id": example_id[j], "top_day_index": int(day[j]),
                               "top_day_is_attack": int(y[j] == 1), "top_score": float(s[j])})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(out_dir / "pooled_summary.csv", index=False)
    pd.DataFrame(maxday).to_csv(out_dir / "max_day_per_malicious_user.csv", index=False)

    fold_rows: List[Dict[str, object]] = []
    benign_eligible = eligible[~np.isin(uid[eligible], pos_users)]
    for name, m in models.items():
        for view, s in m["views"].items():
            for pu in pos_users:
                sel = np.concatenate([eligible[uid[eligible] == pu], benign_eligible])
                fr = pooled_metrics(uid, y, s, sel, FPR_BUDGETS)
                fr.update({"model": name, "view": view, "heldout_user": pu})
                fold_rows.append(fr)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out_dir / "fold_rows.csv", index=False)
    agg = {"day_roc": ("day_roc", "mean"), "day_ap": ("day_ap", "mean"),
           "user_roc": ("user_roc", "mean"), "user_ap": ("user_ap", "mean")}
    for b in FPR_BUDGETS:
        agg[f"recall_fpr_{b}"] = (f"day_recall_at_fpr_{b}", "mean")
    fold_mean = folds.groupby(["model", "view"]).agg(**agg).reset_index()
    fold_mean.to_csv(out_dir / "fold_means.csv", index=False)

    contrasts: List[Dict[str, object]] = []
    for metric in ("user_roc", "day_roc"):
        for name, m in models.items():
            boot, _ = cluster_bootstrap(uid, y, m["views"], eligible, pos_users,
                                        metric=metric, draws=args.bootstrap_draws, seed=args.seed)
            point = {v: float(pooled.loc[(pooled.model == name) & (pooled.view == v), metric].iloc[0])
                     for v in m["views"]}
            for c in paired_contrasts(boot, point):
                c["model"] = name
                c["metric"] = metric
                contrasts.append(c)
    pd.DataFrame(contrasts).to_csv(out_dir / "contrasts.csv", index=False)

    gaps: List[Dict[str, object]] = []
    for name, m in models.items():
        full = m["views"]["full"]
        tot = float(full[eligible][y[eligible] == 1].mean() - full[eligible][y[eligible] == 0].mean())
        acc = 0.0
        for c in ("PROFILE", "BEHAV"):
            contrib = m["class_sums"][c][eligible] / m["n_total"]
            gp = float(contrib[y[eligible] == 1].mean())
            gn = float(contrib[y[eligible] == 0].mean())
            acc += gp - gn
            gaps.append({"model": name, "class": c, "delta_contrib": gp - gn,
                         "total_mean_gap": tot, "share": (gp - gn) / tot if tot else float("nan")})
        if abs(acc - tot) > 1e-9:
            raise RuntimeError(f"{name}: mean-gap decomposition does not reconstruct the total")
    pd.DataFrame(gaps).to_csv(out_dir / "mean_gap.csv", index=False)

    prov["bootstrap"] = (f"cluster bootstrap over {len(pos_users)} malicious users, "
                         f"{args.bootstrap_draws} draws, benign cohort fixed"
                         + (" — DESCRIPTIVE at this cluster count" if len(pos_users) <= 8 else ""))
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    print(f"\n=== pooled, population={args.population}, common eligible rows={len(eligible)} ===")
    print(pooled.pivot(index="model", columns="view", values=["day_roc", "user_roc"]).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\n=== equal-malicious-user fold means ===")
    print(fold_mean.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    md = pd.DataFrame(maxday)
    print("\nmalicious users whose top eligible row is an attack day:")
    print(md.groupby(["model", "view"])["top_day_is_attack"].agg(["sum", "count"]).to_string())
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
