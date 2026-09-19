#!/usr/bin/env python3
"""Evaluate full / profile-only / behaviour-channel-only views of the
reconstruction score, on the user-disjoint population.

Population discipline matches the language-model branch: positives are scored
against benign users the detector never trained on (the validation split), so
the comparison is user-disjoint on both sides and is **not** a seen-user
detection test. Uncertainty is a cluster bootstrap over the malicious users,
which on r6.2 is four clusters and therefore descriptive only.

Also reports the exact mean-score-gap decomposition

    E_pos[s] - E_neg[s] = sum_c ( E_pos[w_c s_c] - E_neg[w_c s_c] )

with w_c s_c = err_sum_c / n_total per row. Signed contributions cancel, so a
class share can exceed the total or be negative; this is an accounting identity
for the mean gap, not an AUC decomposition and not a mechanism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

VIEWS = {
    "full": ("PROFILE", "BEHAV"),
    "profile_only": ("PROFILE",),
    "behavior_only": ("BEHAV",),
}
FPR_BUDGETS = (0.001, 0.01)


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
    ap.add_argument("--channel-scores", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.channel_scores, allow_pickle=True)
    user_id = z["user_id"].astype(str)
    y = z["y"].astype(int)
    split = z["split"].astype(str)

    counts = {c: int(z[f"n_{c}"]) for c in ("PROFILE", "BEHAV")}
    err = {c: z[f"err_sum_{c}"].astype(float) for c in ("PROFILE", "BEHAV")}

    scores: Dict[str, np.ndarray] = {}
    for name, cls in VIEWS.items():
        num = sum(err[c] for c in cls)
        den = sum(counts[c] for c in cls)
        scores[name] = num / den

    # exact partition check
    part = abs((err["PROFILE"] + err["BEHAV"]) - z["err_sum_total"].astype(float)).max()
    if part > 1e-6:
        raise RuntimeError(f"channel partition does not sum to the total (max err {part:.3e})")

    # user-disjoint population: positives + never-trained validation benigns
    keep = (y == 1) | ((y == 0) & (split == "val"))
    pos_users = sorted(set(user_id[y == 1]))

    rows: List[Dict[str, object]] = []
    for name in VIEWS:
        s = scores[name][keep]
        yy = y[keep]
        uu = user_id[keep]
        u = pd.DataFrame({"u": uu, "y": yy, "s": s}).groupby("u").agg(y=("y", "max"), s=("s", "max"))
        row = {
            "run_name": args.run_name, "view": name,
            "n_rows": int(keep.sum()), "n_positive": int(yy.sum()),
            "prevalence": float(yy.mean()),
            "n_users": int(len(u)), "n_positive_users": int(u.y.sum()),
            "day_roc": safe_auc(yy, s, "roc"), "day_ap": safe_auc(yy, s, "pr"),
            "user_roc": safe_auc(u.y.to_numpy(), u.s.to_numpy(), "roc"),
            "user_ap": safe_auc(u.y.to_numpy(), u.s.to_numpy(), "pr"),
        }
        for b in FPR_BUDGETS:
            row[f"day_recall_at_fpr_{b}"] = recall_at_fpr(yy, s, b)
        # Within-user ranking, identity held fixed. This needs each malicious
        # user's OWN benign days, which sit in the eval split and are therefore
        # not in the user-disjoint population above; scoring it on `keep` alone
        # leaves only positive days and yields NaN.
        s_all = scores[name]
        wu = []
        for pu in pos_users:
            m = user_id == pu
            if np.unique(y[m]).size == 2:
                wu.append(safe_auc(y[m], s_all[m], "roc"))
        row["within_user_roc"] = float(np.nanmean(wu)) if wu else float("nan")
        row["within_user_n"] = len(wu)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "channel_view_summary.csv", index=False)

    # paired cluster bootstrap over malicious users (descriptive at n=4)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(pos_users), size=(args.bootstrap_draws, len(pos_users)))
    neg_mask = (y == 0) & (split == "val")
    boot: Dict[str, np.ndarray] = {}
    for name in VIEWS:
        s_all = scores[name]
        vals = np.empty(args.bootstrap_draws)
        for d in range(args.bootstrap_draws):
            sel = np.concatenate([np.flatnonzero((user_id == pos_users[j]) & (y == 1)) for j in draws[d]])
            idx = np.concatenate([sel, np.flatnonzero(neg_mask)])
            vals[d] = safe_auc(y[idx], s_all[idx], "roc")
        boot[name] = vals
    contrasts = []
    for name in VIEWS:
        if name == "full":
            continue
        d = boot[name] - boot["full"]
        contrasts.append({
            "view": name, "vs": "full", "metric": "day_roc",
            "delta": float(res.loc[res.view == name, "day_roc"].iloc[0] - res.loc[res.view == "full", "day_roc"].iloc[0]),
            "ci_lo": float(np.nanpercentile(d, 2.5)), "ci_hi": float(np.nanpercentile(d, 97.5)),
        })
    pd.DataFrame(contrasts).to_csv(out_dir / "channel_view_contrasts.csv", index=False)

    # exact mean-gap decomposition
    gap_rows = []
    total_gap = float(scores["full"][keep][y[keep] == 1].mean() - scores["full"][keep][y[keep] == 0].mean())
    n_total = int(z["n_total"])
    for c in ("PROFILE", "BEHAV"):
        contrib = err[c][keep] / n_total
        gp = float(contrib[y[keep] == 1].mean()); gn = float(contrib[y[keep] == 0].mean())
        gap_rows.append({"class": c, "pos_mean_weighted_contrib": gp, "neg_mean_weighted_contrib": gn,
                         "delta_contrib": gp - gn, "total_mean_gap": total_gap,
                         "share_of_total_gap": (gp - gn) / total_gap if total_gap else float("nan")})
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(out_dir / "channel_mean_gap.csv", index=False)
    recon = abs(gap["delta_contrib"].sum() - total_gap)
    if recon > 1e-9:
        raise RuntimeError(f"mean-gap decomposition does not reconstruct the total ({recon:.3e})")

    meta = {
        "run_name": args.run_name,
        "views": {k: list(v) for k, v in VIEWS.items()},
        "channel_counts": counts,
        "population": "positives + never-trained validation benign users (user-disjoint on both sides; not a seen-user test)",
        "n_rows": int(keep.sum()), "n_positive_users": len(pos_users),
        "bootstrap": "cluster bootstrap over malicious users; descriptive at n=4",
        "mean_gap_recon_err": float(recon),
        "note": "every channel contributes exactly one squared-error term, so N_P and N_B are constant across rows; unlike the token-class case the naive difference and the exact conditional differ only by an affine map and leave the ranking unchanged",
    }
    (out_dir / "channel_view_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n=== {args.run_name}: user-disjoint ({int(keep.sum())} rows, "
          f"{int(y[keep].sum())} positive, {len(pos_users)} malicious users) ===")
    print(res[["view", "day_roc", "day_ap", "user_roc", "within_user_roc"]].to_string(
        index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\nmean-gap decomposition:")
    print(gap[["class", "delta_contrib", "share_of_total_gap"]].to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
