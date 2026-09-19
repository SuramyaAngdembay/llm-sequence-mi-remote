#!/usr/bin/env python3
"""One metric implementation, shared by the language-model and numerical
detector evaluators so their numbers are comparable by construction.

Design rules this module enforces, each of which was a real defect earlier:

* **One eligibility set governs everything.** Point estimates, user maxima,
  within-user metrics, the mean-gap decomposition and the bootstrap all take
  the same `eligible` index array. An earlier evaluator changed the point
  estimate population but left the bootstrap resampling a different one
  (positives-only malicious days plus validation negatives), so the interval
  did not describe the statistic it was printed beside.
* **Deterministic ties.** `recall_at_fpr` fixes the threshold with
  `np.quantile(..., method="higher")` and counts strictly greater scores, so a
  run is reproducible and ties never inflate recall.
* **Duplicates must agree before they are dropped.** `dedup_by_id` fails on
  conflicting labels or scores rather than silently keeping the first row.
* **User aggregation reports which row won.** `user_max` returns the argmax row
  index, because a high max-aggregated user AUC says a malicious user outranks
  benign users, not that the day driving that rank is an attack day.

Nothing here computes a score; it only summarizes one that was computed
elsewhere.
"""
from __future__ import annotations

import os
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TIE_POLICY = (
    "ROC/AP from sklearn (ties get the midpoint rank); recall@FPR uses a "
    "threshold at the (1-fpr) quantile of eligible negatives with "
    "method='higher' and counts scores strictly greater than it"
)


def _finite(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(s)
    return y[m], s[m]


def roc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = _finite(np.asarray(y), np.asarray(s, dtype=float))
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def ap(y: np.ndarray, s: np.ndarray) -> float:
    y, s = _finite(np.asarray(y), np.asarray(s, dtype=float))
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """Empirical ROC operating point on the eligible negatives.

    This is not a deployment threshold: it is calibrated on the same negatives
    it is evaluated against, so it describes the ROC curve, not what a fixed
    alert budget would deliver on fresh data.
    """
    y, s = _finite(np.asarray(y), np.asarray(s, dtype=float))
    neg = s[y == 0]
    if neg.size == 0 or int((y == 1).sum()) == 0:
        return float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr, method="higher"))
    return float((s[y == 1] > thr).mean())


def dedup_by_id(ids: Sequence, frames: Dict[str, np.ndarray]) -> np.ndarray:
    """Indices of the first occurrence of each id, after checking agreement.

    Raises if two rows share an id but disagree on any supplied array, so a
    duplicated evaluation row can never be dropped in favour of a different
    score or label.
    """
    ids = np.asarray(ids)
    order = pd.Series(np.arange(len(ids)))
    first = order.groupby(ids).transform("first").to_numpy()
    dup = first != np.arange(len(ids))
    if dup.any():
        for name, arr in frames.items():
            a = np.asarray(arr)
            bad = ~np.isclose(a[dup].astype(float), a[first[dup]].astype(float), equal_nan=True) \
                if a.dtype.kind in "fc" else a[dup] != a[first[dup]]
            if np.any(bad):
                n = int(np.sum(bad))
                raise ValueError(f"{n} duplicated ids disagree on '{name}'; refusing to deduplicate")
    keep = np.zeros(len(ids), dtype=bool)
    keep[np.unique(first)] = True
    return np.flatnonzero(keep)


def user_max(user_id: np.ndarray, y: np.ndarray, s: np.ndarray, eligible: np.ndarray) -> pd.DataFrame:
    """Max-aggregate per user over `eligible` rows, keeping the winning row."""
    df = pd.DataFrame(
        {"user_id": np.asarray(user_id)[eligible], "y": np.asarray(y)[eligible],
         "score": np.asarray(s, dtype=float)[eligible], "row": np.asarray(eligible)}
    )
    idx = df.groupby("user_id")["score"].idxmax()
    out = df.loc[idx].copy()
    out["y"] = df.groupby("user_id")["y"].max().reindex(out["user_id"]).to_numpy()
    return out.reset_index(drop=True)


class LabelDependentSampleError(RuntimeError):
    """A detection metric was asked for on a sample built using labels."""


def assert_sample_usable_for_detection(meta: Dict[str, object] | str | "os.PathLike",
                                       what: str = "this metric") -> None:
    """Refuse to compute a detection metric on a label-dependent sample.

    Some probes sample users with a rule that reads labels -- the history-prefix
    probe retains every user with a positive day, so its sample cannot be made
    easier by dropping malicious users. That is fine for the loss CONTRASTS the
    probe reports, which use no labels at all. It is not fine for ROC, AP or
    recall, where a label-dependent sample silently sets the prevalence.

    The audit against arXiv:2509.08713 found this recorded only as a sentence in
    a help string. A sentence is not an invariant. This raises.
    """
    if not isinstance(meta, dict):
        import json as _json
        meta = _json.loads(pathlib.Path(meta).read_text())
    if bool(meta.get("label_dependent_sample")):
        raise LabelDependentSampleError(
            f"refusing to compute {what}: this sample was built with a "
            f"label-dependent rule ({meta.get('label_dependent_rule', 'unspecified')}). "
            "Loss contrasts are fine; detection metrics are not. Rebuild the "
            "sample without reading labels, or compute the metric elsewhere."
        )


def pooled_metrics(
    user_id: np.ndarray, y: np.ndarray, s: np.ndarray, eligible: np.ndarray,
    fpr_budgets: Sequence[float] = (0.001, 0.01),
    sample_meta: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if sample_meta is not None:
        assert_sample_usable_for_detection(sample_meta, "pooled detection metrics")
    yy = np.asarray(y)[eligible]
    ss = np.asarray(s, dtype=float)[eligible]
    out: Dict[str, object] = {
        "n_rows": int(len(eligible)), "n_positive": int(yy.sum()),
        "prevalence": float(yy.mean()) if len(yy) else float("nan"),
        "day_roc": roc(yy, ss), "day_ap": ap(yy, ss),
    }
    for b in fpr_budgets:
        out[f"day_recall_at_fpr_{b}"] = recall_at_fpr(yy, ss, b)
    u = user_max(user_id, y, s, eligible)
    out.update({
        "n_users": int(len(u)), "n_positive_users": int(u["y"].sum()),
        "user_roc": roc(u["y"].to_numpy(), u["score"].to_numpy()),
        "user_ap": ap(u["y"].to_numpy(), u["score"].to_numpy()),
    })
    pos = u.loc[u["y"] == 1]
    out["n_users_whose_top_row_is_positive"] = int(
        (np.asarray(y)[pos["row"].to_numpy()] == 1).sum()
    ) if len(pos) else 0
    return out


def within_user_roc(
    user_id: np.ndarray, y: np.ndarray, s: np.ndarray, eligible: np.ndarray,
    users: Iterable[str],
) -> Tuple[float, int, Dict[str, float]]:
    """Mean within-user ROC over `eligible` rows only (identity held fixed)."""
    uid = np.asarray(user_id); yy = np.asarray(y); ss = np.asarray(s, dtype=float)
    el = np.zeros(len(uid), dtype=bool); el[eligible] = True
    per: Dict[str, float] = {}
    for u in users:
        m = el & (uid == u)
        if np.unique(yy[m]).size == 2:
            per[str(u)] = roc(yy[m], ss[m])
    vals = list(per.values())
    return (float(np.nanmean(vals)) if vals else float("nan")), len(vals), per


def cluster_bootstrap(
    user_id: np.ndarray, y: np.ndarray, scores: Dict[str, np.ndarray],
    eligible: np.ndarray, cluster_users: Sequence[str], metric: str = "user_roc",
    draws: int = 10000, seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Resample `cluster_users` (the malicious users) with replacement, holding
    the benign cohort fixed, using the SAME eligible rows and the SAME draws for
    every score view so contrasts are paired.

    Returns (per-view bootstrap values, the draw index matrix).
    """
    uid = np.asarray(user_id); yy = np.asarray(y)
    el = np.zeros(len(uid), dtype=bool); el[eligible] = True
    cluster_users = list(cluster_users)
    rows_of = {u: np.flatnonzero(el & (uid == u)) for u in cluster_users}
    fixed = np.flatnonzero(el & ~np.isin(uid, cluster_users))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cluster_users), size=(draws, len(cluster_users)))
    out: Dict[str, np.ndarray] = {}
    for name, s in scores.items():
        ss = np.asarray(s, dtype=float)
        vals = np.empty(draws)
        for d in range(draws):
            sel = np.concatenate([rows_of[cluster_users[j]] for j in idx[d]] + [fixed])
            if metric == "user_roc":
                # a resampled user may appear twice; disambiguate so the max
                # aggregation does not merge the copies
                tag = np.concatenate(
                    [np.full(len(rows_of[cluster_users[j]]), f"{cluster_users[j]}#{k}")
                     for k, j in enumerate(idx[d])] + [uid[fixed]]
                )
                df = pd.DataFrame({"u": tag, "y": yy[sel], "s": ss[sel]})
                g = df.groupby("u").agg(y=("y", "max"), s=("s", "max"))
                vals[d] = roc(g["y"].to_numpy(), g["s"].to_numpy())
            elif metric == "day_roc":
                vals[d] = roc(yy[sel], ss[sel])
            elif metric == "day_ap":
                vals[d] = ap(yy[sel], ss[sel])
            else:
                raise ValueError(f"unsupported bootstrap metric {metric}")
        out[name] = vals
    return out, idx


def paired_contrasts(
    boot: Dict[str, np.ndarray], point: Dict[str, float], baseline: str = "full",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name in boot:
        if name == baseline:
            continue
        d = boot[name] - boot[baseline]
        rows.append({
            "view": name, "vs": baseline,
            "delta": float(point[name] - point[baseline]),
            "ci_lo": float(np.nanpercentile(d, 2.5)),
            "ci_hi": float(np.nanpercentile(d, 97.5)),
            "tail_frac_not_better": float(np.mean(d <= 0)),
        })
    return rows


def weighted_view(
    class_sums: Dict[str, np.ndarray], class_counts: Dict[str, int], classes: Sequence[str]
) -> np.ndarray:
    """Exact conditional mean over `classes`: sum of their loss / sum of counts.

    Note on a documentation error corrected here: with constant weights
    w_c = N_c / N, `s_full = w_P*s_P + w_B*s_B`, so `s_full - s_P` equals
    `w_B*s_B + (w_P - 1)*s_P` = `w_B*(s_B - s_P)`, which can REORDER examples
    relative to s_B. The ranking-preserving subtraction is `s_full - w_P*s_P`,
    which is `w_B*s_B`. This function sidesteps both by taking the conditional
    mean directly from sums and counts.
    """
    num = sum(np.asarray(class_sums[c], dtype=float) for c in classes)
    den = sum(int(class_counts[c]) for c in classes)
    if den == 0:
        return np.full(len(num), np.nan)
    return num / den
