#!/usr/bin/env python3
"""Correctness checks for the shared metric core.

Focused on the defects it exists to prevent:
  1. eligibility must govern the bootstrap, not only the point estimate;
  2. duplicate ids that disagree must raise, not be silently dropped;
  3. recall@FPR must be deterministic under ties;
  4. user_max must report the row that won, so a benign top row is visible;
  5. `s_full - s_profile` can reorder examples relative to the conditional
     behaviour score, while `s_full - w_P*s_P` cannot.

Usage:  python3 scripts/tests/test_eval_metrics_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_metrics_core import (  # noqa: E402
    cluster_bootstrap, dedup_by_id, paired_contrasts, pooled_metrics,
    recall_at_fpr, roc, user_max, weighted_view, within_user_roc,
)

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def test_eligibility_governs_bootstrap() -> None:
    print("1. eligibility governs the bootstrap, not just the point estimate")
    # Constructed so eligibility genuinely matters, mirroring the real finding:
    # each malicious user's HIGHEST row is one of their benign days. Including
    # those rows makes the malicious users outrank the benign cohort; excluding
    # them (positives-only) leaves the lower attack-day scores and a worse
    # ranking. A statistic that is degenerate under both sets would not test
    # anything, which is what an earlier version of this check did.
    uid = np.array(["m1"] * 4 + ["m2"] * 4 + ["b1"] * 4 + ["b2"] * 4)
    y = np.array([1, 1, 0, 0] * 2 + [0] * 8)
    s = np.array([
        0.30, 0.25, 0.95, 0.10,   # m1: attacks low, one benign day very high
        0.28, 0.22, 0.93, 0.12,   # m2: same shape
        0.50, 0.40, 0.45, 0.35,   # b1
        0.55, 0.42, 0.48, 0.38,   # b2
    ])
    all_rows = np.arange(16)
    pos_only = np.flatnonzero((y == 1) | (~np.isin(uid, ["m1", "m2"])))
    b_all, _ = cluster_bootstrap(uid, y, {"v": s}, all_rows, ["m1", "m2"], draws=200, seed=1)
    b_pos, _ = cluster_bootstrap(uid, y, {"v": s}, pos_only, ["m1", "m2"], draws=200, seed=1)
    m_all, m_pos = float(np.nanmean(b_all["v"])), float(np.nanmean(b_pos["v"]))
    check("different eligibility gives a different bootstrap distribution",
          not np.isclose(m_all, m_pos), f"all-rows {m_all:.4f} vs positives-only {m_pos:.4f}")
    check("all-rows eligibility ranks malicious users higher (their top row is benign)",
          m_all > m_pos, f"{m_all:.4f} > {m_pos:.4f}")
    # the point estimate must agree with the bootstrap's own population
    p_all = pooled_metrics(uid, y, s, all_rows)["user_roc"]
    p_pos = pooled_metrics(uid, y, s, pos_only)["user_roc"]
    check("point estimate and bootstrap use the same population",
          abs(p_all - m_all) < 0.2 and abs(p_pos - m_pos) < 0.2,
          f"point {p_all:.3f}/{p_pos:.3f} vs boot {m_all:.3f}/{m_pos:.3f}")
    # the fixed cohort must never be resampled: with one malicious user the
    # bootstrap of a deterministic statistic is degenerate
    b1, _ = cluster_bootstrap(uid, y, {"v": s}, all_rows, ["m1"], draws=50, seed=2)
    check("single-cluster bootstrap is degenerate (benign cohort held fixed)",
          np.allclose(b1["v"], b1["v"][0], equal_nan=True))
    # paired draws: identical views give an exactly zero contrast
    bb, _ = cluster_bootstrap(uid, y, {"a": s, "b": s.copy()}, all_rows, ["m1", "m2"], draws=100, seed=3)
    c = paired_contrasts(bb, {"a": 0.5, "b": 0.5}, baseline="a")[0]
    check("paired draws make identical views contrast to exactly zero",
          c["ci_lo"] == 0.0 and c["ci_hi"] == 0.0)


def test_dedup_conflicts() -> None:
    print("2. duplicate ids")
    ids = np.array(["a", "b", "a", "c"])
    ok = dedup_by_id(ids, {"y": np.array([1, 0, 1, 0]), "s": np.array([0.5, 0.2, 0.5, 0.1])})
    check("agreeing duplicates deduplicate to first occurrences", list(ok) == [0, 1, 3], str(list(ok)))
    try:
        dedup_by_id(ids, {"s": np.array([0.5, 0.2, 0.9, 0.1])})
        check("conflicting duplicates raise", False)
    except ValueError as e:
        check("conflicting duplicates raise", "disagree" in str(e))


def test_recall_determinism() -> None:
    print("3. recall@FPR is deterministic under ties")
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = [recall_at_fpr(y, s, 0.1) for _ in range(5)]
    check("repeated calls agree", len(set(r)) == 1, f"{r[0]:.4f}")
    check("tied positives at the threshold are NOT counted",
          r[0] == 0.5, f"expected 0.5 (only the 0.9 positive), got {r[0]}")


def test_user_max_reports_winner() -> None:
    print("4. user_max reports the winning row")
    uid = np.array(["m1", "m1", "m1"])
    y = np.array([1, 0, 0])
    s = np.array([0.1, 0.9, 0.2])          # the user's TOP row is benign
    u = user_max(uid, y, s, np.arange(3))
    check("user label is the max over the user's rows", int(u["y"].iloc[0]) == 1)
    check("winning row index is reported", int(u["row"].iloc[0]) == 1)
    p = pooled_metrics(uid, y, s, np.arange(3))
    check("top-row-is-positive count catches a benign winner",
          p["n_users_whose_top_row_is_positive"] == 0)


def test_within_user_uses_eligibility() -> None:
    print("5. within-user honours eligibility")
    uid = np.array(["m1"] * 4)
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2])
    full, n, _ = within_user_roc(uid, y, s, np.arange(4), ["m1"])
    check("all rows eligible -> defined", n == 1 and abs(full - 1.0) < 1e-12)
    pos_only, n2, _ = within_user_roc(uid, y, s, np.array([0, 2]), ["m1"])
    check("positives-only eligibility -> undefined, not silently 1.0",
          n2 == 0 and np.isnan(pos_only))


def test_subtraction_documentation() -> None:
    print("6. the subtraction identity the docstring now states")
    rng = np.random.default_rng(7)
    n_P, n_B = 12, 32
    sP = rng.uniform(0, 2, size=500)
    sB = rng.uniform(0, 2, size=500)
    wP, wB = n_P / (n_P + n_B), n_B / (n_P + n_B)
    s_full = wP * sP + wB * sB
    naive = s_full - sP
    correct = s_full - wP * sP
    check("s_full - w_P*s_P equals w_B*s_B exactly",
          np.allclose(correct, wB * sB))
    order_naive = np.argsort(naive)
    order_true = np.argsort(sB)
    check("s_full - s_P REORDERS relative to the behaviour score",
          not np.array_equal(order_naive, order_true))
    check("s_full - w_P*s_P preserves the behaviour ranking",
          np.array_equal(np.argsort(correct), order_true))
    # and the module's own conditional view matches w_B*s_B up to the scale
    v = weighted_view({"P": sP * n_P, "B": sB * n_B}, {"P": n_P, "B": n_B}, ["B"])
    check("weighted_view returns the conditional behaviour mean", np.allclose(v, sB))


def test_label_dependent_guard() -> None:
    print("7. a label-dependent sample is REFUSED for detection metrics")
    from eval_metrics_core import (  # noqa: E402
        LabelDependentSampleError, assert_sample_usable_for_detection,
    )
    uid = np.array(["a", "a", "b", "b"])
    y = np.array([1, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.5, 0.2])
    el = np.arange(4)

    clean = {"label_dependent_sample": False}
    dirty = {"label_dependent_sample": True, "label_dependent_rule": "kept all positive users"}

    # the guard must not fire on a clean sample, or it is useless
    try:
        assert_sample_usable_for_detection(clean, "test")
        check("a clean sample passes the guard", True)
    except LabelDependentSampleError:
        check("a clean sample passes the guard", False)

    try:
        assert_sample_usable_for_detection(dirty, "test")
        check("a label-dependent sample raises", False)
    except LabelDependentSampleError as e:
        check("a label-dependent sample raises", "kept all positive users" in str(e),
              "and the message names the rule")

    # and the guard must actually be wired into the metric, not merely exist
    r = pooled_metrics(uid, y, s, el, sample_meta=clean)
    check("pooled_metrics still computes on a clean sample", "day_roc" in r)
    try:
        pooled_metrics(uid, y, s, el, sample_meta=dirty)
        check("pooled_metrics REFUSES a label-dependent sample", False)
    except LabelDependentSampleError:
        check("pooled_metrics REFUSES a label-dependent sample", True)
    # omitting the meta must stay permissive, so existing call sites keep working
    check("omitting sample_meta does not break existing callers",
          "day_roc" in pooled_metrics(uid, y, s, el))


def main() -> int:
    print("eval_metrics_core correctness checks\n")
    test_eligibility_governs_bootstrap()
    test_dedup_conflicts()
    test_recall_determinism()
    test_user_max_reports_winner()
    test_within_user_uses_eligibility()
    test_subtraction_documentation()
    test_label_dependent_guard()
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
