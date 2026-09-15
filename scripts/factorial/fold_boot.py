#!/usr/bin/env python3
"""
Cluster-bootstrap the fold-aligned detector metrics for the 8B factorial.

Reads fold_<MODE>/fold_aligned_detector_rows.csv (one row per fold = per
held-out malicious user, produced by eval_fold_aligned_detector_metrics.py).
Because the fold construction is deterministic (seed 42), the same malicious
users index the folds across all conditions, so we can:
  (a) report per-condition mean +/- 95% cluster-bootstrap CI over malicious
      users, and
  (b) do a PAIRED bootstrap of the full - no_profile difference (resample the
      malicious users, recompute each condition's mean on the resampled set,
      take the difference) -> a single-seed significance statement for the
      removal effect.
"""
import pandas as pd, numpy as np, sys, os, json

P = sys.argv[1] if len(sys.argv) > 1 else "."
MODES = ["full", "no_psy", "no_profile", "shuffle_profile"]
METRIC = "user_roc_auc"   # user-level, max-agg over days (headline convention)
B = 10000
rng = np.random.default_rng(42)

# load per-fold rows keyed by the held-out malicious user
per = {}
for m in MODES:
    f = os.path.join(P, f"fold_{m}", "fold_aligned_detector_rows.csv")
    if not os.path.exists(f):
        print(f"MISSING {f}"); continue
    d = pd.read_csv(f)
    # each fold identified by its held-out positive user; keep score-name adapted_nll
    d = d[d["score_name"] == "adapted_nll"] if "score_name" in d.columns else d
    key = "held_out_user" if "held_out_user" in d.columns else ("fold" if "fold" in d.columns else d.columns[0])
    per[m] = d.set_index(key)[METRIC]

# align on the common set of folds (malicious users)
common = None
for m in per:
    common = per[m].index if common is None else common.intersection(per[m].index)
common = sorted(common)
n = len(common)
print(f"aligned folds (malicious users): {n}")

def ci(x):
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

# per-condition bootstrap CI over malicious users
print("\n=== per-condition fold-aligned user-ROC (mean [95% cluster-boot CI]) ===")
vals = {m: per[m].reindex(common).to_numpy(dtype=float) for m in per}
for m in MODES:
    if m not in vals: continue
    boot = np.array([np.nanmean(vals[m][rng.integers(0, n, n)]) for _ in range(B)])
    mean, lo, hi = ci(boot)
    print(f"  {m:16s} {np.nanmean(vals[m]):.3f}  [{lo:.3f}, {hi:.3f}]")

# paired removal-effect: no_profile - full
if "full" in vals and "no_profile" in vals:
    diff = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        diff.append(np.nanmean(vals["no_profile"][idx]) - np.nanmean(vals["full"][idx]))
    diff = np.array(diff)
    dm, dlo, dhi = ci(diff)
    p_one = float(np.mean(diff <= 0))  # fraction of draws where removal did NOT help
    print("\n=== removal effect (no_profile - full), paired cluster bootstrap ===")
    print(f"  delta user-ROC = {np.nanmean(vals['no_profile'])-np.nanmean(vals['full']):+.3f}  "
          f"[{dlo:+.3f}, {dhi:+.3f}]   one-sided p(no help) = {p_one:.4f}")
print("\nDONE_FOLD_BOOT")
