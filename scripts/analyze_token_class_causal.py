#!/usr/bin/env python3
"""Where does an SAE intervention act — on profile tokens or behaviour tokens?

Work package 4's reading of `eval_token_delta_sae_causal.py --token-class-schema`.

The published causal result says patching the top delta-SAE features lowers the
anomaly score more than patching an activity-matched control set does. That
score is a mean over ALL scored tokens, so the result is silent about which
tokens the repair lands on. These columns answer that, and this script reads
them the way the design supports:

* **The receiver user is the clustering unit.** Many candidate rows share one
  receiver, and rows for one receiver are not independent. Means are taken per
  receiver user first, then bootstrapped over receiver users.
* **The control set is the comparison, not zero.** A negative delta on its own
  says a patch moved the score; only top-minus-control says the *selected*
  features did something the activity-matched control did not.
* **Deltas are computed against the RECOMPUTED base**, from the identical code
  path and batch composition as the patched score. The cached adapted_nll came
  from a different batching and differs by roughly 1e-2 nats, which is larger
  than the effects being measured. The script reports that discrepancy so the
  reader can see why.

Stdlib only.
Usage: python3 scripts/analyze_token_class_causal.py <out-dir> [--by-alpha]
"""
from __future__ import annotations

import csv
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

VIEWS = ("full", "profile_only", "behavior_only", "behavior_ses_only", "psy_only", "day_only")


def boot_ci(vals: list[float], draws: int = 5000, seed: int = 42) -> tuple[float, float]:
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return ms[int(0.025 * draws)], ms[int(0.975 * draws)]


def per_user(rows, key: str) -> list[float]:
    d = defaultdict(list)
    for r in rows:
        try:
            d[r["receiver_example_id"].split(":")[0]].append(float(r[key]))
        except (KeyError, ValueError):
            pass
    return [st.fmean(v) for v in d.values() if v]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    by_alpha = "--by-alpha" in sys.argv
    rows = list(csv.DictReader((d / "token_delta_sae_causal_candidate_rows.csv").open()))
    if not rows:
        print("no candidate rows")
        return 1
    if "delta_full" not in rows[0]:
        print("this run has no per-class columns; rerun with --token-class-schema")
        return 1

    sets = sorted({r["feature_set"] for r in rows})
    ctrl = next((s for s in sets if s.startswith("control")), None)
    tops = [s for s in sets if s != ctrl]
    alphas = sorted({float(r["alpha"]) for r in rows})
    users = {r["receiver_example_id"].split(":")[0] for r in rows}
    print(f"\n=== {d.name} ===")
    print(f"{len(rows)} candidate rows, {len(users)} receiver users, "
          f"feature sets {sets}, alphas {alphas}")

    disc = [abs(float(r["base_cache_minus_recomputed"])) for r in rows
            if r.get("base_cache_minus_recomputed")]
    if disc:
        print(f"cached base minus matched-batch recomputation: mean |diff| "
              f"{st.fmean(disc):.3e}, max {max(disc):.3e}")
        print("  (this is why the base is recomputed rather than read from cache)")

    groups = [(a, [r for r in rows if float(r["alpha"]) == a]) for a in alphas] \
        if by_alpha else [(None, rows)]

    for a, grp in groups:
        if not grp:
            continue
        print(f"\n--- alpha = {a if a is not None else 'all pooled'} ---")
        hdr = f"{'view':<18}{'set':<17}{'mean delta':>12}{'95% CI (users)':>26}{'users':>7}"
        print(hdr)
        print("-" * len(hdr))
        for view in VIEWS:
            for fs in sets:
                pu = per_user([r for r in grp if r["feature_set"] == fs], f"delta_{view}")
                if not pu:
                    continue
                lo, hi = boot_ci(pu)
                print(f"{view:<18}{fs:<17}{st.fmean(pu):>12.5f}"
                      f"{f'[{lo:+.5f}, {hi:+.5f}]':>26}{len(pu):>7}")
            print()

        if ctrl and tops:
            print(f"{'view':<18}{'contrast':<22}{'delta':>12}{'95% CI (users)':>26}")
            print("-" * 78)
            for view in VIEWS:
                for t in tops:
                    # paired by receiver user, so the contrast is within-user
                    a_d = defaultdict(list)
                    b_d = defaultdict(list)
                    for r in grp:
                        u = r["receiver_example_id"].split(":")[0]
                        try:
                            v = float(r[f"delta_{view}"])
                        except (KeyError, ValueError):
                            continue
                        (a_d if r["feature_set"] == t else b_d)[u].append(v)
                    shared = sorted(set(a_d) & set(b_d))
                    diffs = [st.fmean(a_d[u]) - st.fmean(b_d[u]) for u in shared]
                    if not diffs:
                        continue
                    lo, hi = boot_ci(diffs)
                    print(f"{view:<18}{t + ' - ' + ctrl:<22}{st.fmean(diffs):>12.5f}"
                          f"{f'[{lo:+.5f}, {hi:+.5f}]':>26}")
            print("\n(negative = the selected features lower that view's loss more than "
                  "the activity-matched control does)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
