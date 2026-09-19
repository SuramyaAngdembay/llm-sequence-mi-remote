#!/usr/bin/env python3
"""User-clustered summary of the history-prefix probe.

The example is not the unit of evidence here: days from one user share that
user's profile, so 3,000 examples from 200 users carry roughly 200 independent
observations, not 3,000. This aggregates to the user first, then bootstraps
over users, so the interval describes the quantity actually being claimed.

Reports the paired within-example contrasts that the design supports:
  B - A   the effect of re-presenting the user's OWN profile
  C - A   the effect of presenting a length-matched STRANGER's profile
  B - C   the identity-specificity of the effect

Stdlib only, so it runs anywhere.
Usage: python3 scripts/analyze_history_prefix.py results/history_prefix/<tag>
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

VIEWS = ("full", "profile_only", "behavior_only", "behavior_ses_only", "psy_only", "day_only")
CONTRASTS = (("B", "A"), ("C", "A"), ("B", "C"))


def bootstrap_ci(per_user: list[float], draws: int = 10000, seed: int = 42) -> tuple[float, float]:
    """Percentile CI resampling USERS, which is the clustering unit."""
    if len(per_user) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(per_user)
    means = []
    for _ in range(draws):
        means.append(sum(per_user[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    rows = list(csv.DictReader((d / "history_prefix_per_example.csv").open()))
    meta = json.loads((d / "history_prefix_meta.json").read_text())
    users = sorted({r["user_id"] for r in rows})
    print(f"\n=== {d.name} ===")
    print(f"{len(rows)} examples, {len(users)} users, "
          f"prefix_include_week={meta.get('prefix_include_week')}")
    print(f"static-profile overlap, own earlier record vs current: "
          f"{meta['mean_overlap_self_vs_current']:.4f}  "
          f"(1.0 means the profile is static, so B tests COPYING, not recall of a "
          f"different past)")
    print(f"static-profile overlap, donor vs current: "
          f"{meta['mean_overlap_donor_vs_current']:.4f}")
    print(f"excluded: {meta['n_excluded_no_earlier_record']} no earlier record, "
          f"{meta['n_excluded_no_length_matched_donor']} no length-matched donor")

    hdr = f"{'view':<18}{'contrast':<10}{'mean':>9}{'95% CI (users)':>24}{'users':>7}{'frac<0':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    out = []
    for view in VIEWS:
        for hi, lo in CONTRASTS:
            by_user: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                try:
                    by_user[r["user_id"]].append(float(r[f"{hi}_{view}"]) - float(r[f"{lo}_{view}"]))
                except (KeyError, ValueError):
                    pass
            pu = [st.fmean(v) for v in by_user.values() if v]
            if not pu:
                continue
            m = st.fmean(pu)
            lo_ci, hi_ci = bootstrap_ci(pu)
            frac = sum(1 for v in pu if v < 0) / len(pu)
            print(f"{view:<18}{hi + '-' + lo:<10}{m:>9.4f}"
                  f"{f'[{lo_ci:+.4f}, {hi_ci:+.4f}]':>24}{len(pu):>7}{frac:>9.3f}")
            out.append({"view": view, "contrast": f"{hi}-{lo}", "mean_over_users": m,
                        "ci_lo": lo_ci, "ci_hi": hi_ci, "n_users": len(pu),
                        "frac_users_negative": frac})
        print()
    with (d / "user_clustered_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {d / 'user_clustered_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
