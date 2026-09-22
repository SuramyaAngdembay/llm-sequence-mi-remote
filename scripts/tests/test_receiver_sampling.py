#!/usr/bin/env python3
"""Regression test for the receiver-sampling bug that failed job 20838328.

`--max-receivers N` picks one shared receiver sample so every context mode
scores the same rows. `--receiver-user-file` restricts receivers to held-out
users. Combined, the sample was drawn from ALL positives and the restriction
applied afterwards, so the lookup asked for rows the filter had removed.

It raised, which was lucky. The same defect in a variant that dropped missing
keys instead would have quietly scored fewer receivers than requested, under a
run whose whole purpose is a held-out test.

Usage: python3 scripts/tests/test_receiver_sampling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_token_delta_sae_causal import build_candidate_pairs, build_all_candidate_pairs  # noqa: E402

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def meta(n_users: int = 6, per_user: int = 10) -> pd.DataFrame:
    rows = []
    i = 0
    for u in range(n_users):
        for d in range(per_user):
            rows.append({"row_idx": i, "user_id": f"U{u:02d}", "y": int(d < 4),
                         "team": f"T{u % 2}", "role": f"R{u % 3}"})
            i += 1
    return pd.DataFrame(rows)


def main() -> int:
    print("receiver sampling with a user restriction\n")
    m = meta()
    held_out = {"U00", "U01"}                       # 2 of 6 users, 8 positive rows
    n_pos_restricted = int(((m["y"] == 1) & (m["user_id"].isin(held_out))).sum())
    print(f"  pool: {int((m['y']==1).sum())} positives overall, "
          f"{n_pos_restricted} within the held-out users")

    print("\n1. the combination that crashed job 20838328")
    try:
        pairs = build_all_candidate_pairs(
            m, context_modes=["team", "role"], max_receivers=4,
            max_candidate_donors=3, seed=42, receiver_users=held_out)
        check("restricted users + max_receivers does not raise", True)
    except Exception as e:
        check("restricted users + max_receivers does not raise", False, f"{type(e).__name__}: {e}")
        return 1

    print("\n2. and the receivers it chose are correct")
    by_user = dict(zip(m["row_idx"], m["user_id"]))
    recv = {r for v in pairs.values() for r, _ in v}
    outside = sorted({by_user[r] for r in recv} - held_out)
    check("every receiver belongs to a held-out user", not outside, f"stray users: {outside}")
    check("no more receivers than requested", len(recv) <= 4, f"{len(recv)} receivers")
    check("the cap was actually reached", len(recv) == 4, f"{len(recv)} of 4")

    print("\n3. the same receivers are shared across context modes")
    per_mode = {k: {r for r, _ in v} for k, v in pairs.items() if v}
    sets = list(per_mode.values())
    check("all context modes score the same receiver rows",
          all(x == sets[0] for x in sets), f"{len(sets)} non-empty mode/donor keys")

    print("\n4. a stale shared sample now fails loudly, not silently")
    try:
        build_candidate_pairs(m, donor_label=0, context_mode="team", max_receivers=4,
                              max_candidate_donors=3, rng=np.random.default_rng(0),
                              receiver_indices=np.array([0, 1, 40, 41]),   # 40,41 are U04
                              receiver_users=held_out)
        check("ineligible shared receivers raise a clear error", False, "no error raised")
    except ValueError as e:
        check("ineligible shared receivers raise a clear error", "not eligible" in str(e))
    except KeyError:
        check("ineligible shared receivers raise a clear error", False,
              "still a bare pandas KeyError")

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
