#!/usr/bin/env python3
"""Correctness checks for the history-prefix probe construction.

These are the checks that decide whether a loss difference between conditions
A/B/C means anything:

  1. the current day's token ids are identical across conditions;
  2. scored target membership is identical across conditions, which requires
     excluding the current day's first token in ALL of them;
  3. no prefix token is ever scored;
  4. donor choice is deterministic and label-free;
  5. length matching is exact, and unmatched examples are excluded not fudged;
  6. the prefix carries the static profile only — no `week`.

Runs with stdlib + numpy; no torch or tokenizer needed.
Usage:  python3 scripts/tests/test_history_prefix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from history_prefix import (  # noqa: E402
    DAY_STATIC_FIELDS, PSY_FIELDS, assemble, build_prefix,
    choose_length_matched_donor, donor_order, parse_profile, profile_overlap,
    validate_conditions,
)

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


TEXT = (
    "DAY week=0 project=114 role=31 b_unit=0 f_unit=9 dept=11 team=62 itadmin=0\n"
    "PSY O=36 C=35 E=19 A=44 N=21\n"
    "SESSIONS total=2 kept=2\n"
    "SES idx=0 pc=0 duration=187.667 n_logon=1\n"
    "SES idx=1 pc=0 duration=205.467 n_logon=1"
)


def toy_tokenize(text: str) -> list[int]:
    """Deterministic stand-in: one id per whitespace-delimited chunk plus one
    per newline, so token counts respond to content the way a real tokenizer
    roughly does."""
    ids = []
    for i, line in enumerate(text.split("\n")):
        if i:
            ids.append(10)                    # the newline token
        for tok in line.split(" "):
            if tok:
                ids.append(100 + (hash(tok) % 500))
    return ids


def test_prefix_content() -> None:
    print("1. prefix content")
    prof = parse_profile(TEXT)
    check("profile parsed from DAY/PSY lines",
          all(f in prof for f in DAY_STATIC_FIELDS + PSY_FIELDS), str(sorted(prof)[:6]))
    pre = build_prefix(prof)
    check("prefix has no `week` field", "week=" not in pre, pre.split("\n")[0][:60])
    check("prefix carries every static field",
          all(f"{f}={prof[f]}" in pre for f in DAY_STATIC_FIELDS + PSY_FIELDS))
    check("prefix ends on a line boundary", pre.endswith("\n"))
    missing = dict(prof); missing.pop("O")
    try:
        build_prefix(missing); check("missing field raises", False)
    except ValueError:
        check("missing field raises", True)


def test_target_membership() -> None:
    print("2. token ids and scored targets identical across conditions")
    cur = toy_tokenize(TEXT)
    n = len(cur)
    prof = parse_profile(TEXT)
    pre_ids = toy_tokenize(build_prefix(prof))
    other = dict(prof); other["role"] = "77"
    pre_ids_c = toy_tokenize(build_prefix(other))
    # force the same length for the C control, as the eligibility rule requires
    pre_ids_c = pre_ids_c[: len(pre_ids)] + [999] * max(0, len(pre_ids) - len(pre_ids_c))

    built = {
        "A": assemble([], cur),
        "B": assemble(pre_ids, cur),
        "C": assemble(pre_ids_c, cur),
    }
    rec = validate_conditions(built, n_current=n, max_seq_len=2048)
    check("validation passes on a well-formed triple", rec["n_scored_targets"] == n - 1,
          f"{rec['n_scored_targets']} targets for {n} current tokens")
    check("A has no prefix, B and C do",
          rec["prefix_tokens"]["A"] == 0 and rec["prefix_tokens"]["B"] == len(pre_ids))
    for name in ("A", "B", "C"):
        ids, tgt = built[name]
        check(f"{name}: current-day ids unchanged", list(ids[len(ids) - n:]) == cur)
        check(f"{name}: current day's first token is not a target", not tgt[len(ids) - n])
        check(f"{name}: no prefix token scored", not any(tgt[: len(ids) - n]))


def test_validation_catches_mistakes() -> None:
    print("3. the validator rejects the mistakes it exists for")
    cur = toy_tokenize(TEXT)
    n = len(cur)
    pre = [7] * 12

    # scoring the current day's first token in B (the natural bug) must fail
    ids, tgt = assemble(pre, cur)
    tgt = list(tgt); tgt[len(pre)] = True
    try:
        validate_conditions({"A": assemble([], cur), "B": (ids, tgt)}, n, 2048)
        check("extra target in B is rejected", False)
    except ValueError as e:
        check("extra target in B is rejected", "targets" in str(e) or "tail" in str(e))

    # marking a prefix token as a target must fail
    ids2, tgt2 = assemble(pre, cur)
    tgt2 = list(tgt2); tgt2[0] = True
    try:
        validate_conditions({"A": assemble([], cur), "B": (ids2, tgt2)}, n, 2048)
        check("prefix token as target is rejected", False)
    except ValueError:
        check("prefix token as target is rejected", True)

    # altering the current-day ids must fail
    cur_bad = list(cur); cur_bad[3] = cur_bad[3] + 1
    try:
        validate_conditions({"A": assemble([], cur), "B": assemble(pre, cur_bad)}, n, 2048)
        check("changed current-day ids are rejected", False)
    except ValueError as e:
        check("changed current-day ids are rejected", "token ids" in str(e) or "tail" in str(e))

    # exceeding max_seq_len must fail
    try:
        validate_conditions({"A": assemble([], cur), "B": assemble([7] * 5000, cur)}, n, 2048)
        check("over-length condition is rejected", False)
    except ValueError as e:
        check("over-length condition is rejected", "max_seq_len" in str(e))


def test_donor_selection() -> None:
    print("4. donor selection is deterministic, label-free and length-exact")
    pool = [f"U{i:03d}" for i in range(300)]
    o1 = donor_order("ACM2278", pool, seed=42)
    o2 = donor_order("ACM2278", pool, seed=42)
    check("order is reproducible", o1 == o2)
    check("recipient excluded from its own pool", "ACM2278" not in o1)
    check("a different recipient gets a different order", o1 != donor_order("CDE1846", pool, seed=42))
    check("a different seed gets a different order", o1 != donor_order("ACM2278", pool, seed=43))

    base = parse_profile(TEXT)
    profiles = {}
    for i, u in enumerate(pool):
        p = dict(base)
        p["role"] = str(30 + i % 7)           # same token length
        p["O"] = str(10 + i % 80)             # same token length
        profiles[u] = p
    want = len(toy_tokenize(build_prefix(base)))
    donor, examined = choose_length_matched_donor(
        "ACM2278", want, pool, profiles, lambda t: len(toy_tokenize(t)), seed=42)
    check("a length-matched donor is found", donor is not None, f"{donor} after {examined} candidates")
    if donor:
        check("donor prefix length matches exactly",
              len(toy_tokenize(build_prefix(profiles[donor]))) == want)
        check("donor is not the recipient", donor != "ACM2278")

    # when nothing matches, the example is excluded rather than approximated
    none_match, _ = choose_length_matched_donor(
        "ACM2278", 99999, pool, profiles, lambda t: len(toy_tokenize(t)), seed=42)
    check("unmatched recipient yields no donor (example excluded)", none_match is None)


def test_overlap_reported() -> None:
    print("5. profile overlap is measurable")
    a = parse_profile(TEXT)
    b = dict(a); b["role"] = "77"; b["O"] = "99"
    check("identical profiles overlap 1.0", profile_overlap(a, a) == 1.0)
    n = len(DAY_STATIC_FIELDS + PSY_FIELDS)
    check("two differing fields reduce overlap accordingly",
          abs(profile_overlap(a, b) - (n - 2) / n) < 1e-12, f"{profile_overlap(a, b):.4f}")


def main() -> int:
    print("history_prefix correctness checks\n")
    test_prefix_content()
    test_target_membership()
    test_validation_catches_mistakes()
    test_donor_selection()
    test_overlap_reported()
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
