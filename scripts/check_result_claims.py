#!/usr/bin/env python3
"""Refuse unverifiable claims about our own process in results write-ups.

The self-audit against arXiv:2509.08713 found that every claim about *data* in
this repository is checked by an assertion, while every claim about *process* —
"pre-registered", "the same adapter", "not selected after seeing the outcome" —
was asserted from memory. One of them was simply false. Process claims are the
ones a reader cannot check and the ones the paper's failure modes are made of.

This linter does one thing: when a write-up makes a claim about how the work was
done, it must cite something checkable on the same line or the line before —
a commit hash, a content digest, a file path, or a progress-record entry.

It is deliberately noisy in one direction. A flagged line is not necessarily
wrong; it is a line whose truth a reader has to take on trust. The fix is either
to cite the evidence or to soften the claim to what is actually known.

Usage:
  python3 scripts/check_result_claims.py [paths...]     # default: results/, docs/
  python3 scripts/check_result_claims.py --self-test
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Claims about how the work was done. Each needs evidence a reader can follow.
PROCESS_CLAIMS = [
    (r"\bpre-?registered\b", "pre-registration"),
    (r"\b(declared|fixed|frozen|chosen|decided)\s+(before|prior to)\b", "ordering"),
    (r"\bbefore (any|seeing|the) (loss|result|data|outcome|numbers?)\b", "ordering"),
    # "neither was selected after seeing the outcome" -- the sentence that got
    # past review -- needs the negation and the verb to be allowed to separate.
    (r"\b(not|neither|none|never)\b[^.]{0,40}\b(selected|chosen|picked|decided|run)\b[^.]{0,20}\bafter\b", "ordering"),
    (r"\b(after|before) seeing\b", "ordering"),
    (r"\bnever (edited|changed|modified|touched)\b", "immutability"),
    (r"\b(unchanged|untouched) (since|from)\b", "immutability"),
    (r"\bidentical to the (published|original|reference)\b", "provenance"),
    (r"\bsame (adapter|checkpoint|model|weights) as\b", "provenance"),
    (r"\bmatch(es|ing)? the (run|job|configuration) that produced\b", "provenance"),
    (r"\bbyte-identical\b", "provenance"),
    (r"\breproduc(es|ed) (the|its) (cache|published|original)\b", "reproduction"),
]

# Things a reader can actually follow up.
EVIDENCE = [
    r"`?\b[0-9a-f]{7,64}\b`?",                 # commit hash or content digest
    r"\b[\w./-]+\.(py|json|csv|md|yaml|sbatch|sh)\b",   # a file to open
    r"\bV\d{1,3}\b",                           # progress-record entry
    r"\bjob[ _]?\d{6,}\b",                     # a scheduler job id
    r"\bcommit\b", r"\bdigest\b", r"\bmd5\b", r"\bsha256\b",
    r"\bgit log\b", r"\bsee\s+`", r"\brecorded in\b",
]

# Lines that are meta-discussion of claim discipline, not claims themselves.
EXEMPT = re.compile(
    r"(which was \*\*false\*\*|was \*\*false\*\*|this linter|PROCESS_CLAIMS|"
    r"must cite|^\s*[-*]\s*\(r\"|claim patterns|for example|e\.g\.)", re.I)

# A claim inside quotation marks, introduced as something previously said, is a
# quotation being corrected -- not a new assertion. Without this, a write-up is
# penalised for quoting the very sentence it is retracting.
QUOTED_PAST = re.compile(
    r"(claimed|said|stated|asserted|read|wrote)\b[^\"]{0,40}[\"\u201c]", re.I)


def has_evidence(window: str) -> bool:
    return any(re.search(p, window, re.I) for p in EVIDENCE)


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    out = []
    in_code = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or EXEMPT.search(line) or QUOTED_PAST.search(line):
            continue
        for pat, kind in PROCESS_CLAIMS:
            if re.search(pat, line, re.I):
                # Evidence may sit a couple of lines away: a heading is often
                # followed by a blank line and then the citation.
                window = "\n".join(lines[max(0, i - 2) : i + 3])
                if not has_evidence(window):
                    out.append((i + 1, kind, line.strip()[:110]))
                break
    return out


def structural_checks(path: Path) -> list[str]:
    """Two things every results write-up owes a reader."""
    text = path.read_text(encoding="utf-8", errors="replace")
    problems = []
    if re.search(r"\b(ROC|AUC|delta|contrast|nats)\b", text):
        if not re.search(r"\b(dataset|CERT|TWOS|LANL|r4\.2|r6\.2)\b", text):
            problems.append("reports metrics but never names the dataset")
        if re.search(r"\bcontrast\b", text, re.I) and not re.search(
                r"\b(primary|pre-?declared|headline) (comparison|test|table)\b", text, re.I):
            problems.append("reports contrasts but declares no primary comparison")
    return problems


def self_test() -> int:
    """The linter must catch the sentence that actually got past review."""
    import tempfile
    cases = [
        # (text, should_flag, label)
        ("Both variants are reported; neither was selected after seeing the outcome.",
         True, "the real false sentence from results/history_prefix/README.md"),
        ("The views were frozen before any result existed.", True, "bare ordering claim"),
        ("The views were frozen on 2026-09-17 in commit 007ca54 and never edited.",
         False, "same claim, with a commit to check"),
        ("This adapter is identical to the published one.", True, "bare provenance claim"),
        ("Identical to the published adapter by weights digest 5a6c2ffa (see "
         "results/provenance/adapter_fingerprints/aquaman_r62.json).", False,
         "same claim, with a digest and a file"),
        ("The day-field contrast is -0.0272 [-0.045, -0.010].", False, "a plain number"),
    ]
    ok = True
    print("self-test: does this linter catch what actually got past review?\n")
    for text, should, label in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(text)
            p = Path(fh.name)
        flagged = bool(scan_file(p))
        p.unlink()
        good = flagged == should
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {'flags' if should else 'allows':6s}: {label}")
    print()
    print("all self-tests passed" if ok else "SELF-TEST FAILED -- the linter is theatre")
    return 0 if ok else 1


# A dated handoff or audit is a record of what was believed on that date.
# Rewriting one to satisfy a linter would falsify the record. These are reported
# separately rather than skipped, so they stay visible without inviting edits.
HISTORICAL = re.compile(r"(HANDOFF_|VALIDITY_AUDIT_|ANVIL_AUDIT_NOTES_|_20\d\d-\d\d-\d\d)")


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    roots = [Path(a) for a in sys.argv[1:] if not a.startswith("-")] or \
            [Path("results"), Path("docs")]
    files = sorted({f for r in roots for f in (r.rglob("*.md") if r.is_dir() else [r])})
    n_claims = n_struct = n_hist = 0
    historical: list[str] = []
    for f in files:
        if HISTORICAL.search(f.name):
            h = scan_file(f)
            if h:
                n_hist += len(h)
                historical.append(f"  {f}: {len(h)} claim(s)")
            continue
        claims = scan_file(f)
        struct = structural_checks(f) if f.name == "README.md" and "results" in f.parts else []
        if not claims and not struct:
            continue
        print(f"\n{f}")
        for line_no, kind, text in claims:
            print(f"  L{line_no:<5} [{kind}] uncited process claim")
            print(f"         {text}")
            n_claims += 1
        for s in struct:
            print(f"  [structure] {s}")
            n_struct += 1
    if historical:
        print("\nDated historical records (NOT to be rewritten -- a handoff is a record "
              "of\nwhat was believed on its date):")
        for h in historical:
            print(h)
    print(f"\n{len(files)} files scanned: {n_claims} uncited process claims, "
          f"{n_struct} structural gaps, {n_hist} in historical records (exempt)")
    if n_claims or n_struct:
        print("\nA flagged line is not necessarily wrong. It is a line a reader must")
        print("take on trust. Cite the commit, digest, file or V-entry -- or soften")
        print("the claim to what is actually known.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
