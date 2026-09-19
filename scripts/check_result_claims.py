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
import subprocess
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

# Things a reader can actually follow up. Each is RESOLVED, not merely matched:
# an earlier version of this linter accepted any citation-shaped token, so
# "See nonexistent_evidence.json" passed. External review caught that. Bare
# words like "commit" or "digest" are no longer evidence of anything.
EVIDENCE_PATTERNS = [
    ("hash", r"`?\b([0-9a-f]{7,40})\b`?"),                       # commit or digest
    ("file", r"\b([\w./-]+\.(?:py|json|csv|md|yaml|sbatch|sh|tex|npz))\b"),
    ("ventry", r"\b(V\d{1,3})\b"),                              # progress-record entry
    ("job", r"\bjob[ _]?(\d{6,})\b"),                           # scheduler job id
]

_REPO = Path(__file__).resolve().parents[1]
_GIT_CACHE: dict[str, bool] = {}
_PROGRESS = _REPO / "docs" / "SCORE_DECOMPOSITION_PROGRESS.md"


def _commit_exists(h: str) -> bool:
    if h not in _GIT_CACHE:
        try:
            _GIT_CACHE[h] = subprocess.run(
                ["git", "cat-file", "-e", f"{h}^{{commit}}"], cwd=_REPO,
                capture_output=True, timeout=10).returncode == 0
        except Exception:
            _GIT_CACHE[h] = False
    return _GIT_CACHE[h]


def _resolves(kind: str, token: str) -> bool:
    """Does this citation point at something that exists?"""
    if kind == "file":
        # a path anywhere in the repo, or relative to it
        if (_REPO / token).exists():
            return True
        name = Path(token).name
        return any(True for _ in _REPO.rglob(name)) if name else False
    if kind == "hash":
        return _commit_exists(token)
    if kind == "ventry":
        return _PROGRESS.exists() and f"**{token}" in _PROGRESS.read_text(errors="replace")
    if kind == "job":
        return True          # scheduler ids cannot be resolved from here
    return False

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
    """True only if a citation in the window RESOLVES to something real."""
    for kind, pat in EVIDENCE_PATTERNS:
        for m in re.finditer(pat, window, re.I):
            if _resolves(kind, m.group(1)):
                return True
    return False


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
        # External review defeated the first version with a citation-shaped
        # string pointing at nothing. These two cases exist because of that.
        ("This adapter is identical to the published adapter. See nonexistent_evidence.json",
         True, "citation-SHAPED text pointing at a file that does not exist"),
        ("Frozen before any result existed, commit deadbeefdeadbeef.",
         True, "a commit hash that is not in this repository"),
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
        print("take on trust. Cite a commit, file or V-entry THAT EXISTS -- or soften")
        print("the claim to what is actually known.")
        return 1          # findings fail the check; exit 0 previously hid them
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
