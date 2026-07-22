#!/usr/bin/env python3
"""Produce profile-masked variants of a session JSONL for score-time ablation.

Variants:
- no_psy:     drop the PSY line entirely.
- no_profile: drop the PSY line AND replace the DAY line's organizational
              identifier values (project, role, b_unit, f_unit, dept, team,
              itadmin) with "X". week is kept (temporal, not identity).

Only `text` changes; example order and all metadata fields are preserved so
example_idx alignment with the original scoring artifacts holds.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DAY_MASK_KEYS = ("project", "role", "b_unit", "f_unit", "dept", "team", "itadmin")


def mask_day_line(line: str) -> str:
    def repl(m: re.Match) -> str:
        return f"{m.group(1)}=X"
    pattern = re.compile(r"\b(" + "|".join(DAY_MASK_KEYS) + r")=([^ ]+)")
    return pattern.sub(repl, line)


def transform(text: str, variant: str) -> str:
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i == 1 and line.startswith("PSY "):
            continue  # drop psychometrics in both variants
        if variant == "no_profile" and i == 0 and line.startswith("DAY "):
            line = mask_day_line(line)
        out.append(line)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variants", default="no_psy,no_profile")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outs = {v: (args.out_dir / f"eval_{v}.jsonl").open("w", encoding="utf-8") for v in variants}
    n = 0
    with args.in_jsonl.open("r", encoding="utf-8") as f:
        for raw in f:
            ex = json.loads(raw)
            for v in variants:
                ex2 = dict(ex)
                ex2["text"] = transform(ex["text"], v)
                outs[v].write(json.dumps(ex2) + "\n")
            n += 1
    for fh in outs.values():
        fh.close()
    print(json.dumps({"examples": n, "variants": variants, "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
