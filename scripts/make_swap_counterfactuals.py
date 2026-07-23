#!/usr/bin/env python3
"""Generate profile/behavior swap counterfactuals for positive examples.

For each positive user-day, produce:
- profile_swapped: DAY organizational fields and the PSY line replaced by
  those of a randomly chosen benign user-day; SES behavior kept.
- behavior_swapped: DAY/PSY kept; SESSIONS count line and all SES lines
  replaced by those of a randomly chosen benign user-day.

Scoring these with the frozen model tests feature semantics directly:
if anomaly evidence follows the substituted profile, the mechanism is
identity-like; if it follows substituted behavior, it is behavioral.
Prediction from the attribution result: r6.2 scores drop under
profile_swapped (benign profile installed) but not behavior_swapped;
r4.2 shows the reverse.

Only positives (plus their benign donors, re-emitted unmodified for
reference) are written, so scoring is cheap.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DAY_KEYS = ("project", "role", "b_unit", "f_unit", "dept", "team", "itadmin")


def split_lines(text: str):
    lines = text.split("\n")
    day, psy = lines[0], lines[1] if len(lines) > 1 and lines[1].startswith("PSY ") else ""
    rest_start = 2 if psy else 1
    return day, psy, lines[rest_start:]


def swap_day_fields(day_recv: str, day_donor: str) -> str:
    donor_vals = dict(re.findall(r"\b(\w+)=([^ ]+)", day_donor))
    def repl(m):
        k = m.group(1)
        return f"{k}={donor_vals.get(k, m.group(2))}" if k in DAY_KEYS else m.group(0)
    return re.sub(r"\b(\w+)=([^ ]+)", repl, day_recv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.in_jsonl.open()]
    positives = [r for r in rows if int(r["y"]) == 1]
    benign = [r for r in rows if int(r["y"]) == 0]
    rng = np.random.default_rng(args.seed)
    donors = [benign[i] for i in rng.integers(0, len(benign), size=len(positives))]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outs = {v: (args.out_dir / f"positives_{v}.jsonl").open("w") for v in
            ("original", "profile_swapped", "behavior_swapped")}
    manifest = []
    for ex, donor in zip(positives, donors):
        d_r, p_r, ses_r = split_lines(ex["text"])
        d_d, p_d, ses_d = split_lines(donor["text"])
        variants = {
            "original": ex["text"],
            "profile_swapped": "\n".join([swap_day_fields(d_r, d_d), p_d] + ses_r) if p_d else "\n".join([swap_day_fields(d_r, d_d)] + ses_r),
            "behavior_swapped": "\n".join(([d_r, p_r] if p_r else [d_r]) + ses_d),
        }
        for v, txt in variants.items():
            ex2 = dict(ex); ex2["text"] = txt
            outs[v].write(json.dumps(ex2) + "\n")
        manifest.append({"example_id": ex["example_id"], "donor_example_id": donor["example_id"]})
    for fh in outs.values():
        fh.close()
    (args.out_dir / "swap_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"positives": len(positives), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
