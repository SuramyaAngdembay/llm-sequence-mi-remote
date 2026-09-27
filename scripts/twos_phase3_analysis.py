#!/usr/bin/env python3
"""Per-user analysis of the Phase 3 TWOS validation rows (EXPLORATORY; 8 users, descriptive intervals).

Every predetermined receiver is kept, including receivers with no active
selected feature (all edits are then zero); an active-only analysis is given
separately. Estimates are means of user means with a user-clustered percentile
bootstrap (10,000 and 5,000 draws). With 8 users the intervals are descriptive.

  python3 scripts/twos_phase3_analysis.py phase3_rows.csv OUT.json
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pilots import fmt, summary, user_means  # noqa: E402

CONTRASTS = [("rpU", None), ("rpO", None), ("projO", None), ("recon", None),
             ("rpU", "randI_U"), ("rpU", "randR_U"), ("rpO", "randI_O"), ("rpO", "randR_O"),
             ("projO", "randI_P"), ("projO", "randR_P"), ("rpU", "rpO"), ("rpO", "projO"),
             # common movement budget (added 2026-09-27; present only in runs that scored these conditions)
             ("rpO_atP", "projO"), ("rpO_atP", "randR_P"), ("rpO", "projO_atD"), ("projO_atD", "randR_O")]


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    d = {(r["condition"], r["receiver_id"]): r for r in rows}
    present = {r["condition"] for r in rows}
    global CONTRASTS
    CONTRASTS = [(a, b) for a, b in CONTRASTS if a in present and (b is None or b in present)]
    meta = {r["receiver_id"]: (r["user"], r["kind"]) for r in rows}
    active = {rid for (c, rid), r in d.items() if c == "rpU" and int(r["edit_tokens"]) > 0}
    norms = defaultdict(list)
    for (c, rid), r in d.items():
        if rid in active:
            norms[c].append(float(r["edit_raw_l2"]))
    res, lines = {"edit_raw_l2_median_active": {c: float(np.median(v)) for c, v in norms.items()}}, []
    lines.append("Median edit size over active receivers (raw residual L2): " +
                 ", ".join(f"{c} {np.median(v):.1f}" for c, v in sorted(norms.items()) if c != "zero"))
    for subset, keep in (("all receivers", set(meta)), ("active only", active)):
        for view in ("behavior_only", "profile_only"):
            lines.append(f"\n== {subset} | {view}")
            for kind in ("malicious", "benign"):
                for a, b in CONTRASTS:
                    per = {}
                    for rid in keep:
                        u, k = meta[rid]
                        if k != kind:
                            continue
                        v = float(d[(a, rid)][f"delta_{view}"]) - (float(d[(b, rid)][f"delta_{view}"]) if b else 0.0)
                        per[(u, rid)] = v
                    um = user_means(per)
                    s = summary(um); s["per_user"] = um
                    name = a if b is None else f"{a} - {b}"
                    res[f"{subset}|{view}|{kind}|{name}"] = s
                    lines.append(f"  {kind:<9} {name:<18} {fmt(s)}")
            for a, b in CONTRASTS[4:]:
                mal, ben = {}, {}
                for rid in keep:
                    u, k = meta[rid]
                    v = float(d[(a, rid)][f"delta_{view}"]) - float(d[(b, rid)][f"delta_{view}"])
                    (mal if k == "malicious" else ben)[(u, rid)] = v
                um_m, um_b = user_means(mal), user_means(ben)
                s = summary({u: um_m[u] - um_b[u] for u in um_m if u in um_b})
                res[f"{subset}|{view}|malicious-minus-benign|{a} - {b}"] = s
                lines.append(f"  paired    {'[' + a + ' - ' + b + '] mal-ben':<18} {fmt(s)}")
    key = "all receivers|behavior_only|malicious|rpO - projO"
    lines.append("\nPer-user values, all receivers, behaviour, malicious: rpO - projO " +
                 json.dumps({u: round(v, 5) for u, v in res[key]["per_user"].items()}))
    key = "all receivers|behavior_only|malicious|rpU"
    lines.append("Per-user values, all receivers, behaviour, malicious: rpU alone " +
                 json.dumps({u: round(v, 5) for u, v in res[key]["per_user"].items()}))
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + "\n")
    Path(sys.argv[2]).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
