#!/usr/bin/env python3
"""Results table and the frozen success criteria for P4a (FROZEN_PROTOCOL.md)."""
import json
import sys
from collections import defaultdict

T = json.load(open(sys.argv[1])); W = json.load(open(sys.argv[2]))
procs = ["P0_zero_in_ci", "P1_applied", "P2_realization_gated", "P3_direct_ideal", "P4a_bound_certifiedK",
         "P4b_bound_sampledK", "P4c_hoffman_eq_certifiedK", "P5_two_lifts"]
out = {"table": {}, "criteria": {}}
lines = []
for c, P in T.items():
    realE = sum(w["mean_realization_error"] for w in W[c]) / len(W[c])
    for p in procs:
        r = P[p]; d = r["decisions"]
        out["table"].setdefault(c, {})[p] = {
            "correct": r["correct_rate"], "negligible": d.get("negligible", 0), "non_negligible": d.get("non-negligible", 0),
            "inconclusive": d.get("inconclusive", 0) + d.get("insufficient (bound unavailable)", 0),
            "ill_posed": d.get("ill-posed", 0), "not_identified": d.get("not identified (lift-dependent)", 0),
            "false_negligible": r["false_negligible_rate"], "false_nonnegligible": r["false_nonnegligible_rate"],
            "coverage": r["coverage_of_ideal2"], "median_width": r["median_width"], "realization_error": realE}
# criteria
tab = out["table"]
defined = [c for c in tab if tab[c]["P3_direct_ideal"]["ill_posed"] < 1]
inexact = [c for c in defined if tab[c]["P4a_bound_certifiedK"]["realization_error"] > 0.01]
v1 = {c: tab[c]["P4a_bound_certifiedK"]["false_negligible"] for c in tab if tab[c]["P4a_bound_certifiedK"]["false_negligible"] is not None}
crit1 = all(v <= 0.05 for v in v1.values())
dec_ok = {c: tab[c]["P4a_bound_certifiedK"]["correct"] for c in inexact}
crit2 = sum(v >= 0.5 for v in dec_ok.values()) >= len(inexact) / 2
def wi(p):
    return sum(1 - tab[c][p]["correct"] for c in defined) / len(defined)
crit3 = wi("P4a_bound_certifiedK") < wi("P2_realization_gated")
out["criteria"] = {
    "1_validity_max_false_negligible": max(v1.values()), "1_pass": crit1,
    "2_inexact_conditions": len(inexact), "2_conditions_with_correct_decisive_ge_0.5": sum(v >= 0.5 for v in dec_ok.values()),
    "2_correct_by_condition": dec_ok, "2_pass": crit2,
    "3_wrong_or_inconclusive_P4a": wi("P4a_bound_certifiedK"), "3_wrong_or_inconclusive_P2": wi("P2_realization_gated"),
    "3_wrong_or_inconclusive_P3": wi("P3_direct_ideal"), "3_wrong_or_inconclusive_P1": wi("P1_applied"), "3_pass": crit3,
    "4_note": "P3 is available whenever P4a is in these worlds; criterion 4 is argued, not measured."}
json.dump(out, open(sys.argv[3], "w"), indent=1)
ab = {"P0_zero_in_ci": "P0", "P1_applied": "P1", "P2_realization_gated": "P2", "P3_direct_ideal": "P3", "P4a_bound_certifiedK": "P4a",
      "P4b_bound_sampledK": "P4b", "P4c_hoffman_eq_certifiedK": "P4c", "P5_two_lifts": "P5"}
print(f"{'condition':<28}{'realE':>6} | " + " ".join(f"{ab[p]:>16}" for p in procs))
for c in tab:
    cells = []
    for p in procs:
        r = tab[c][p]
        tag = f"{r['correct']:.2f}"
        if r["false_negligible"]:
            tag += f" FN{r['false_negligible']:.2f}"
        elif r["false_nonnegligible"]:
            tag += f" FP{r['false_nonnegligible']:.2f}"
        elif r["ill_posed"] == 1:
            tag = "ill-posed"
        cells.append(f"{tag:>16}")
    print(f"{c:<28}{tab[c]['P1_applied']['realization_error']:6.2f} | " + " ".join(cells))
print(json.dumps(out["criteria"], indent=1))
