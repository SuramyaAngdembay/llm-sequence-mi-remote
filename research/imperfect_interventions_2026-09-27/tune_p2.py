#!/usr/bin/env python3
"""Choose the P2 realization threshold on development rows only.

Rule (fixed before looking): minimize the wrong-decision rate over all development datasets whose I2 estimand
exists, where wrong = declared negligible while |theta2| >= tau, or declared non-negligible while |theta2| < tau.
Ties go to the larger threshold (more decisive). The grid is fixed below.
"""
import json
import sys

GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
rows = [r for r in json.load(open(sys.argv[1])) if r["procedure"] == "P2_realization_gated"]
p1 = {(r["condition"], r["world_seed"], r["rep"]): r["decision"]
      for r in json.load(open(sys.argv[1])) if r["procedure"] == "P1_applied"}
res = []
for thr in GRID:
    wrong = inc = n = 0
    for r in rows:
        if r["truth_ideal2"] != r["truth_ideal2"]:           # NaN: estimand undefined
            continue
        d = p1[(r["condition"], r["world_seed"], r["rep"])] if r["realization_error"] <= thr else "inconclusive"
        nonneg = abs(r["truth_ideal2"]) >= r["tau"]
        wrong += (d == "negligible" and nonneg) or (d == "non-negligible" and not nonneg)
        inc += d == "inconclusive"; n += 1
    res.append({"threshold": thr, "wrong_rate": wrong / n, "inconclusive_rate": inc / n, "n": n})
best = min(res, key=lambda x: (round(x["wrong_rate"], 12), -x["threshold"]))
print(json.dumps({"grid": res, "chosen": best["threshold"]}, indent=1))
