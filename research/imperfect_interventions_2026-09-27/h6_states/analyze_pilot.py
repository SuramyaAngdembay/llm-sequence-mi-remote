#!/usr/bin/env python3
"""Part 3b analysis: verification of realized codes and the declared estimands and decisions (PRECHECK_PROTOCOL.md).

  python3 analyze_pilot.py PILOT_DIR PROJECTION_DIR CAPTURE_DIR OUT_JSON [--tau 0.012]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ap = argparse.ArgumentParser()
ap.add_argument("pilot", type=Path); ap.add_argument("projection", type=Path); ap.add_argument("capture", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--tau", type=float, default=0.012)
a = ap.parse_args()
rows = list(csv.DictReader((a.pilot / "pilot_rows.csv").open()))
L = {(r["condition"], r["receiver_id"]): float(r["behavior_only"]) for r in rows}
meta = {r["receiver_id"]: (r["user"], r["kind"]) for r in rows}
recv_ids = sorted({r["receiver_id"] for r in rows}, key=lambda x: [r["receiver_id"] for r in rows].index(x))
precs = json.loads((a.projection / "projection_records.json").read_text())
verif = json.loads((a.pilot / "pilot_verification.json").read_text())
checks = json.loads((a.pilot / "pilot_checks.json").read_text())

# ---- verification of the realized codes (GPU, path B) against the declared targets
name_of = {"I2_sel": ("restore_sel", "I2_decoder"), "I1_sel": ("restore_sel", "I1_min_change"), "I2_ctl": ("restore_ctl", "I2_decoder"),
           "noise_I2": ("noise_sel", "I2_decoder")}
tk = np.load(a.capture / "h6_capture_tokens.npz")
src_of = {(int(r), int(d)): int(q) for r, d, q in zip(tk["recv"], tk["dst"], tk["src"])}
target = {}
for rec in precs:
    for cname, (nm, an) in name_of.items():
        if nm in rec and rec[nm].get(an, {}).get("verified"):
            pos = src_of[(rec["recv"], rec["dst"])] if nm == "noise_sel" else rec["dst"]
            target[(cname, rec["recv"], pos)] = (rec[nm]["fixed"], rec[nm]["refill"])
vres = defaultdict(lambda: {"tokens": 0, "matched_target": 0, "support_ok": 0, "values_ok": 0})
for v in verif:
    c = v["condition"]
    if c not in name_of:
        continue
    vres[c]["tokens"] += 1
    key = (c, v["recv"], v["pos"])
    if key not in target:
        continue
    vres[c]["matched_target"] += 1
    z = {int(i): float(x) for i, x in zip(v["idx"], v["val"]) if x > 0}
    fixed, refill = target[key]
    vres[c]["support_ok"] += set(z) == (set(map(int, fixed)) | set(refill))
    vres[c]["values_ok"] += all(abs(z.get(int(f), 0.0) - val) <= max(0.05, 0.02 * abs(val)) for f, val in fixed.items())

# ---- estimands


def user_means(fn, kind):
    per = defaultdict(list)
    for rid, (u, k) in meta.items():
        if k == kind:
            per[u].append(fn(rid))
    return {u: float(np.mean(v)) for u, v in per.items()}


def summarize(um, tau):
    v = np.array(list(um.values())); G = len(v)
    half = student_t.ppf(0.95, G - 1) * v.std(ddof=1) / np.sqrt(G)
    lo, hi = v.mean() - half, v.mean() + half
    rng = np.random.default_rng(42); idx = rng.integers(0, G, size=(10000, G)); bm = v[idx].mean(1)
    dec = "negligible" if (-tau < lo and hi < tau) else ("non-negligible" if (lo >= tau or hi <= -tau) else "inconclusive")
    return {"estimate": float(v.mean()), "ci90_t": [float(lo), float(hi)], "ci95_boot": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))],
            "users": G, "positive_users": int((v > 0).sum()), "decision": dec}


est = {}
defs = {
    "rescue dec (H6 edit)": lambda r: L[("R", r)] - L[("dec", r)],
    "rescue I2_sel": lambda r: L[("R", r)] - L[("I2_sel", r)],
    "rescue I1_sel": lambda r: L[("R", r)] - L[("I1_sel", r)],
    "rescue I2_ctl": lambda r: L[("R", r)] - L[("I2_ctl", r)],
    "rescue rand_I2": lambda r: L[("R", r)] - L[("rand_I2", r)],
    "necessity noise_I2": lambda r: L[("noise_I2", r)] - L[("O", r)],
    "necessity rand_noise": lambda r: L[("rand_noise", r)] - L[("O", r)],
    "I2_sel minus I2_ctl": lambda r: L[("I2_ctl", r)] - L[("I2_sel", r)],
    "I2_sel minus rand_I2": lambda r: L[("rand_I2", r)] - L[("I2_sel", r)],
    "I2_sel minus I1_sel": lambda r: L[("I1_sel", r)] - L[("I2_sel", r)],
    "noise_I2 minus rand_noise": lambda r: L[("noise_I2", r)] - L[("rand_noise", r)],
    "input effect R - O (context)": lambda r: L[("R", r)] - L[("O", r)],
}
for name, fn in defs.items():
    for kind in ("malicious", "benign"):
        est[f"{name}|{kind}"] = summarize(user_means(fn, kind), a.tau)
    um_a, um_b = user_means(fn, "malicious"), user_means(fn, "benign")
    est[f"{name}|attack-minus-benign"] = summarize({u: um_a[u] - um_b[u] for u in um_a if u in um_b}, a.tau)
lift = {}
for kind in ("malicious", "benign"):
    d1, d2 = est[f"rescue I1_sel|{kind}"]["decision"], est[f"rescue I2_sel|{kind}"]["decision"]
    lift[kind] = "not identified (lift-dependent)" if {d1, d2} == {"negligible", "non-negligible"} else f"I1 {d1}, I2 {d2}"
out = {"tau": a.tau, "checks": checks, "verification": dict(vres), "estimands": est, "lift_identification": lift,
       "interaction": "not tested (no joint intervention)"}
a.out.write_text(json.dumps(out, indent=1) + "\n")
for k_, v in est.items():
    print(f"{k_:<48} {v['estimate']:+.5f} t90 [{v['ci90_t'][0]:+.5f}, {v['ci90_t'][1]:+.5f}] boot95 [{v['ci95_boot'][0]:+.5f}, {v['ci95_boot'][1]:+.5f}] {v['decision']}")
print(json.dumps({"verification": dict(vres), "lift": lift, "checks": {k_: checks[k_] for k_ in checks if k_ != "patched_tokens"}}, indent=1))
