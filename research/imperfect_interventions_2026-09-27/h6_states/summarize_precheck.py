#!/usr/bin/env python3
"""Apply the frozen precheck pass criteria (PRECHECK_PROTOCOL.md, Part 3a) to project_targets.py output.

  python3 summarize_precheck.py PROJECTION_DIR CAPTURE_DIR OUT_JSON
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

proj, cap, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
R = json.loads((proj / "projection_records.json").read_text())
recv = json.loads((cap / "h6_capture_receivers.json").read_text())
res = {"cpu_vs_gpu": json.loads((proj / "projection_run.json").read_text())["cpu_vs_gpu_path_B_codes"]}


def block(name, anchor):
    rs = [(r, r[name]) for r in R if name in r and anchor in r[name]]
    if not rs:
        return None
    ver = [x[anchor]["verified"] for _, x in rs]
    mass = np.array([x["req_sq"] for _, x in rs]); vm = np.array(ver)
    b = {"tokens": len(rs), "verified_tokens": int(vm.sum()), "verified_token_share": float(vm.mean()),
         "verified_mass_share": float(mass[vm].sum() / mass.sum()),
         "receivers": len({r["recv"] for r, _ in rs}), "users": len({recv[r["recv"]]["user"] for r, _ in rs}),
         "target_kinds": dict(Counter(x["kind"] for _, x in rs)),
         "target_kinds_mass_share": {k: float(sum(x["req_sq"] for _, x in rs if x["kind"] == k) / mass.sum()) for k in {x["kind"] for _, x in rs}},
         "verified_by_kind": {k: f"{sum(1 for _, x in rs if x['kind'] == k and x[anchor]['verified'])}/{sum(1 for _, x in rs if x['kind'] == k)}" for k in {x['kind'] for _, x in rs}},
         "margins_used": dict(Counter(str(x[anchor]["margin"]) for _, x in rs if x[anchor]["verified"])),
         "solver_status": dict(Counter(x[anchor]["solver"]["status"] for _, x in rs)),
         "failures": [{"recv": r["recv"], "dst": r["dst"], "kind": x["kind"], "req_sq": x["req_sq"],
                       "last_margin": x[anchor]["margin"], "solver": x[anchor]["solver"].get("status"),
                       "verify": x[anchor].get("verify")} for r, x in rs if not x[anchor]["verified"]][:50]}
    okr = [(r, x) for r, x in rs if x[anchor]["verified"]]
    if okr:
        to_full = np.array([x[anchor]["move_from_context"] / max(r["full_restoration_norm"], 1e-12) for r, x in okr])
        corr = np.array([x[anchor]["move_from_anchor"] / max(x["decoder_edit_norm"], 1e-12) for r, x in okr])
        w = np.array([x["req_sq"] for r, x in okr])
        q = lambda v: {"q50": float(np.median(v)), "q90": float(np.quantile(v, 0.9)), "max": float(v.max())}
        b.update({"move_over_full_restoration": q(to_full), "move_from_anchor_over_decoder_edit": q(corr),
                  "move_from_anchor_over_decoder_edit_mass_weighted_mean": float((corr * w).sum() / w.sum()),
                  "move_from_context_abs": q(np.array([x[anchor]["move_from_context"] for r, x in okr])),
                  "decoder_edit_norm": q(np.array([x["decoder_edit_norm"] for r, x in okr])),
                  "full_restoration_norm": q(np.array([r["full_restoration_norm"] for r, x in okr])),
                  "max_abs_fixed_coordinate_error": float(max(x[anchor]["verify"]["max_abs_fixed_err"] for r, x in okr)),
                  "kkt_ok_share": float(np.mean([x[anchor]["solver"]["status"] == "ok" for r, x in okr]))})
    return b


for name, anchor in (("restore_sel", "I2_decoder"), ("restore_sel", "I1_min_change"), ("restore_ctl", "I2_decoder"), ("noise_sel", "I2_decoder")):
    res[f"{name}|{anchor}"] = block(name, anchor)
crit = {}
for key in ("restore_sel|I2_decoder", "restore_sel|I1_min_change", "noise_sel|I2_decoder"):
    b = res[key]
    crit[f"P-a {key}"] = b["verified_token_share"] >= 0.95 and b["verified_mass_share"] >= 0.95
for key in ("restore_sel|I2_decoder", "restore_sel|I1_min_change"):
    b = res[key]
    crit[f"P-b {key} median move / full restoration <= 1"] = b.get("move_over_full_restoration", {}).get("q50", np.inf) <= 1
crit["P-b restore_sel|I2_decoder mass-weighted correction / edit <= 1"] = res["restore_sel|I2_decoder"].get("move_from_anchor_over_decoder_edit_mass_weighted_mean", np.inf) <= 1
crit["P-c KKT for every accepted token"] = all(res[k].get("kkt_ok_share", 0) == 1.0 for k in res if k != "cpu_vs_gpu" and res[k])
res["criteria"] = crit
res["precheck_passes"] = all(crit.values())
outp.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k not in ("restore_ctl|I2_decoder",)}, indent=1)[:6000])
