#!/usr/bin/env python3
"""Extension of Phase 1 (EXPLORATORY, CPU only): donor-policy bounds for CERT package 4.

Same construction as scripts/twos_donor_policy_lp.py (common donor weights for
both arms, equal user weight, equal receiver weight within user, uniform donor
baseline, total-variation ball), applied to the package-4 candidate rows at
alpha 1 for every context mode and donor type. The rho = 0 value must equal
the published primary endpoint (endpoints_alpha1.0.json) exactly. Bounds are
exact for this finite donor bank; they are not confidence intervals.

  python3 scripts/cert_donor_policy_lp.py ROWS.csv ENDPOINTS_ALPHA1.json OUT_DIR
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twos_donor_policy_lp import RHOS, breakdown_radius, concentration, greedy, solve_lp, structure  # noqa: E402

EXPECTED_SHA = "2387be94"          # prefix recorded in results/cert_package4_token_class/INPUT_HASHES.txt
VIEWS = ("behavior_only", "profile_only", "full")


def main():
    path, ends_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    sha = h.hexdigest()
    if not sha.startswith(EXPECTED_SHA):
        raise SystemExit(f"rows sha256 {sha[:12]} does not match the recorded package-4 input")
    cells = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        if float(r["alpha"]) != 1.0:
            continue
        k = (r["context_mode"], r["donor_type"], r["receiver_example_id"], r["donor_example_id"])
        cells[k][r["feature_set"]] = r
    ends = json.loads(ends_path.read_text())["by_context_and_donor"]
    report, lines, checks = {"input_sha256": sha, "rho_grid": list(RHOS), "analyses": {}}, [], {}
    lines.append("CERT package 4, alpha 1: donor-policy bounds (EXPLORATORY; exact finite-bank bounds, not confidence intervals)\n")
    for mode in ("dept", "dept_role", "role", "team"):
        for dtype in ("benign", "anomalous"):
            pairs = {}
            for (m, dt, rv, dn), v in cells.items():
                if m == mode and dt == dtype and {"top5", "control5_active"} <= set(v):
                    if rv.split(":")[0] == dn.split(":")[0]:
                        raise SystemExit("same-user donor")
                    pairs[(rv, dn)] = {w: float(v["top5"][f"delta_{w}"]) - float(v["control5_active"][f"delta_{w}"]) for w in VIEWS}
            for view in VIEWS:
                recvs, users, a, keys, t, p, ri = structure(pairs, view)
                a_vec = np.array([a[r] for r in recvs])
                base = float((p * t).sum())
                want = ends[f"{mode} | {dtype} donors"]["at_alpha"][view]["selected_minus_control"]["primary_mean_of_user_means"]
                checks[f"{mode}|{dtype}|{view} rho=0 equals published endpoint"] = abs(base - want) < 1e-12
                rows = []
                for rho in RHOS:
                    glo, qlo = greedy(t, p, ri, rho, -1.0); ghi, qhi = greedy(t, p, ri, rho, +1.0)
                    if view == "behavior_only" and rho in (0.05, 0.25):
                        lo, _, _ = solve_lp(t, p, ri, a_vec, rho, +1.0); hi, _, _ = solve_lp(t, p, ri, a_vec, rho, -1.0)
                        checks[f"{mode}|{dtype}|{view}|rho={rho} LP = exact"] = max(abs(lo - glo), abs(hi - ghi)) < 1e-9
                    rows.append({"rho": rho, "lower": glo, "upper": ghi, "width": max(ghi - glo, 0.0),
                                 "lower_policy": concentration(qlo, ri, a_vec, p), "upper_policy": concentration(qhi, ri, a_vec, p)})
                ext = [sum(a[r] * min(pairs[(r, d)][view] for rr, d in keys if rr == r) for r in recvs),
                       sum(a[r] * max(pairs[(r, d)][view] for rr, d in keys if rr == r) for r in recvs)]
                rstar = breakdown_radius(t, p, ri, base)
                report["analyses"][f"{mode}|{dtype}|{view}"] = {"n_pairs": len(keys), "n_receivers": len(recvs), "n_users": len(users),
                                                               "baseline_uniform": base, "published_endpoint": want, "bounds": rows,
                                                               "fully_concentrated_limits": ext, "breakdown_radius_to_zero": rstar}
                r25 = rows[-1]
                lines.append(f"{mode:<9} {dtype:<9} {view:<13} base {base:+.5f}  rho*={rstar:.4f}  "
                             + "  ".join(f"rho {r['rho']:.2f} [{r['lower']:+.5f}, {r['upper']:+.5f}]" for r in rows[1:])
                             + f"  full concentration [{ext[0]:+.5f}, {ext[1]:+.5f}]  ({len(keys)} pairs, {len(users)} users)")
    report["checks"] = checks
    lines.append(f"\nChecks ({len(checks)}): " + ("all pass" if all(checks.values()) else "FAILURES: " + ", ".join(k for k, v in checks.items() if not v)))
    (out / "cert_donor_policy_bounds.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "cert_donor_policy_bounds.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
