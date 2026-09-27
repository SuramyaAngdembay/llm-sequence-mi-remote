#!/usr/bin/env python3
"""Phase 1 of the 2026-09-26 TWOS feasibility pilot (EXPLORATORY, CPU only).

Donor-policy sensitivity of the corrected TWOS intervention contrast, computed
from the saved candidate scores. No model inference.

For every matched receiver/donor pair (context team, fixed donor type, fixed
alpha) the paired contrast is t_ij = delta(top5) - delta(control5_active), for
the behaviour-only and profile-only views. The same donor weights q apply to
both arms, because t is formed before weighting.

Baseline policy p: each user gets total weight 1/U, split equally over that
user's receivers (a_i), and each receiver's weight is split equally over its
eligible donors (p_ij = a_i / m_i). Eligible donor sets are those of the run
(same-user donors already excluded; re-checked here).

For each rho in the declared grid, the finite-bank bounds are
    lower/upper(rho) = min/max_q sum_ij q_ij t_ij
    s.t. q_ij >= 0, sum_j q_ij = a_i, (1/2) sum_ij |q_ij - p_ij| <= rho,
solved with HiGHS (scipy.optimize.linprog) and, independently, by the exact
greedy solution of this LP (within each receiver, moving mass delta from donor
j to the receiver's best donor changes the objective by delta*(t_best - t_ij)
at total-variation cost delta, so the optimum fills the budget with the moves
of largest gain: a fractional knapsack). The two must agree.

These are exact bounds over the declared policy set for this finite donor
bank. They are NOT population confidence intervals, and averaging donor
outcomes is not the same estimand as patching an averaged donor representation.

  python3 scripts/twos_donor_policy_lp.py CANDIDATE_ROWS.csv ENDPOINTS.json OUT_DIR
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

RHOS = (0.0, 0.05, 0.10, 0.25)
EXPECTED_SHA = "2fbfe5f2da58d7a18fd700ecf88d93f98c67aebea12a90ee34348359145922e4"
ARMS = ("top5", "control5_active")


def load_pairs(path: Path, donor_type: str, alpha: str, views: Tuple[str, ...]):
    """Return {(receiver, donor): {view: t}} for pairs present in both arms."""
    cells: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        if r["context_mode"] != "team" or r["donor_type"] != donor_type or float(r["alpha"]) != float(alpha):
            continue
        k = (r["receiver_example_id"], r["donor_example_id"])
        if r["feature_set"] in cells[k]:
            raise SystemExit(f"duplicate row for {k} {r['feature_set']}")
        cells[k][r["feature_set"]] = r
    out = {}
    for k, v in cells.items():
        if set(ARMS) <= set(v):
            if k[0].split(":")[0] == k[1].split(":")[0]:
                raise SystemExit(f"same-user donor pair {k}")
            out[k] = {view: float(v["top5"][f"delta_{view}"]) - float(v["control5_active"][f"delta_{view}"]) for view in views}
    return out


def structure(pairs: Dict[Tuple[str, str], Dict[str, float]], view: str):
    recvs = sorted({r for r, _ in pairs})
    users = sorted({r.split(":")[0] for r in recvs})
    n_by_user = defaultdict(int)
    for r in recvs:
        n_by_user[r.split(":")[0]] += 1
    a = {r: 1.0 / (len(users) * n_by_user[r.split(":")[0]]) for r in recvs}
    donors = {r: sorted(d for rr, d in pairs if rr == r) for r in recvs}
    keys = [(r, d) for r in recvs for d in donors[r]]
    t = np.array([pairs[k][view] for k in keys])
    p = np.array([a[r] / len(donors[r]) for r, _ in keys])
    recv_index = np.array([recvs.index(r) for r, _ in keys])
    return recvs, users, a, keys, t, p, recv_index


def solve_lp(t, p, recv_index, a_vec, rho, sense):
    from scipy.sparse import coo_matrix
    n = len(t)
    c = np.concatenate([sense * t, np.zeros(n)])
    ar = np.arange(n)
    # equalities: receiver marginals
    A_eq = coo_matrix((np.ones(n), (recv_index, ar)), shape=(len(a_vec), 2 * n)).tocsr()
    # |q - p| <= s ; (1/2) sum s <= rho   (sparse: the CERT banks have ~10^4 pairs)
    rows = np.concatenate([ar, ar, n + ar, n + ar, np.full(n, 2 * n)])
    cols = np.concatenate([ar, n + ar, ar, n + ar, n + ar])
    vals = np.concatenate([np.ones(n), -np.ones(n), -np.ones(n), -np.ones(n), np.full(n, 0.5)])
    A_ub = coo_matrix((vals, (rows, cols)), shape=(2 * n + 1, 2 * n)).tocsr()
    b_ub = np.concatenate([p, -p, [rho]])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=a_vec, bounds=[(0, None)] * (2 * n), method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return sense * res.fun, res.x[:n], res


def greedy(t, p, recv_index, rho, sense):
    """Exact solution: fill the TV budget with the largest-gain moves to each receiver's extreme donor."""
    q = p.copy()
    moves = []
    for i in np.unique(recv_index):
        idx = np.flatnonzero(recv_index == i)
        best = idx[np.argmax(sense * t[idx])]
        for j in idx:
            if j != best:
                moves.append((sense * (t[best] - t[j]), j, best))
    moves.sort(key=lambda m: -m[0])
    budget = rho
    for gain, j, best in moves:
        if budget <= 0 or gain <= 0:
            break
        mv = min(p[j], budget)
        q[j] -= mv; q[best] += mv; budget -= mv
    return float((q * t).sum()), q


def breakdown_radius(t, p, recv_index, base):
    """Smallest rho at which the bound in the direction opposite the baseline sign reaches zero."""
    sense = -1.0 if base > 0 else 1.0          # move toward zero
    moves = []
    for i in np.unique(recv_index):
        idx = np.flatnonzero(recv_index == i)
        best = idx[np.argmax(sense * t[idx])]
        for j in idx:
            if j != best:
                g = sense * (t[best] - t[j])
                if g > 0:
                    moves.append((g, p[j]))
    moves.sort(key=lambda m: -m[0])
    need, used = abs(base), 0.0
    for g, cap in moves:
        if g * cap >= need:
            return used + need / g
        need -= g * cap; used += cap
    return float("inf")


def concentration(q, recv_index, a_vec, p):
    shares = q / a_vec[recv_index]
    moved = np.abs(q - p) > 1e-12
    per_recv = defaultdict(list)
    for s, i in zip(shares, recv_index):
        per_recv[i].append(s)
    eff = [1.0 / np.sum(np.square(v)) for v in per_recv.values()]
    return {"receivers_reweighted": int(len({i for i, m in zip(recv_index, moved) if m})),
            "max_single_donor_share": float(shares.max()),
            "mean_effective_donors_per_receiver": float(np.mean(eff)),
            "tv_used": float(0.5 * np.abs(q - p).sum())}


def main() -> int:
    csv_path, endpoints_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise SystemExit(f"candidate rows sha256 {sha} != expected")
    views = ("behavior_only", "profile_only")
    ends = json.loads(endpoints_path.read_text())
    report: dict = {"input_sha256": sha, "rho_grid": list(RHOS), "label": "finite-bank donor-policy bounds (exploratory; not confidence intervals)", "analyses": {}}
    lines = ["TWOS donor-policy sensitivity (EXPLORATORY; finite-bank bounds over the declared policy set, not confidence intervals)", ""]
    checks = {}
    max_gap = [0.0]
    for donor_type in ("benign", "anomalous"):
        a1 = load_pairs(csv_path, donor_type, "1.0", views)
        a0 = load_pairs(csv_path, donor_type, "0.0", views)
        # reconstruction-only contrast should not depend on the donor
        spread = defaultdict(list)
        for (r, d), v in a0.items():
            spread[r].append(v["behavior_only"])
        max_spread = max(max(v) - min(v) for v in spread.values())
        series = {}
        for view in views:
            series[view] = {k: v[view] for k, v in a1.items()}
            series[f"{view} | alpha-0 reconstruction contrast"] = {k: a0[k][view] for k in a1 if k in a0}
            series[f"{view} | alpha-1 minus alpha-0 (additive subtraction; nonlinear interactions not removed)"] = {
                k: a1[k][view] - a0[k][view] for k in a1 if k in a0}
        for name, s in series.items():
            view = name.split(" | ")[0]
            pairs = {k: {view: v} for k, v in s.items()}
            recvs, users, a, keys, t, p, ri = structure(pairs, view)
            a_vec = np.array([a[r] for r in recvs])
            base = float((p * t).sum())
            rows = []
            for rho in RHOS:
                lo, qlo, _ = solve_lp(t, p, ri, a_vec, rho, +1.0)
                hi, qhi, _ = solve_lp(t, p, ri, a_vec, rho, -1.0)
                glo, _ = greedy(t, p, ri, rho, -1.0)
                ghi, _ = greedy(t, p, ri, rho, +1.0)
                gap = max(abs(lo - glo), abs(hi - ghi))
                max_gap[0] = max(max_gap[0], gap)
                checks[f"{donor_type}|{name}|rho={rho}|LP=exact greedy within 1e-9"] = bool(gap < 1e-9)
                lo, hi = glo, ghi                     # report the exact optimum; the LP is its check
                rows.append({"rho": rho, "lower": lo, "upper": hi, "width": max(hi - lo, 0.0), "lp_minus_exact_max_abs": gap,
                             "lower_policy": concentration(qlo, ri, a_vec, p), "upper_policy": concentration(qhi, ri, a_vec, p),
                             "sign": "both signs possible" if lo < 0 < hi else ("all positive" if lo > 0 else ("all negative" if hi < 0 else "touches zero"))})
            checks[f"{donor_type}|{name}|rho=0 reproduces baseline"] = abs(rows[0]["lower"] - base) < 1e-12 and abs(rows[0]["upper"] - base) < 1e-12
            # fully concentrated limit: every receiver on its extreme donor
            ext_lo = sum(a[r] * min(s[(r, d)] for rr, d in keys if rr == r) for r in recvs)
            ext_hi = sum(a[r] * max(s[(r, d)] for rr, d in keys if rr == r) for r in recvs)
            lo1, _, _ = solve_lp(t, p, ri, a_vec, 1.0, +1.0)
            hi1, _, _ = solve_lp(t, p, ri, a_vec, 1.0, -1.0)
            glo1, _ = greedy(t, p, ri, 1.0, -1.0); ghi1, _ = greedy(t, p, ri, 1.0, +1.0)
            checks[f"{donor_type}|{name}|rho=1: exact equals per-receiver extreme donors"] = abs(glo1 - ext_lo) < 1e-12 and abs(ghi1 - ext_hi) < 1e-12
            checks[f"{donor_type}|{name}|rho=1: LP within 1e-9 of extremes"] = abs(lo1 - ext_lo) < 1e-9 and abs(hi1 - ext_hi) < 1e-9
            rstar = breakdown_radius(t, p, ri, base) if base != 0 else 0.0
            fine = [round(x, 4) for x in np.linspace(0.0, 0.25, 51)]
            fine_curve = [[x, greedy(t, p, ri, x, -1.0)[0], greedy(t, p, ri, x, +1.0)[0]] for x in fine]
            report["analyses"][f"{donor_type}|{name}"] = {
                "n_pairs": len(keys), "n_receivers": len(recvs), "n_users": len(users),
                "donors_per_receiver": [int(min(np.bincount(ri))), int(np.median(np.bincount(ri))), int(max(np.bincount(ri)))],
                "baseline_uniform": base, "bounds": rows, "fully_concentrated_limits": [ext_lo, ext_hi],
                "breakdown_radius_to_zero_from_baseline_sign": rstar, "exact_curve_rho_lower_upper": fine_curve}
            lines.append(f"== {donor_type} donors | {name}  ({len(keys)} pairs, {len(recvs)} receivers, {len(users)} users)")
            lines.append(f"   uniform baseline {base:+.5f}; breakdown radius to zero {rstar:.4f}; fully concentrated limits [{ext_lo:+.5f}, {ext_hi:+.5f}]")
            for row in rows:
                lines.append(f"   rho {row['rho']:.2f}: [{row['lower']:+.5f}, {row['upper']:+.5f}] width {row['width']:.5f}  {row['sign']}; "
                             f"lower policy: {row['lower_policy']['receivers_reweighted']} receivers reweighted, max donor share {row['lower_policy']['max_single_donor_share']:.2f}; "
                             f"upper policy: {row['upper_policy']['receivers_reweighted']} reweighted, max share {row['upper_policy']['max_single_donor_share']:.2f}")
            lines.append("")
        report["analyses"][f"{donor_type}|alpha0_max_within_receiver_spread_behavior"] = max_spread
        lines.append(f"   ({donor_type}) alpha-0 contrast spread across donors within a receiver (behaviour): max {max_spread:.2e}")
        lines.append("")
    # reproduce the published endpoint at rho = 0
    ep = ends["by_context_and_donor"]["team | benign donors"]["at_alpha"]
    for view in views:
        want = ep[view]["selected_minus_control"]["primary_mean_of_user_means"]
        got = report["analyses"][f"benign|{view}"]["baseline_uniform"]
        checks[f"rho=0 baseline equals endpoints.json primary ({view})"] = abs(want - got) < 1e-12
        report["analyses"][f"benign|{view}"]["endpoints_json_primary"] = want
    # synthetic check: donor-constant contrasts give zero width
    t_s = np.repeat(np.array([0.3, -0.1, 0.2]), 4); p_s = np.full(12, 1 / 12); ri_s = np.repeat(np.arange(3), 4)
    lo_s, _, _ = solve_lp(t_s, p_s, ri_s, np.full(3, 1 / 3), 0.25, +1.0)
    hi_s, _, _ = solve_lp(t_s, p_s, ri_s, np.full(3, 1 / 3), 0.25, -1.0)
    checks["synthetic donor-constant contrasts give zero width"] = abs(hi_s - lo_s) < 1e-12
    # synthetic check: one receiver, two donors, analytic bound
    lo_2, _, _ = solve_lp(np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0, 0]), np.array([1.0]), 0.1, +1.0)
    checks["synthetic two-donor case equals 0.5 - rho"] = abs(lo_2 - 0.4) < 1e-12
    report["checks"] = checks
    report["max_abs_lp_minus_exact"] = max_gap[0]
    lines.append(f"Largest |HiGHS LP - exact greedy| over all bounds: {max_gap[0]:.2e}")
    lines.append(f"Checks ({len(checks)}): " + ("all pass" if all(checks.values()) else "FAILURES: " + ", ".join(k for k, v in checks.items() if not v)))
    (out / "donor_policy_bounds.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "donor_policy_bounds.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
