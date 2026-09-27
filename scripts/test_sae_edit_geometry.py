#!/usr/bin/env python3
"""Checks for scripts/sae_edit_geometry.py (numpy/scipy only).

  python3 scripts/test_sae_edit_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_edit_geometry import exact_code_realizable, fixed_cell_edit, ldp, min_norm_equality, topk_relu  # noqa: E402

failures = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        failures.append(name)


# Research-note toy encoder: E(x, y) = Top2(ReLU(x, x + y, y))
W_toy = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]); b_toy = np.zeros(3)
enc = lambda x: topk_relu((W_toy @ x + b_toy)[None, :], 2)[0]
check("toy: E(2,1) = (2,3,0)", np.allclose(enc(np.array([2.0, 1.0])), [2, 3, 0]))
check("toy: E(1,2) = (0,3,2)", np.allclose(enc(np.array([1.0, 2.0])), [0, 3, 2]))
pre0 = W_toy @ np.array([2.0, 1.0])
check("toy: requesting (1,3,0) is not realizable", exact_code_realizable(pre0, W_toy, np.array([1.0, 3.0, 0.0]), 2, 1e-6) == "infeasible")
check("toy: requesting (2,3,0) is realizable", exact_code_realizable(pre0, W_toy, np.array([2.0, 3.0, 0.0]), 2, 1e-6) == "realizable")
check("toy: requesting three nonzeros under Top2 is certified unrealizable",
      exact_code_realizable(pre0, W_toy, np.array([1.0, 1.0, 1.0]), 2, 1e-6) == "more_than_k")

# min-norm equality solution matches the unconstrained least-norm formula
rng = np.random.default_rng(0)
B = rng.standard_normal((3, 10)); c = rng.standard_normal(3); ginv = rng.uniform(0.5, 2.0, 10)
v = min_norm_equality(B, c, ginv)
check("min-norm equality: constraints hold", np.allclose(B @ v, c))
v_alt = v + ginv * (np.eye(10) - np.linalg.pinv(B * ginv) @ (B * ginv))[:, 0] * 0
null = np.linalg.svd(B)[2][3:].T
cost = lambda x: float(x @ (x / ginv))
check("min-norm equality: no feasible null-space move lowers the cost",
      all(cost(v) <= cost(v + 1e-3 * null[:, i]) + 1e-12 for i in range(null.shape[1])))

# LDP: projection of the origin onto a half-space and onto an infeasible set
w = ldp(np.array([[1.0, 1.0]]), np.array([2.0]))
check("ldp: min ||w|| s.t. w1 + w2 >= 2 is (1, 1)", w is not None and np.allclose(w, [1.0, 1.0], atol=1e-8))
check("ldp: contradictory constraints are infeasible", ldp(np.array([[1.0], [-1.0]]), np.array([1.0, 0.0])) is None)
w = ldp(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([-1.0, 3.0]))
check("ldp: inactive constraint ignored (solution (0, 3))", w is not None and np.allclose(w, [0.0, 3.0], atol=1e-8))

# fixed-cell edit on a small random encoder
L, d, k = 12, 6, 3
W = rng.standard_normal((L, d)); b = rng.standard_normal(L)
x_std = rng.uniform(0.5, 2.0, d)
u = rng.standard_normal(d)
pre = W @ u + b
z = topk_relu(pre[None, :], k)[0]
winners = list(np.flatnonzero(z > 0))
if len(winners) == k:
    t = winners[0]
    res = fixed_cell_edit(pre, W, x_std, winners, {t: 0.1 * z[t]}, {p: 1e-6 for p in winners[1:]}, 1e-6)
    if res["status"] == "feasible":
        v = res["w"] / x_std
        z_new = topk_relu((pre + W @ v)[None, :], k)[0]
        check("fixed cell: target change realized after re-encoding", abs((z_new - z)[t] - 0.1 * z[t]) < 1e-6)
        check("fixed cell: protected winners unchanged within tolerance", all(abs((z_new - z)[p]) <= 1e-6 + 1e-9 for p in winners[1:]))
        check("fixed cell: support unchanged", set(np.flatnonzero(z_new > 0)) == set(winners))
        free = min_norm_equality((W / x_std[None, :])[[t]], np.array([0.1 * z[t]]), np.ones(d))
        check("fixed cell: movement at least the equality-only minimum", res["raw_movement"] >= np.linalg.norm(free) - 1e-9)
    else:
        check(f"fixed cell: small feasible instance solved (status {res['status']})", False)
    big = fixed_cell_edit(pre, W, x_std, winners, {winners[0]: -z[winners[0]] - 1.0}, {}, 1e-6)
    check("fixed cell: request below zero is reported as leaving the cell", big["status"] == "request_leaves_cell")
else:
    check("fixed cell: test instance has k positive winners", False)

print(f"{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
