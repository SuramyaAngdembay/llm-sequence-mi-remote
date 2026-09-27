#!/usr/bin/env python3
"""Geometry of SAE feature edits under a TopK(ReLU(W u + b)) encoder (numpy/scipy only).

Used by the 2026-09-26 TWOS feasibility pilot. All vectors are in the SAE's
standardized coordinates u = (delta - x_mean) / x_std unless stated; raw
residual movement is x_std * v. Established optimization only (least-norm
solutions, least-distance programming via NNLS, LP feasibility with HiGHS);
nothing here is claimed as a new method.

Terminology used in reports:
  requested code   z_req, the code an edit rule asks for
  realized code    E(u + v), what the frozen encoder returns after the edit
  fixed cell       the region where the set of k positive winners is unchanged;
                   E is affine there, so code changes of winners equal
                   preactivation changes
A fixed-cell infeasibility concerns that cell and those constraints only.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import linprog, nnls


def topk_relu(pre: np.ndarray, k: int) -> np.ndarray:
    """TopK(ReLU(pre)) row-wise; ties broken by index order (argsort is stable)."""
    relu = np.maximum(pre, 0.0)
    order = np.argsort(-relu, axis=1, kind="stable")[:, :k]
    z = np.zeros_like(relu)
    rows = np.arange(relu.shape[0])[:, None]
    z[rows, order] = relu[rows, order]
    return z


def min_norm_equality(B: np.ndarray, c: np.ndarray, g_inv: np.ndarray) -> np.ndarray:
    """argmin_v v^T G v s.t. B v = c, with G = diag(1/g_inv); pseudoinverse handles rank deficiency."""
    BG = B * g_inv[None, :]
    M = BG @ B.T
    return g_inv * (B.T @ (np.linalg.pinv(M) @ c))


def ldp(Gm: np.ndarray, h: np.ndarray) -> Optional[np.ndarray]:
    """Least-distance programming: argmin ||w|| s.t. Gm w >= h (Lawson and Hanson, ch. 23), or None if infeasible."""
    m, n = Gm.shape
    E = np.vstack([Gm.T, h[None, :]])
    f = np.zeros(n + 1); f[-1] = 1.0
    u, _ = nnls(E, f, maxiter=50 * max(m, n))
    r = E @ u - f
    if np.linalg.norm(r) < 1e-12 or abs(r[-1]) < 1e-14:
        return None
    return -r[:n] / r[-1]


def lp_small_move(A_ub, b_ub, A_eq=None, b_eq=None):
    """Feasibility LP with an L1 movement objective (v = p - q, p, q >= 0), so solutions stay near the origin.

    Returns (feasible: bool, v or None). A zero objective would return arbitrary distant vertices,
    which makes constraint generation slow; the objective does not affect the feasibility verdict.
    """
    n = A_ub.shape[1] if A_ub is not None else A_eq.shape[1]
    A_ub2 = None if A_ub is None else np.hstack([A_ub, -A_ub])
    A_eq2 = None if A_eq is None else np.hstack([A_eq, -A_eq])
    res = linprog(np.ones(2 * n), A_ub=A_ub2, b_ub=b_ub, A_eq=A_eq2, b_eq=b_eq, bounds=[(0, None)] * (2 * n),
                  method="highs", options={"primal_feasibility_tolerance": 1e-9})
    if res.status == 2:
        return False, None
    if res.status != 0:
        raise RuntimeError(res.message)
    return True, res.x[:n] - res.x[n:]


def lp_feasible(A_ub, b_ub, A_eq=None, b_eq=None) -> bool:
    return lp_small_move(A_ub, b_ub, A_eq, b_eq)[0]


def fixed_cell_edit(pre: np.ndarray, W: np.ndarray, x_std: np.ndarray, winners: Sequence[int],
                    targets: Dict[int, float], eps: Dict[int, float], margin: float,
                    working_losers: int = 64, max_rounds: int = 20) -> dict:
    """Minimum raw-movement edit inside the current TopK cell.

    pre      preactivations W u + b of the token (length L)
    W        encoder weight (L x d) in standardized coordinates
    winners  the k positive winners of the token (the fixed cell)
    targets  {feature: requested code change}; each must be a winner
    eps      {protected winner: allowed |code change|}
    Constraints (in raw-movement coordinates w = x_std * v, R_f = W_f / x_std):
      R_t w = a_t (targets); |R_p w| <= eps_p (protected);
      pre_s + R_s w >= margin (winners stay positive);
      pre_s + R_s w >= pre_l + R_l w + margin for every loser l (winners stay on top).
    Losers enter by constraint generation: start with the highest `working_losers`,
    solve, check every loser, add violators, repeat. Infeasibility with a subset of
    losers certifies infeasibility of the full set (it is a relaxation).
    Returns status 'feasible' (with the minimum-norm w), 'infeasible_fixed_cell', or
    'request_leaves_cell' when a requested value falls below the margin.
    """
    winners = list(winners)
    for t, a in targets.items():
        if t not in winners:
            raise ValueError("fixed-cell targets must be current winners")
        if pre[t] + a < margin:
            return {"status": "request_leaves_cell", "feature": int(t)}
    R = W / x_std[None, :]
    losers_all = np.setdiff1d(np.arange(len(pre)), winners)
    work = list(losers_all[np.argsort(-pre[losers_all])[:working_losers]])
    for _ in range(max_rounds):
        rows, rhs = [], []                                   # all as  row . w >= rhs
        for t, a in targets.items():
            rows += [R[t], -R[t]]; rhs += [a, -a]
        for p, e in eps.items():
            rows += [R[p], -R[p]]; rhs += [-e, -e]
        for s in winners:
            rows.append(R[s]); rhs.append(margin - pre[s])
            for l in work:
                rows.append(R[s] - R[l]); rhs.append(pre[l] - pre[s] + margin)
        G = np.asarray(rows); h = np.asarray(rhs)
        w = ldp(G, h)
        if w is None or (G @ w - h).min() < -1e-7:
            # confirm infeasibility with an LP before reporting it (the LDP verdict is numerical)
            if not lp_feasible(-G, -h):
                return {"status": "infeasible_fixed_cell", "n_losers_checked": len(work)}
            return {"status": "solver_disagreement", "n_losers_checked": len(work)}
        new_pre = pre + R @ w
        top_winner_floor = min(new_pre[s] for s in winners)
        viol = [int(l) for l in losers_all if new_pre[l] > top_winner_floor - margin and l not in work]
        if not viol:
            return {"status": "feasible", "w": w, "raw_movement": float(np.linalg.norm(w)), "n_losers_checked": len(work),
                    "slack_min_constraint": float((G @ w - h).min())}
        work += viol
    return {"status": "no_convergence"}


def exact_code_realizable(pre: np.ndarray, W: np.ndarray, z_req: np.ndarray, k: int, margin: float,
                          working_losers: int = 256, max_rounds: int = 20) -> str:
    """Is there a v with TopK(ReLU(pre + W v)) == z_req exactly (strict margin)? LP feasibility.

    If z_req has more than k nonzeros the answer is 'more_than_k' without solving.
    With support S (|S| = k): pre_S + W_S v = z_S and pre_l + W_l v <= min(z_S) - margin.
    With |S| < k: pre_S + W_S v = z_S and pre_l + W_l v <= 0 for every other l.
    """
    S = np.flatnonzero(z_req > 0)
    if len(S) > k:
        return "more_than_k"
    if len(S) == 0:
        return "empty_request"
    thresh = (z_req[S].min() - margin) if len(S) == k else 0.0
    if thresh < -1e-12 and len(S) == k:
        return "infeasible"
    others = np.setdiff1d(np.arange(len(pre)), S)
    A_eq, b_eq = W[S], z_req[S] - pre[S]
    # closed form first: the minimum-norm solution of the equalities; realizable if it violates nothing
    v0 = min_norm_equality(A_eq, b_eq, np.ones(W.shape[1]))
    new_pre = pre + W @ v0
    viol0 = others[new_pre[others] > thresh + 1e-7]
    if len(viol0) == 0:
        return "realizable"
    work = sorted(set(others[np.argsort(-pre[others])[:working_losers]].tolist()) | set(viol0.tolist()))
    for _ in range(max_rounds):
        feasible, v = lp_small_move(W[work], thresh - pre[work], A_eq, b_eq)
        if not feasible:
            return "infeasible"
        new_pre = pre + W @ v
        viol = [int(l) for l in others if new_pre[l] > thresh + 1e-7 and l not in work]
        if not viol:
            return "realizable"
        work += viol
    return "no_convergence"
