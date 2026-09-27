#!/usr/bin/env python3
"""Checks for synth.py: projection optimality, code realization, bound validity, coordinates."""
import numpy as np
from scipy.optimize import minimize

import synth

BASE = {"d": 12, "d_visible": 9, "m": 8, "k": 3, "crosstalk": 0.3, "bias": -0.05, "sigma_range": [0.6, 1.6], "alpha_sel": 0.8,
        "alpha_prot": 0.8, "gamma": 0.8, "eta": 0.7, "noise": 0.05, "corruption": "reduce", "prot_corruption": [0.2, 0.4],
        "mlp_hidden": 8, "mlp_in_scale": 2.0, "mlp_out_scale": 1.5, "exec": {"scale": 1.0, "noise": 0.0}}


def feasible_instances(n, seed=0):
    w = synth.World(BASE, seed); rng = np.random.default_rng(seed + 1); out = []
    while len(out) < n:
        xO, xR, _ = w.unit_states(rng)
        zO, zR = w.code(xO), w.code(xR)
        tau = zR.copy(); tau[0] = zO[0]
        P = synth.build_P(w, tau, 1e-3)
        if P is None:
            continue
        y = xR + w.sigma * (w.D[[0]].T @ (tau[[0]] - zR[[0]])) + rng.standard_normal(w.d) * 0.5
        pr = synth.project(y, P)
        if pr is not None:
            out.append((w, tau, P, y, pr))
    return out


def test_projection_optimal_and_realizes_code():
    worst = 0.0
    for w, tau, P, y, (x, dist, eqd, ineq) in feasible_instances(25):
        B, dv, A, a = P
        cons = [{"type": "eq", "fun": lambda v, B=B, dv=dv: B @ v - dv}, {"type": "ineq", "fun": lambda v, A=A, a=a: a - A @ v}]
        ref = minimize(lambda v: 0.5 * np.sum((v - y) ** 2), x, constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        assert ref.success
        worst = max(worst, dist - np.linalg.norm(ref.x - y))
        assert np.allclose(w.code(x), tau, atol=1e-6)
        assert eqd <= dist + 1e-9
    assert worst < 1e-6, worst


def test_hoffman_equality_bound_valid_when_inequalities_slack():
    n = 0
    for w, tau, P, y, (x, dist, eqd, ineq) in feasible_instances(40, seed=3):
        if not ineq:
            assert dist <= synth.hoffman_eq_bound(y, P) + 1e-9
            n += 1
    assert n > 0


def test_certified_lipschitz_valid_on_segment():
    w = synth.World(BASE, 5); rng = np.random.default_rng(6)
    for _ in range(200):
        xa, xb = rng.standard_normal(w.d) * 3, rng.standard_normal(w.d) * 3
        K = w.K_certified(xa, xb)
        for lam in rng.uniform(0, 1, 5):
            xm = xa + lam * (xb - xa)
            assert np.linalg.norm(w.grad_F(xm, 0.3)) <= K + 1e-9
        assert abs(w.F(xa, 0.3) - w.F(xb, 0.3)) <= K * np.linalg.norm(xa - xb) + 1e-9


def test_gradient_matches_finite_differences():
    w = synth.World(BASE, 7); rng = np.random.default_rng(8); x = rng.standard_normal(w.d)
    g = w.grad_F(x, 0.2); eps = 1e-6
    fd = np.array([(w.F(x + eps * e, 0.2) - w.F(x - eps * e, 0.2)) / (2 * eps) for e in np.eye(w.d)])
    assert np.allclose(g, fd, atol=1e-6)


def test_raw_versus_standardized_distance_differ():
    # the projection is in raw coordinates; the same move measured in standardized units differs when sigma varies
    w, tau, P, y, (x, dist, _, _) = feasible_instances(1, seed=9)[0]
    assert abs(np.linalg.norm((x - y) / w.sigma) - dist) > 1e-3


def test_over_k_target_is_empty():
    spec = {**BASE, "corruption": "swap"}
    w = synth.World(spec, 11); rng = np.random.default_rng(12)
    xO, xR, _ = w.unit_states(rng)
    zO, zR = w.code(xO), w.code(xR)
    tau = zR.copy(); tau[0] = zO[0]
    if (tau > 0).sum() > w.k:
        assert synth.build_P(w, tau, 1e-3) is None


def test_sampled_lipschitz_is_not_a_bound():
    """Counterexample: gradients sampled at the observed states miss a steep change between x_hat and x*.

    F(t) = h * sigmoid((t - 1/2) / w) along the segment t in [0, 1] (x_hat at t = 0, x* at t = 1). The observed
    states sit at the ends, where the slope is h e^{-1/(2w)} / w. For w = 0.02 the sampled 'bound' is ~1e-9 h
    while the true change is ~h.
    """
    h, w = 1.0, 0.02
    F = lambda t: h / (1 + np.exp(-(t - 0.5) / w))
    dF = lambda t: F(t) * (1 - F(t) / h) / w
    K_sampled = max(dF(0.0), dF(1.0))
    assert K_sampled * 1.0 < 1e-6 * abs(F(1.0) - F(0.0))
    assert abs(F(1.0) - F(0.0)) > 0.99 * h


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
