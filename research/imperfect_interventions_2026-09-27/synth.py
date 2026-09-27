#!/usr/bin/env python3
"""Known-answer synthetic study: interpreting near-zero effects of imperfect SAE feature interventions.

EXPLORATORY research prototype (2026-09-27), CPU only (numpy/scipy). See README.md in this folder for
the estimand, derivations and decision rules. Nothing here is a claim about the CERT or TWOS models.

Each synthetic world has
  * a raw hidden state x in R^d with standardized coordinates u = x / sigma (sigma non-uniform);
  * a TopK(ReLU(W u + beta)) SAE encoder whose rows span a visible subspace V; directions in
    V-perp ("null" directions n1, n2) are invisible to the SAE;
  * a decoder D (tied to W unless the world adds a null-space component to the selected atom);
  * a downstream loss F_i(x) = softplus(-s_i(u)) with a known score s (linear readout, optional
    small MLP, optional interaction term), so exact effects and valid regional Lipschitz bounds are
    available;
  * per unit an original state x_O and a profile-replaced state x_R (the corruption).

The intervention restores the selected feature's SAE code from O into R while holding every other
coordinate of the code at its R value (the full-code target tau). Ideal interventions:
  I2  decoder-anchored: the exact projection of the declared decoder edit onto the code-tau polyhedron;
  I1  minimum change: the exact projection of x_R onto the same polyhedron.
The applied state x_hat is the declared decoder edit with an optional execution error.

  python3 synth.py dev   --config config_dev.json   --out results/dev
  python3 synth.py eval  --config config_frozen.json --out results/eval
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize, nnls
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

# numpy 2.0 with Apple Accelerate raises spurious floating-point flags in some matmuls; the products are
# finite and match einsum (checked). Only that message is filtered, and project() asserts finiteness.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

SEL, PROT = 0, 1          # selected feature and the protected feature that is always active


def softplus(t):
    return np.logaddexp(0.0, t)


def sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def topk_relu(p, k):
    relu = np.maximum(p, 0.0)
    idx = np.argsort(-relu, kind="stable")[:k]
    z = np.zeros_like(relu)
    z[idx] = relu[idx]
    return z


class World:
    """One synthetic world. All SAE quantities are in standardized coordinates unless named raw."""

    def __init__(self, spec: dict, seed: int):
        rng = np.random.default_rng(seed)
        self.spec, self.seed = spec, seed
        d, dV, m, k = spec["d"], spec["d_visible"], spec["m"], spec["k"]
        self.d, self.m, self.k = d, m, k
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        V = Q[:, :dV]
        self.n1, self.n2 = Q[:, dV], Q[:, dV + 1]
        if m > dV:
            raise ValueError("this study uses m <= d_visible so that crosstalk is a controlled knob")
        B0, _ = np.linalg.qr(rng.standard_normal((dV, dV)))
        W = (V @ B0[:, :m]).T                         # orthonormal encoder rows in V
        chi = spec.get("crosstalk", 0.0)
        if chi > 0:                                   # controlled crosstalk between atoms
            W = W + chi * (V @ rng.standard_normal((dV, m))).T / np.sqrt(dV)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        kap = spec.get("coherence", 0.0)
        if kap > 0:                                   # protected row nearly collinear with the selected row
            g = V @ rng.standard_normal(dV)
            g -= (g @ W[SEL]) * W[SEL]; g /= np.linalg.norm(g)
            W[PROT] = kap * W[SEL] + np.sqrt(1 - kap ** 2) * g
        self.W = W
        self.beta = np.full(m, spec["bias"])
        D = W.copy()                                  # tied, unit rows: W_j . D_j = 1
        if spec.get("decoder_null", 0.0):
            D[SEL] = W[SEL] + spec["decoder_null"] * self.n1
        self.D = D
        lo, hi = spec["sigma_range"]
        self.sigma = rng.uniform(lo, hi, d)
        self.R = W / self.sigma[None, :]              # raw-coordinate encoder rows: p = R x + beta
        self.w = spec["alpha_sel"] * W[SEL] + spec["alpha_prot"] * W[PROT] + spec["gamma"] * self.n1
        self.eta = spec.get("eta", 0.0)
        h = spec.get("mlp_hidden", 0)
        if h:
            self.U = rng.standard_normal((h, d)) * spec["mlp_in_scale"] / np.sqrt(d)
            if spec.get("mlp_orth_sel"):              # the MLP does not read the selected encoder direction
                self.U = self.U @ (np.eye(d) - np.outer(W[SEL], W[SEL]))
            self.c0 = rng.standard_normal(h) * 0.5
            self.v = rng.standard_normal(h) * spec["mlp_out_scale"] / np.sqrt(h)
        else:
            self.U = None
        self.offset0 = 0.0

    # ---- encoder ----
    def pre(self, x):
        return self.R @ x + self.beta

    def code(self, x):
        return topk_relu(self.pre(x), self.k)

    # ---- downstream score, loss and gradients (raw coordinates) ----
    def score(self, x, off):
        u = x / self.sigma
        s = self.w @ u + self.eta * (self.W[SEL] @ u) * (self.n2 @ u) + off
        if self.U is not None:
            s = s + self.v @ np.tanh(self.U @ u + self.c0)
        return s

    def F(self, x, off):
        return float(softplus(-self.score(x, off)))

    def grad_F(self, x, off):
        u = x / self.sigma
        g = self.w + self.eta * ((self.n2 @ u) * self.W[SEL] + (self.W[SEL] @ u) * self.n2)
        if self.U is not None:
            g = g + self.U.T @ (self.v * (1 - np.tanh(self.U @ u + self.c0) ** 2))
        return -sigmoid(-self.score(x, off)) * g / self.sigma

    def K_certified(self, xa, xb):
        """Valid Lipschitz bound for F on the ball containing the segment [xa, xb] (raw Euclidean)."""
        inv = 1.0 / self.sigma
        xc, rho = 0.5 * (xa + xb), 0.5 * np.linalg.norm(xa - xb)
        uc, rho_u = xc * inv, rho * inv.max()
        lin = np.linalg.norm(self.w)
        if self.U is not None:
            lin += np.linalg.norm(self.v) * np.linalg.norm(self.U, 2)
        a, b = self.W[SEL], self.n2
        inter = abs(self.eta) * (np.linalg.norm(a) * (abs(b @ uc) + np.linalg.norm(b) * rho_u)
                                 + np.linalg.norm(b) * (abs(a @ uc) + np.linalg.norm(a) * rho_u))
        return inv.max() * (lin + inter)            # |softplus'| <= 1

    def K_sampled(self, pts, off):
        """Largest gradient norm at observed states: an empirical diagnostic, not a bound."""
        return max(np.linalg.norm(self.grad_F(p, off)) for p in pts)

    # ---- data ----
    def unit_states(self, rng):
        sp = self.spec
        m, kind = self.m, sp["corruption"]
        noise = (np.eye(self.d) - np.outer(self.n1, self.n1) - np.outer(self.n2, self.n2)) @ rng.standard_normal(self.d) * sp["noise"]
        pool = np.arange(2, m)
        j3, j4, r = rng.choice(pool, size=3, replace=False)
        aO = np.zeros(m)
        om = rng.uniform(0.8, 1.2); bO = rng.uniform(0.8, 1.2)
        if kind in ("reduce", "swap"):
            aO[SEL] = rng.uniform(2.0, 3.0); aO[PROT] = rng.uniform(1.5, 2.5); aO[j3] = rng.uniform(1.5, 3.0)
        else:                                         # delete: the selected feature is absent in O
            aO[PROT] = rng.uniform(1.5, 2.5); aO[j3] = rng.uniform(1.5, 3.0); aO[j4] = rng.uniform(1.5, 3.0)
        aR = aO.copy()
        if kind == "reduce":
            aR[SEL] *= 1 - rng.uniform(0.4, 0.7)
        elif kind == "swap":
            aR[SEL] = 0.0; aR[r] = rng.uniform(1.5, 2.5)
        else:
            aR[SEL] = rng.uniform(2.0, 3.0)
        aR[PROT] *= 1 - rng.uniform(*sp["prot_corruption"])
        omR = om - rng.uniform(0.6, 1.0)
        bR = 0.0 if sp.get("corrupt_n2") else bO
        uO = self.D.T @ aO + om * self.n1 + bO * self.n2 + noise
        uR = self.D.T @ aR + omR * self.n1 + bR * self.n2 + noise
        return self.sigma * uO, self.sigma * uR, {"bO": bO, "bR": bR}


SOLVER_STATS = {"nnls": 0, "dual_lbfgsb": 0}


def ldp(G, h):
    """argmin ||z|| s.t. G z >= h, or None if infeasible.

    Lawson and Hanson's NNLS route first (as in scripts/sae_edit_geometry.py, with a larger iteration cap).
    If NNLS stalls on an ill-conditioned instance, solve the convex dual min_{lam >= 0} 1/2 ||G^T lam||^2 - h^T lam
    with L-BFGS-B and accept z = G^T lam only if it is primal feasible and the duality gap is small.
    """
    m, n = G.shape
    E = np.vstack([G.T, h[None, :]]); f = np.zeros(n + 1); f[-1] = 1.0
    try:
        u, _ = nnls(E, f, maxiter=500 * max(m, n))
        r = E @ u - f
        if np.linalg.norm(r) < 1e-12 or abs(r[-1]) < 1e-14:
            return None
        SOLVER_STATS["nnls"] += 1
        return -r[:n] / r[-1]
    except RuntimeError:
        pass
    GG = G @ G.T
    fun = lambda lam: (0.5 * lam @ GG @ lam - h @ lam, GG @ lam - h)
    res = minimize(fun, np.zeros(m), jac=True, method="L-BFGS-B", bounds=[(0, None)] * m,
                   options={"maxiter": 50000, "ftol": 1e-15, "gtol": 1e-12})
    z = G.T @ res.x
    scale = 1 + np.abs(h).max()
    primal, dual = 0.5 * z @ z, -(0.5 * res.x @ GG @ res.x - h @ res.x)
    if (G @ z - h).min() < -1e-7 * scale or primal - dual > 1e-6 * (1 + primal):
        return None
    SOLVER_STATS["dual_lbfgsb"] += 1
    return z


def build_P(world, tau, margin):
    """Closed code-tau polyhedron {x : B x = dvec, A x <= a} in raw coordinates, or None if over-k."""
    S = np.flatnonzero(tau > 0)
    if len(S) > world.k:
        return None
    others = np.setdiff1d(np.arange(world.m), S)
    thr = (tau[S].min() - margin) if len(S) == world.k else -margin
    return world.R[S], tau[S] - world.beta[S], world.R[others], thr - world.beta[others]


def project(y, P):
    """Exact Euclidean projection of y onto P (null-space elimination + least-distance programming).

    Returns (x, dist, eq_dist, ineq_active) or None when P is empty.
    dist^2 = eq_dist^2 + ||z||^2, where eq_dist is the distance to the affine hull {Bx = d}.
    """
    B, dvec, A, a = P
    U_, sv, Vt = np.linalg.svd(B, full_matrices=True)
    r = int((sv > 1e-10 * sv[0]).sum())
    res = dvec - B @ y
    corr = Vt[:r].T @ ((U_[:, :r].T @ res) / sv[:r])
    x0 = y + corr
    if np.linalg.norm(B @ x0 - dvec) > 1e-8 * (1 + np.linalg.norm(dvec)):
        return None
    h = A @ x0 - a
    if h.max() <= 1e-12:
        return x0, float(np.linalg.norm(corr)), float(np.linalg.norm(corr)), False
    N = Vt[r:].T
    G = -A @ N
    assert np.isfinite(G).all() and np.isfinite(h).all()
    z = ldp(G, h)
    if z is None or (G @ z - h).min() < -1e-7:
        lp = linprog(np.zeros(N.shape[1]), A_ub=-G, b_ub=-h, bounds=[(None, None)] * N.shape[1], method="highs")
        if lp.status == 2:
            return None
        raise RuntimeError("projection solver disagreement")
    x = x0 + N @ z
    return x, float(np.linalg.norm(x - y)), float(np.linalg.norm(corr)), True


def hoffman_eq_bound(y, P):
    """||B y - d|| / sigma_min(B): the equality-subsystem Hoffman bound (valid for dist(y, P) only when the
    equality projection already satisfies the inequalities)."""
    B, dvec, _, _ = P
    sv = np.linalg.svd(B, compute_uv=False)
    return float(np.linalg.norm(B @ y - dvec) / sv[sv > 1e-10 * sv[0]].min())


def unit_record(world, rng, off, spec, margin):
    xO, xR, nul = world.unit_states(rng)
    zO, zR = world.code(xO), world.code(xR)
    S = [SEL]
    tau = zR.copy(); tau[S] = zO[S]
    req = tau[S] - zR[S]
    x_dec = xR + world.sigma * (world.D[S].T @ req)
    ex = spec["exec"]
    step = x_dec - xR
    xi = rng.standard_normal(world.d); xi *= ex.get("noise", 0.0) * np.linalg.norm(step) / max(np.linalg.norm(xi), 1e-12)
    x_hat = xR + ex.get("scale", 1.0) * step + xi
    FR, FO, Fh = world.F(xR, off), world.F(xO, off), world.F(x_hat, off)
    zh = world.code(x_hat)
    notS = np.setdiff1d(np.arange(world.m), S)
    rec = {"FR": FR, "FO": FO, "full": FR - FO, "applied": FR - Fh,
           "req_norm": float(np.linalg.norm(req)), "tgt_err": float(np.linalg.norm(zh[S] - tau[S])),
           "off_err": float(np.linalg.norm(zh[notS] - tau[notS])), "exec_err": float(np.linalg.norm(x_hat - x_dec)),
           "Ks": world.K_sampled([xR, x_hat, xO], off)}
    P = build_P(world, tau, margin)
    pr2 = project(x_dec, P) if P is not None else None
    pr1 = project(xR, P) if P is not None else None
    rec["feasible"] = pr2 is not None and pr1 is not None
    rec["over_k"] = P is None
    if rec["feasible"]:
        x2, dist_dec, eqd, ineq = pr2
        x1 = pr1[0]
        for x_ in (x1, x2):                            # the projected states must carry the target code
            zc = world.code(x_)
            assert np.allclose(zc, tau, atol=1e-6), "projection does not realize the target code"
        rec.update({"ideal2": FR - world.F(x2, off), "ideal1": FR - world.F(x1, off),
                    "d_hat_to_ideal2": float(np.linalg.norm(x_hat - x2)), "d_dec_to_P": dist_dec,
                    "ineq_active": bool(ineq), "Kc": world.K_certified(x_hat, x2),
                    "hoff_eq": hoffman_eq_bound(x_dec, P) if not ineq else float("inf"),
                    "hoff_ratio": (hoffman_eq_bound(x_dec, P) / dist_dec) if (not ineq and dist_dec > 1e-12) else float("nan"),
                    "d1_move": float(np.linalg.norm(x1 - xR)), "d2_move": float(np.linalg.norm(x2 - xR))})
        # structural role: noising the selected feature in the clean context (decoder-anchored ideal)
        tn = zO.copy(); tn[S] = zR[S]
        Pn = build_P(world, tn, margin)
        prn = project(xO + world.sigma * (world.D[S].T @ (tn[S] - zO[S])), Pn) if Pn is not None else None
        rec["necessity"] = (world.F(prn[0], off) - FO) if prn is not None else float("nan")
        # interaction partner B = the n2 component (invisible to the SAE)
        dB = world.sigma * world.n2 * (nul["bO"] - nul["bR"])
        rec["rescue_B"] = FR - world.F(xR + dB, off)
        rec["rescue_joint"] = FR - world.F(x2 + dB, off)
    else:
        # a declared alternative for over-k targets: swap-in (the weakest non-selected winner of R may drop)
        alt = np.nan
        if P is None:
            ta = tau.copy()
            win = [j for j in np.flatnonzero(zR > 0) if j not in S]
            ta[min(win, key=lambda j: zR[j])] = 0.0
            Pa = build_P(world, ta, margin)
            pa = project(x_dec, Pa) if Pa is not None else None
            alt = FR - world.F(pa[0], off) if pa is not None else np.nan
        rec["alt_swapin"] = float(alt)
    return rec


def make_pool(spec, world_seed, n_clusters, per_cluster, margin, max_tries=40):
    world = World(spec, world_seed)
    rng = np.random.default_rng(world_seed + 7919)
    # calibrate the offset so the clean score is about +2.5 on average (clean predictions are good)
    probe = [world.unit_states(np.random.default_rng(world_seed + 17 + i)) for i in range(200)]
    world.offset0 = 2.5 - float(np.mean([world.score(p[0], 0.0) for p in probe]))
    recs, cl, rejected = [], [], 0
    for g in range(n_clusters):
        ce = rng.normal(0, 0.5)
        got = 0
        while got < per_cluster:
            off = world.offset0 + ce + rng.normal(0, 0.3)
            r = unit_record(world, rng, off, spec, margin)
            if spec["corruption"] == "reduce" and not r["feasible"]:
                rejected += 1                        # the reduce worlds are defined on feasible targets
                if rejected > max_tries * n_clusters:
                    raise RuntimeError("too many infeasible targets")
                continue
            recs.append(r); cl.append(g); got += 1
    return world, recs, np.asarray(cl), rejected


def cluster_mean(vals, cl):
    u = np.unique(cl)
    return np.array([vals[cl == g].mean() for g in u])


def truth(recs, cl):
    out = {}
    for key in ("full", "applied", "ideal2", "ideal1", "necessity", "rescue_B", "rescue_joint", "alt_swapin"):
        v = np.array([r.get(key, np.nan) for r in recs], dtype=float)
        ok = ~np.isnan(v)
        out[key] = float(cluster_mean(v[ok], cl[ok]).mean()) if ok.any() else float("nan")
    if not np.isnan(out["ideal2"]):
        out["interaction"] = out["rescue_joint"] - out["ideal2"] - out["rescue_B"]
    out["share_feasible"] = float(np.mean([r["feasible"] for r in recs]))
    out["share_over_k"] = float(np.mean([r["over_k"] for r in recs]))
    return out


def decide(lo, hi, tau):
    if -tau < lo and hi < tau:
        return "negligible"
    if lo >= tau or hi <= -tau:
        return "non-negligible"
    return "inconclusive"


def run_procedures(recs, cl, sel_clusters, tau, cfg, rng):
    idx = np.flatnonzero(np.isin(cl, sel_clusters))
    R = [recs[i] for i in idx]; c = cl[idx]
    G = len(sel_clusters)
    draws = rng.integers(0, G, size=(cfg["boot"], G))
    groups = [np.flatnonzero(c == g) for g in sel_clusters]

    method = cfg.get("interval_method", "percentile")

    def cmeans(vals):
        return np.array([vals[gi].mean() for gi in groups])

    def boot(vals):
        return ("t", cmeans(vals)) if method == "t" else ("b", cmeans(vals)[draws].mean(1))

    def q(bv, lev):
        kind, v = bv
        if kind == "t":
            half = student_t.ppf(1 - (1 - lev) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            return float(v.mean() - half), float(v.mean() + half)
        return float(np.quantile(v, (1 - lev) / 2)), float(np.quantile(v, 1 - (1 - lev) / 2))
    lev = cfg["interval_level"]
    e = np.array([r["applied"] for r in R])
    out = {}
    lo, hi = q(boot(e), lev)
    out["P1_applied"] = {"lo": lo, "hi": hi, "decision": decide(lo, hi, tau)}
    lo0, hi0 = q(boot(e), 0.95)
    out["P0_zero_in_ci"] = {"lo": lo0, "hi": hi0, "decision": "negligible" if lo0 <= 0 <= hi0 else "non-negligible"}
    real = (sum(r["tgt_err"] + r["off_err"] for r in R) / max(sum(r["req_norm"] for r in R), 1e-12))
    out["P2_realization_gated"] = {"lo": lo, "hi": hi, "realization_error": real,
                                   "decision": out["P1_applied"]["decision"] if real <= cfg["realization_threshold"] else "inconclusive"}
    feasible = all(r["feasible"] for r in R)
    if not feasible:
        for name in ("P3_direct_ideal", "P4a_bound_certifiedK", "P4b_bound_sampledK", "P4c_hoffman_eq_certifiedK", "P5_two_lifts"):
            out[name] = {"decision": "ill-posed"}
        return out
    e2 = np.array([r["ideal2"] for r in R]); e1 = np.array([r["ideal1"] for r in R])
    lo, hi = q(boot(e2), lev); out["P3_direct_ideal"] = {"lo": lo, "hi": hi, "decision": decide(lo, hi, tau)}
    for name, Bv in (("P4a_bound_certifiedK", np.array([r["Kc"] * r["d_hat_to_ideal2"] for r in R])),
                     ("P4b_bound_sampledK", np.array([r["Ks"] * r["d_hat_to_ideal2"] for r in R])),
                     ("P4c_hoffman_eq_certifiedK", np.array([r["Kc"] * (r["exec_err"] + r["hoff_eq"]) for r in R]))):
        if not np.isfinite(Bv).all():
            out[name] = {"decision": "insufficient (bound unavailable)"}
            continue
        lo = q(boot(e - Bv), lev)[0]; hi = q(boot(e + Bv), lev)[1]
        out[name] = {"lo": lo, "hi": hi, "mean_bound": float(Bv.mean()), "decision": decide(lo, hi, tau)}
    lo1, hi1 = q(boot(e1), lev)
    d1, d2 = decide(lo1, hi1, tau), out["P3_direct_ideal"]["decision"]
    if d1 == d2:
        dec = d1
    elif {d1, d2} == {"negligible", "non-negligible"}:
        dec = "not identified (lift-dependent)"
    else:
        dec = "inconclusive"
    out["P5_two_lifts"] = {"lo1": lo1, "hi1": hi1, "decision": dec}
    return out


def run(cfg, out_dir: Path, stage: str):
    t0 = time.process_time(); w0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for cname, cond in cfg["conditions"].items():
        spec = {**cfg["base_spec"], **cond.get("spec", {})}
        spec["exec"] = {**cfg["base_spec"].get("exec", {}), **cond.get("spec", {}).get("exec", {})}
        seeds = cfg["world_seeds"][stage]
        for ws in seeds:
            world, recs, cl, rej = make_pool(spec, ws, cfg["pool_clusters"], cfg["units_per_cluster"], cfg["margin"])
            tr = truth(recs, cl)
            tau = cfg["margin_fraction"] * abs(tr["full"])
            rng = np.random.default_rng(ws + 104729)
            for rep in range(cfg["datasets_per_world"]):
                sel = rng.choice(cfg["pool_clusters"], size=cfg["clusters_per_dataset"], replace=False)
                res = run_procedures(recs, cl, sel, tau, cfg, rng)
                for proc, rr in res.items():
                    rows.append({"condition": cname, "world_seed": ws, "rep": rep, "procedure": proc, "tau": tau,
                                 "truth_ideal2": tr["ideal2"], "truth_ideal1": tr["ideal1"], "truth_applied": tr["applied"],
                                 "truth_full": tr["full"], **{k: v for k, v in rr.items()}})
            summary.setdefault(cname, []).append({"world_seed": ws, "rejected_infeasible_units": rej, "tau": tau, **tr,
                                                  "mean_realization_error": float(sum(r["tgt_err"] + r["off_err"] for r in recs) / sum(r["req_norm"] for r in recs)),
                                                  "mean_d_hat_to_ideal2": float(np.nanmean([r.get("d_hat_to_ideal2", np.nan) for r in recs])),
                                                  "mean_Kc": float(np.nanmean([r.get("Kc", np.nan) for r in recs])),
                                                  "mean_Ks": float(np.nanmean([r["Ks"] for r in recs])),
                                                  "share_ineq_active": float(np.nanmean([r.get("ineq_active", np.nan) for r in recs])),
                                                  "hoffman_eq_over_exact_median": float(np.nanmedian([r.get("hoff_ratio", np.nan) for r in recs])) if any(np.isfinite(r.get("hoff_ratio", np.nan)) for r in recs) else None,
                                                  "hoffman_eq_over_exact_max": float(np.nanmax([r.get("hoff_ratio", np.nan) for r in recs])) if any(np.isfinite(r.get("hoff_ratio", np.nan)) for r in recs) else None})
        print(f"[{stage}] {cname} done ({time.time() - w0:.0f}s)", flush=True)
    (out_dir / "rows.json").write_text(json.dumps(rows) + "\n")
    (out_dir / "worlds.json").write_text(json.dumps(summary, indent=1) + "\n")
    tab = evaluate(rows)
    (out_dir / "evaluation.json").write_text(json.dumps(tab, indent=1) + "\n")
    timing = {"cpu_s": time.process_time() - t0, "wall_s": time.time() - w0, "solver_paths": dict(SOLVER_STATS)}
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(timing))
    return tab


def evaluate(rows):
    """Per condition and procedure: decision shares, false negligible, coverage of the I2 ideal effect, width."""
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["condition"]][r["procedure"]].append(r)
    tab = {}
    for c, procs in agg.items():
        tab[c] = {}
        for p, rs in procs.items():
            n = len(rs)
            dec = defaultdict(int)
            for r in rs:
                dec[r["decision"]] += 1
            truth_nonneg = [abs(r["truth_ideal2"]) >= r["tau"] if not np.isnan(r["truth_ideal2"]) else None for r in rs]
            fn = sum(1 for r, t in zip(rs, truth_nonneg) if t and r["decision"] == "negligible")
            n_nonneg = sum(1 for t in truth_nonneg if t)
            cov = [r["lo"] - 1e-9 <= r["truth_ideal2"] <= r["hi"] + 1e-9 for r in rs if "lo" in r and not np.isnan(r["truth_ideal2"])]
            truth_neg = [abs(r["truth_ideal2"]) < r["tau"] if not np.isnan(r["truth_ideal2"]) else None for r in rs]
            fp = sum(1 for r, t in zip(rs, truth_neg) if t and r["decision"] == "non-negligible")
            n_neg = sum(1 for t in truth_neg if t)
            def truth_class(r):
                if np.isnan(r["truth_ideal2"]):
                    return "ill-posed"
                if p == "P5_two_lifts":
                    c1, c2 = abs(r["truth_ideal1"]) >= r["tau"], abs(r["truth_ideal2"]) >= r["tau"]
                    return "not identified (lift-dependent)" if c1 != c2 else ("non-negligible" if c2 else "negligible")
                return "non-negligible" if abs(r["truth_ideal2"]) >= r["tau"] else "negligible"
            correct = sum(1 for r in rs if r["decision"] == truth_class(r))
            wid = [r["hi"] - r["lo"] for r in rs if "lo" in r]
            tab[c][p] = {"n": n, "decisions": {k: v / n for k, v in sorted(dec.items())},
                         "false_negligible_rate": (fn / n_nonneg) if n_nonneg else None,
                         "false_nonnegligible_rate": (fp / n_neg) if n_neg else None,
                         "correct_rate": correct / n,
                         "coverage_of_ideal2": (float(np.mean(cov)) if cov else None),
                         "median_width": (float(np.median(wid)) if wid else None)}
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "eval"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    print("config sha256", hashlib.sha256(Path(a.config).read_bytes()).hexdigest())
    run(cfg, Path(a.out), a.stage)
