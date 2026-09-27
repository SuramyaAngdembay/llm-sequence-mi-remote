#!/usr/bin/env python3
"""Part 3 precheck: feasibility and exact projection of declared feature interventions on captured states.

CPU only (numpy/scipy); no behavioural outcome. See PRECHECK_PROTOCOL.md for the declared targets,
metric, margins, verification arithmetic and pass criteria.

  python3 project_targets.py --capture DIR --sae SAE.npz --out DIR [--limit N]

Arithmetic used for every new target and every verification ("path B", H6's verification path):
delta = float32(x_bf16) - float32(b_bf16); u = (delta - x_mean) / x_std; pre = u @ W^T + b; ReLU; TopK(k); float32.
A projected state is cast float64 -> float32 -> bf16 (round to nearest even), which is what the model receives
when the pilot writes it with put(add=False), and is then re-encoded with path B.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

MARGINS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)          # declared schedule, code (preactivation) units


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_f32(x):
    """Round float32 to bfloat16 (round to nearest even) and return the value as float32."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


class SAE:
    def __init__(self, path):
        z = np.load(path)
        self.W = z["W"].astype(np.float32); self.b = z["b"].astype(np.float32); self.D = z["D"].astype(np.float32)
        self.mu = z["x_mean"].astype(np.float32); self.sd = z["x_std"].astype(np.float32); self.k = int(z["k"])
        self.R = (self.W.astype(np.float64) / self.sd.astype(np.float64)[None, :])       # raw-coordinate rows

    def pre_B(self, x_f32, b_f32):
        u = ((x_f32.astype(np.float32) - b_f32.astype(np.float32)) - self.mu) / self.sd
        return u @ self.W.T + self.b

    def code_B(self, x_f32, b_f32):
        p = np.maximum(self.pre_B(x_f32, b_f32), 0)
        idx = np.argsort(-p, kind="stable")[: self.k]
        return {int(i): float(p[i]) for i in idx if p[i] > 0}, p

    def g(self, b_f32):
        """Affine offset in float64: pre(x) = R x + g for the base state b."""
        return self.b.astype(np.float64) - self.W.astype(np.float64) @ ((b_f32.astype(np.float64) + self.mu) / self.sd)


def declared_target(z_ctx, z_src, feats, p_ctx, k):
    """Target in the context code z_ctx with coordinates `feats` taken from z_src.

    Returns (fixed {feature: value}, refill [features], dropped [features], kind) where kind is
    'full_code' (exactly k fixed), 'swap_in' (over-k: weakest non-selected context winners dropped) or
    'refill' (under-k: highest-preactivation non-winners of the context admitted with a free value).
    """
    fixed = {f: v for f, v in z_ctx.items() if f not in feats}
    for f in feats:
        if z_src.get(f, 0.0) > 0:
            fixed[f] = z_src[f]
    dropped, refill, kind = [], [], "full_code"
    if len(fixed) > k:
        kind = "swap_in"
        cands = sorted((f for f in fixed if f not in feats), key=lambda f: (z_ctx[f], f))
        while len(fixed) > k:
            f = cands.pop(0); dropped.append(f); del fixed[f]
    elif len(fixed) < k:
        kind = "refill"
        order = np.argsort(-p_ctx, kind="stable")
        for f in order:
            f = int(f)
            if len(fixed) + len(refill) == k:
                break
            if f not in fixed and f not in feats and f not in z_ctx:
                refill.append(f)
    return fixed, refill, dropped, kind


class Constraints:
    """Implicit rows for the declared target, raw coordinates, with pre(x) = R x + g.

    Equalities:   pre_f(x) = tau_f for every fixed f.
    Inequalities (violation <= 0 means satisfied):
      type L (every loser l):          pre_l(x) - (min fixed value - margin)
      type Q (every loser l, refill r): pre_l(x) - pre_r(x) + margin
      type P (every refill r):          margin - pre_r(x)
    """

    def __init__(self, sae, gvec, fixed, refill, margin):
        self.R, self.g, self.m = sae.R, gvec, margin
        self.F = np.array(sorted(fixed), dtype=int); self.vals = np.array([fixed[f] for f in self.F])
        self.refill = list(refill)
        used = np.concatenate([self.F, np.array(self.refill, dtype=int)])
        self.losers = np.setdiff1d(np.arange(self.R.shape[0]), used)
        self.thr = self.vals.min() - margin if len(self.F) else np.inf
        self.E = self.R[self.F]; self.e = self.vals - gvec[self.F]
        self.nL = len(self.losers); self.nQ = self.nL * len(self.refill); self.nP = len(self.refill)

    def violations(self, x):
        p = self.R @ x + self.g
        v = [p[self.losers] - self.thr]
        for r in self.refill:
            v.append(p[self.losers] - p[r] + self.m)
        if self.refill:
            v.append(np.array([self.m - p[r] for r in self.refill]))
        return np.concatenate(v)

    def rows(self, idx):
        """Materialize rows a and right-hand sides c (a x <= c) for implicit indices idx."""
        A, c = [], []
        for t in idx:
            if t < self.nL:
                l = self.losers[t]; A.append(self.R[l]); c.append(self.thr - self.g[l])
            elif t < self.nL + self.nQ:
                q = t - self.nL; r = self.refill[q // self.nL]; l = self.losers[q % self.nL]
                A.append(self.R[l] - self.R[r]); c.append(self.g[r] - self.g[l] - self.m)
            else:
                r = self.refill[t - self.nL - self.nQ]; A.append(-self.R[r]); c.append(self.g[r] - self.m)
        return np.asarray(A), np.asarray(c)


def project(y, K, tol=1e-7, max_rounds=40, add_per_round=256):
    """argmin ||x - y|| s.t. the constraints K. Dual active set with constraint generation."""
    work = np.zeros(0, dtype=int); Mw = np.zeros((0, y.size)); cw = np.zeros(0)
    stats = {"rounds": 0}
    ne = K.E.shape[0]
    for rnd in range(max_rounds):
        A = np.vstack([K.E, Mw]); rhs = np.concatenate([K.e, cw])
        G = A @ A.T; h = A @ y - rhs
        if len(work) == 0:
            nu = np.linalg.lstsq(G, h, rcond=None)[0]
        else:
            fun = lambda v: (0.5 * v @ G @ v - h @ v, G @ v - h)
            r = minimize(fun, np.zeros(len(h)), jac=True, method="L-BFGS-B",
                         bounds=[(None, None)] * ne + [(0, None)] * len(work),
                         options={"maxiter": 20000, "ftol": 1e-16, "gtol": 1e-12, "maxcor": 50})
            nu = r.x
        x = y - A.T @ nu
        viol = K.violations(x)
        stats.update(rounds=rnd + 1, working=int(len(work)), max_ineq_violation=float(viol.max()),
                     max_eq_residual=float(np.abs(K.E @ x - K.e).max()))
        bad = np.setdiff1d(np.flatnonzero(viol > tol), work)
        if len(bad) == 0:
            stats["status"] = "ok" if (stats["max_ineq_violation"] <= 1e-5 and stats["max_eq_residual"] <= 1e-6) else "inaccurate"
            lam = nu[ne:]
            stats["min_multiplier"] = float(lam.min()) if len(lam) else 0.0
            stats["active_rows"] = int((lam > 1e-9).sum()) if len(lam) else 0
            # KKT certificate for the convex QP: stationarity holds by construction (x = y - A^T nu); check
            # primal feasibility (above), dual feasibility (lam >= 0) and complementary slackness on working rows.
            slack = (cw - Mw @ x) if len(work) else np.zeros(0)
            stats["max_complementarity"] = float(np.max(np.abs(lam * slack))) if len(lam) else 0.0
            if stats["status"] == "ok" and (stats["min_multiplier"] < -1e-9 or stats["max_complementarity"] > 1e-6):
                stats["status"] = "kkt_failed"
            return x, stats
        add = bad[np.argsort(-viol[bad])][:add_per_round]
        Ma, ca = K.rows(add)
        work = np.concatenate([work, add]); Mw = np.vstack([Mw, Ma]); cw = np.concatenate([cw, ca])
    stats["status"] = "no_convergence"
    return None, stats


def verify(sae, x_star, b_f32, fixed, refill, tol_abs=0.05, tol_rel=0.02):
    """Cast as the model would (float64 -> float32 -> bf16) and re-encode with path B."""
    xc = f32_to_bf16_f32(x_star.astype(np.float32))
    z, p = sae.code_B(xc, b_f32)
    want = set(fixed) | set(refill)
    ok_support = set(z) == want
    errs = {f: abs(z.get(f, 0.0) - v) for f, v in fixed.items()}
    ok_vals = all(errs[f] <= max(tol_abs, tol_rel * abs(fixed[f])) for f in fixed)
    return ok_support and ok_vals, {"support_ok": ok_support, "values_ok": ok_vals,
                                    "max_abs_fixed_err": float(max(errs.values())) if errs else 0.0,
                                    "realized": z}, xc


def solve_token(sae, y, x_ctx, b, fixed, refill, gvec):
    last = None
    for m in MARGINS:
        K = Constraints(sae, gvec, fixed, refill, m)
        t0 = time.process_time()
        x, st = project(y.astype(np.float64), K)
        st["cpu_s"] = time.process_time() - t0
        if x is None or st["status"] != "ok":
            last = {"margin": m, "solver": st, "verified": False, "reason": "solver"}
            continue
        ok, info, xc = verify(sae, x, b, fixed, refill)
        last = {"margin": m, "solver": st, "verified": ok, "verify": {k_: v for k_, v in info.items() if k_ != "realized"},
                "move_from_anchor": float(np.linalg.norm(xc - y)), "move_from_context": float(np.linalg.norm(xc - x_ctx)),
                "x": xc}
        if ok:
            return last
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True); ap.add_argument("--sae", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cpu0, w0 = time.process_time(), time.time()
    sae = SAE(a.sae)
    meta = json.loads((a.capture / "h6_capture_meta.json").read_text())
    S, C, k = meta["selected"], meta["control"], sae.k
    st = np.load(a.capture / "h6_capture_states.npz")
    xR, bR, xO, bO = (bf16_to_f32(st[n]) for n in ("xR", "bR", "xO", "bO"))
    n = len(xR) if not a.limit else min(a.limit, len(xR))
    # CPU versus GPU arithmetic: recompute path-B codes of the unedited states and compare with the capture
    tk = np.load(a.capture / "h6_capture_tokens.npz")
    key = {(int(r), int(d)): j for j, (r, d) in enumerate(zip(tk["recv"], tk["dst"]))}
    cpu_gpu = {"tokens": 0, "support_mismatch": 0, "max_abs_value_diff": 0.0}
    records, xs_out = [], {}
    for i in range(n):
        j = key[(int(st["recv"][i]), int(st["dst"][i]))]
        zR, pR = sae.code_B(xR[i], bR[i]); zO, pO = sae.code_B(xO[i], bO[i])
        gR = {int(q): float(v) for q, v in zip(tk["R_B_idx"][j], tk["R_B_val"][j]) if v > 0}
        cpu_gpu["tokens"] += 1
        cpu_gpu["support_mismatch"] += set(gR) != set(zR)
        if set(gR) == set(zR):
            cpu_gpu["max_abs_value_diff"] = max(cpu_gpu["max_abs_value_diff"], max(abs(gR[q] - zR[q]) for q in zR))
        gRv, gOv = sae.g(bR[i]), sae.g(bO[i])
        rec = {"i": i, "recv": int(st["recv"][i]), "dst": int(st["dst"][i])}
        for name, feats, ctx, z_ctx, z_src, p_ctx, b_ctx, gvec in (
                ("restore_sel", S, xR[i], zR, zO, pR, bR[i], gRv),
                ("restore_ctl", C, xR[i], zR, zO, pR, bR[i], gRv),
                ("noise_sel", S, xO[i], zO, zR, pO, bO[i], gOv)):
            req = np.array([z_src.get(f, 0.0) - z_ctx.get(f, 0.0) for f in feats])
            if not np.any(req):
                continue
            fixed, refill, dropped, kind = declared_target(z_ctx, z_src, feats, p_ctx, k)
            x_dec = ctx.astype(np.float64) + sae.sd.astype(np.float64) * (sae.D[:, feats].astype(np.float64) @ req)
            r = {"kind": kind, "dropped": dropped, "refill": refill, "req_sq": float((req ** 2).sum()),
                 "decoder_edit_norm": float(np.linalg.norm(x_dec - ctx))}
            anchors = {"I2_decoder": x_dec} if name != "restore_sel" else {"I2_decoder": x_dec, "I1_min_change": ctx.astype(np.float64)}
            for an, y in anchors.items():
                out = solve_token(sae, y, ctx.astype(np.float64), b_ctx, fixed, refill, gvec)
                xs = out.pop("x", None)
                r[an] = out
                if xs is not None and out["verified"]:
                    xs_out[f"{name}__{an}__{i}"] = xs.astype(np.float32)
            rec[name] = r
        rec["full_restoration_norm"] = float(np.linalg.norm(xO[i].astype(np.float64) - xR[i].astype(np.float64)))
        records.append(rec)
        if (i + 1) % 25 == 0:
            print(f"[project] {i + 1}/{n} tokens ({time.time() - w0:.0f}s)", flush=True)
    np.savez_compressed(a.out / "projected_states.npz", **xs_out)
    (a.out / "projection_records.json").write_text(json.dumps(records) + "\n")
    summ = {"cpu_vs_gpu_path_B_codes": cpu_gpu, "tokens": n, "cpu_s": time.process_time() - cpu0, "wall_s": time.time() - w0,
            "margins": MARGINS}
    (a.out / "projection_run.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
