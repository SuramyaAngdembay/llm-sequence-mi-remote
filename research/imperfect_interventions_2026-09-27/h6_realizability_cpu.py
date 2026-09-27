#!/usr/bin/env python3
"""CPU-only checks of whether H6's declared selected-feature target was realizable (no model inference).

EXPLORATORY research, 2026-09-27. Two parts:

  local  (laptop)  From the saved H6 token codes, count edited session tokens whose declared
                   target (selected features from O, every other feature held at its R value)
                   asks for more selected actives than R has. With exactly k active features in
                   every R code, such a target has more than k nonzeros and no hidden state
                   realizes it under TopK.
  anvil  (Anvil CPU, cached layer-26 deltas and the frozen SAE) Verify the "exactly k active"
                   premise, measure the TopK winner margin, and simulate the bf16 subtraction
                   rounding that separates H6's intended and verification arithmetic.

  python3 h6_realizability_cpu.py local --codes results/h6_2026_09_27/out/h6_token_codes.npz --out OUT.json
  python h6_realizability_cpu.py anvil --share /anvil/scratch/x-sangdembay/pkg4_share --out OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def local(args):
    z = np.load(args.codes)
    feats = [int(f) for f in z["feats"]]
    S = feats[:5]
    ids = sorted({k.split("__")[0] for k in z.files if "__" in k})
    rows = []
    for rid in ids:
        o, r = z[f"{rid}__O_sess"][:, :5], z[f"{rid}__R_sess"][:, :5]
        d = o - r
        for t in np.flatnonzero(np.linalg.norm(d, axis=1) > 0):
            rows.append({"receiver": rid, "n_sel_active_O": int((o[t] > 0).sum()), "n_sel_active_R": int((r[t] > 0).sum()),
                         "req_sq": float((d[t] ** 2).sum()), "involves_7693": bool(abs(d[t, S.index(7693)]) > 0)})
    n = len(rows)
    over = [r for r in rows if r["n_sel_active_O"] > r["n_sel_active_R"]]
    under = [r for r in rows if r["n_sel_active_O"] < r["n_sel_active_R"]]
    tot = sum(r["req_sq"] for r in rows)
    res = {
        "edited_tokens": n,
        "target_over_k_if_R_has_exactly_k": len(over),
        "target_under_k_deletion": len(under),
        "same_selected_support_size": n - len(over) - len(under),
        "share_requested_squared_change_on_over_k_tokens": sum(r["req_sq"] for r in over) / tot,
        "share_requested_squared_change_involving_7693": sum(r["req_sq"] for r in rows if r["involves_7693"]) / tot,
        "receivers_with_over_k_tokens": len({r["receiver"] for r in over}),
        "note": "Over-k holds exactly when every R code has k active features; the anvil part checks that premise.",
    }
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def anvil(args):
    import torch
    S = Path(args.share)
    b = torch.load(S / "frontier/layer_26/m02_k04/delta_sae_model.pt", map_location="cpu", weights_only=False)
    sd = b["state_dict"]; W = sd["encoder.weight"].float(); be = sd["encoder.bias"].float(); k = int(b["k"])
    xm = torch.tensor(np.asarray(b["x_mean"], dtype=np.float32).reshape(-1))
    xs = torch.tensor(np.asarray(b["x_std"], dtype=np.float32).reshape(-1))
    X = torch.as_tensor(np.asarray(torch.load(S / "token_deltas/layer_26/chunk_00000.pt", map_location="cpu", weights_only=False)["delta"]))
    x = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(args.seed))[:args.n]].float()
    pre = ((x - xm) / xs) @ W.T + be
    npos = (pre > 0).sum(1)
    v, _ = torch.topk(pre, k + 1, dim=1)

    def code(d):
        p = torch.relu(((d - xm) / xs) @ W.T + be); vv, ii = torch.topk(p, k, dim=1)
        return torch.zeros_like(p).scatter_(1, ii, vv)

    q = lambda t, ps: [float(u) for u in torch.quantile(t.float(), torch.tensor(ps))]
    res = {"k": k, "sae_width": int(W.shape[0]), "tokens": int(x.shape[0]), "source": "layer_26/chunk_00000.pt, original inputs",
           "share_tokens_with_at_least_k_positive_preactivations": float((npos >= k).float().mean()),
           "positive_preactivations_q001_q01_q50": q(npos, [0.001, 0.01, 0.5]),
           "kth_minus_k1th_preactivation_q05_q50_q95": q(v[:, k - 1] - v[:, k], [0.05, 0.5, 0.95]),
           "kth_winner_value_q05_q50_q95": q(v[:, k - 1], [0.05, 0.5, 0.95]),
           "cache_exactly_representable_in_bf16": bool((x.bfloat16().float() == x).all())}
    z0 = code(x); g = torch.Generator().manual_seed(args.seed + 1)
    for label, scale in (("half_ulp", 2.0 ** -9), ("one_ulp", 2.0 ** -8)):
        xp = x * (1 + (torch.rand(x.shape, generator=g) * 2 - 1) * scale)
        z1 = code(xp); a, c = z0 > 0, z1 > 0
        jac = (a & c).sum(1).float() / (a | c).sum(1).float()
        rel = (z1 - z0).norm(dim=1) / z0.norm(dim=1)
        res[f"bf16_subtraction_{label}"] = {"support_identical_share": float((jac == 1).float().mean()),
                                            "rel_code_change_q50_q90_q99": q(rel, [0.5, 0.9, 0.99])}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("local"); a.add_argument("--codes", required=True); a.add_argument("--out", required=True)
    b_ = sub.add_parser("anvil"); b_.add_argument("--share", required=True); b_.add_argument("--out", required=True)
    b_.add_argument("--n", type=int, default=20000); b_.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    local(args) if args.cmd == "local" else anvil(args)
