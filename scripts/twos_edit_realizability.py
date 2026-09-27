#!/usr/bin/env python3
"""Phase 2 of the 2026-09-26 TWOS feasibility pilot (EXPLORATORY; cached activations, no LM forward pass).

For each preselected receiver/donor edit (layer 24, m = 4, k = 8; selected and
control feature sets fixed by the corrected discovery ranking), compute

  z        = E(u)                          original code, u = (delta - x_mean) / x_std
  z_req    = union or own-support request  (alpha 1, package-4 donor prototype)
  u_new    = u + D (z_req - z)             residual-preserving decoder edit
  z_real   = E(u_new)                      realized code under the frozen encoder

and, as the simple baseline the specification asks for, the minimum
raw-movement encoder projection v that sets the targeted preactivations to the
requested values (ignoring TopK competition and other features).

Populations: development = discovery users' positive windows, each with up to
16 benign same-team donors of other users chosen by sha256 order (metadata
only); evaluation = the exact receiver/donor pairs scored in the corrected run
(candidate rows CSV), which are confirmation users and therefore exploratory.

Encoder-realizability statements (for example, a requested code with more
than k nonzeros) are about this encoder only. They are not evidence that the
edited model state is unnatural or that its causal effect is invalid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from pilot_intervention_specificity import edited_codes  # noqa: E402
from sae_edit_geometry import exact_code_realizable, fixed_cell_edit, min_norm_equality, topk_relu  # noqa: E402

EXPECTED = {"ranking": "12b1069034954d01a83232a6f2c73be39b2f04e3a2e8454116f159659bf03ce5",
            "candidate_rows": "2fbfe5f2da58d7a18fd700ecf88d93f98c67aebea12a90ee34348359145922e4",
            "confirmation_users": "a17e75c4ebe1e7264f3610c7630eaa8d5cad58e3d757a4b23ea278c2f0b66cd3",
            "discovery_users": "44b1f0324522c36b31bff93d90c84ed32528ee29d9965e32713655592f894f41"}
SALT = "twos-dev-donors-2026-09-26"
CHANGED = 1e-4          # a code coordinate counts as changed above this (code units)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def edit_metrics(z, z_new, z_req, X, k, x_std, D, recon_raw, move_raw):
    """Per-token metrics for one edit (rows = edited tokens)."""
    notX = np.setdiff1d(np.arange(z.shape[1]), X)
    req_T = (z_req - z)[:, X]; real_T = (z_new - z)[:, X]
    req_n = np.linalg.norm(req_T, axis=1)
    off = (z_new - z)[:, notX]
    sz, sn = z > 0, z_new > 0
    out = {
        "req_norm": req_n,
        "realized_frac": np.sum(real_T * req_T, axis=1) / np.maximum(req_n ** 2, 1e-30),
        "target_rel_err": np.linalg.norm(real_T - req_T, axis=1) / np.maximum(req_n, 1e-30),
        "collateral_l2": np.linalg.norm(off, axis=1),
        "n_off_changed": (np.abs(off) > CHANGED).sum(axis=1),
        "new_active_off": (sn & ~sz)[:, notX].sum(axis=1),
        "deleted_active_off": (sz & ~sn)[:, notX].sum(axis=1),
        "jaccard": (sz & sn).sum(axis=1) / np.maximum((sz | sn).sum(axis=1), 1),
        "nnz_req": (z_req > 0).sum(axis=1),
        "raw_move": move_raw, "recon_raw": recon_raw,
    }
    out["collateral_rel"] = out["collateral_l2"] / np.maximum(req_n, 1e-30)
    out["over_k"] = out["nnz_req"] > k
    # per target coordinate: writes into zero, and requested decreases
    zero_writes = (z[:, X] == 0) & (z_req[:, X] > 0)
    became = zero_writes & (z_new[:, X] > 0)
    dec = (z[:, X] > 0) & (z_req[:, X] < z[:, X])
    dec_real = dec & (np.abs((z_new - z_req)[:, X]) <= 0.1 * np.abs((z_req - z)[:, X]))
    dropped = dec & (z_new[:, X] == 0) & (z_req[:, X] > 0)
    out["n_zero_writes"] = zero_writes.sum(axis=1); out["n_zero_writes_became_active"] = became.sum(axis=1)
    out["n_decreases"] = dec.sum(axis=1); out["n_decreases_within_10pct"] = dec_real.sum(axis=1)
    out["n_decreases_dropped_out"] = dropped.sum(axis=1)
    return out


def summarize(recs):
    """recs: list of dicts of per-token arrays with 'user' key; returns token-level and per-user summaries."""
    if not recs:
        return {"n_tokens": 0}
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0] if k not in ("user", "receiver")}
    users = np.concatenate([np.full(len(r["req_norm"]), r["user"]) for r in recs])
    q = lambda a: [float(np.quantile(a, x)) for x in (0.1, 0.5, 0.9)]
    s = {"n_tokens": int(len(cat["req_norm"])), "n_edits": len(recs),
         "realized_frac_q10_50_90": q(cat["realized_frac"]), "target_rel_err_q10_50_90": q(cat["target_rel_err"]),
         "collateral_rel_q10_50_90": q(cat["collateral_rel"]), "n_off_changed_mean": float(cat["n_off_changed"].mean()),
         "new_active_off_mean": float(cat["new_active_off"].mean()), "deleted_active_off_mean": float(cat["deleted_active_off"].mean()),
         "jaccard_mean": float(cat["jaccard"].mean()), "share_over_k": float(cat["over_k"].mean()),
         "nnz_req_max": int(cat["nnz_req"].max()), "raw_move_median": float(np.median(cat["raw_move"])),
         "recon_raw_median": float(np.median(cat["recon_raw"])),
         "zero_writes": int(cat["n_zero_writes"].sum()), "zero_writes_became_active": int(cat["n_zero_writes_became_active"].sum()),
         "decreases": int(cat["n_decreases"].sum()), "decreases_within_10pct": int(cat["n_decreases_within_10pct"].sum()),
         "decreases_dropped_out": int(cat["n_decreases_dropped_out"].sum())}
    per_user = {}
    for u in sorted(set(users)):
        m = users == u
        per_user[u] = {"tokens": int(m.sum()), "realized_frac_median": float(np.median(cat["realized_frac"][m])),
                       "collateral_rel_median": float(np.median(cat["collateral_rel"][m])), "share_over_k": float(cat["over_k"][m].mean())}
    s["per_user"] = per_user
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--candidate-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--latent-mult", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--selected", default="6036,7375,1197,7420,3218")
    ap.add_argument("--control", default="7962,972,8082,7232,3157")
    ap.add_argument("--dev-donors", type=int, default=16)
    ap.add_argument("--qp-cells", type=int, default=40)
    ap.add_argument("--eps-grid", default="0.01,0.05,0.10,0.25")
    ap.add_argument("--margin", type=float, default=1e-3)
    ap.add_argument("--realizability-sample", type=int, default=150)
    ap.add_argument("--stage-budget-s", type=float, default=1800.0, help="wall-clock cap per solver stage; partial counts are reported")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from sae_core import add_active_control_feature_sets, choose_feature_sets, load_ranking
    from eval_token_delta_sae_causal import build_example_slices, donor_feature_prototype, load_eval_examples

    t0 = time.time()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    S = [int(v) for v in args.selected.split(",")]; C = [int(v) for v in args.control.split(",")]
    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    hashes = {"ranking": sha(cfg_dir / "delta_sae_top_features.csv"), "candidate_rows": sha(args.candidate_rows),
              "confirmation_users": sha(args.split_dir / "twos_confirmation_users.txt"),
              "discovery_users": sha(args.split_dir / "twos_discovery_users.txt"),
              "sae_model": sha(cfg_dir / "delta_sae_model.pt")}
    for k, v in EXPECTED.items():
        if hashes[k] != v:
            raise SystemExit(f"{k} sha256 {v[:12]} expected, got {hashes[k][:12]}")
    ranking = load_ranking(cfg_dir)                    # validates finiteness
    sets = add_active_control_feature_sets(choose_feature_sets(ranking), ranking, min_active_frac=0.002)
    if sets["top5"] != S or sets["control5_active"] != C:
        raise SystemExit(f"feature sets differ from the ranking rule: {sets['top5']} / {sets.get('control5_active')}")
    conf = [l.strip() for l in (args.split_dir / "twos_confirmation_users.txt").read_text().split() if l.strip()]
    disc = [l.strip() for l in (args.split_dir / "twos_discovery_users.txt").read_text().split() if l.strip()]

    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    idx_of = {e: i for i, e in enumerate(meta["example_id"])}
    team = meta["team"].astype(str).to_numpy() if "team" in meta else meta["text"].str.extract(r"team=(\S+)")[0].to_numpy()
    user = meta["user_id"].astype(str).to_numpy(); y = meta["y"].astype(int).to_numpy()

    # evaluation pairs: exactly the scored ones
    evalp = defaultdict(set)
    for r in csv.DictReader(args.candidate_rows.open()):
        if r["context_mode"] == "team" and float(r["alpha"]) == 1.0:
            evalp[r["donor_type"]].add((idx_of[r["receiver_example_id"]], idx_of[r["donor_example_id"]]))
    # development pairs: discovery positives, benign same-team donors of other users, hash order
    devp = set()
    for i in np.flatnonzero((y == 1) & np.isin(user, disc)):
        pool = [j for j in np.flatnonzero((y == 0) & (team == team[i]) & (user != user[i]))]
        pool.sort(key=lambda j: hashlib.sha256(f"{SALT}|{meta.loc[i, 'example_id']}|{meta.loc[j, 'example_id']}".encode()).hexdigest())
        devp.update((int(i), int(j)) for j in pool[: args.dev_donors])
    pops = {"development|benign": sorted(devp), "evaluation|benign": sorted(evalp["benign"]),
            "evaluation|anomalous": sorted(evalp["anomalous"])}
    for name, pr in pops.items():
        recv_users = {user[i] for i, _ in pr}
        allowed = set(disc) if name.startswith("development") else set(conf)
        if not recv_users <= allowed:
            raise SystemExit(f"{name}: receivers outside the declared users: {recv_users - allowed}")
        if any(user[i] == user[j] for i, j in pr):
            raise SystemExit(f"{name}: same-user donor")
    manifest = {"exploratory": True, "written_before_computation": True, "hashes": hashes, "selected": S, "control": C,
                "layer": args.layer, "k": args.k, "dev_donor_rule": f"sha256('{SALT}|receiver|donor') order, benign, same team, other users, up to {args.dev_donors}",
                "populations": {n: {"pairs": len(p), "receivers": len({i for i, _ in p}), "users": sorted({user[i] for i, _ in p})} for n, p in pops.items()},
                "changed_threshold": CHANGED, "eps_grid": args.eps_grid, "margin": args.margin, "qp_cells": args.qp_cells}
    (out / "phase2_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[phase2] populations {json.dumps(manifest['populations'])}", flush=True)

    # activations for the needed examples only (not the full pool)
    need = sorted({i for p in pops.values() for pair in p for i in pair})
    obj = torch.load(args.extract_dir / f"layer_{args.layer}" / "chunk_00000.pt", map_location="cpu", weights_only=False)
    ex = np.asarray(obj["example_idx"], dtype=np.int64)
    sl = build_example_slices(ex)
    for i in need:
        if sl[i].stop - sl[i].start != int(scores.loc[i, "n_tokens"]):
            raise SystemExit(f"example {i}: delta rows != n_tokens")
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    We = sd["encoder.weight"].double().numpy(); be = sd["encoder.bias"].double().numpy(); D = sd["decoder.weight"].double().numpy()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float64).reshape(-1); x_std = np.asarray(bundle["x_std"], dtype=np.float64).reshape(-1)
    receivers = sorted({i for p in pops.values() for i, _ in p})
    donors = sorted({j for p in pops.values() for _, j in p})
    U, PRE, Z, PROTO = {}, {}, {}, {}
    for i in receivers:                        # full arrays for receivers only
        u = (np.asarray(obj["delta"][sl[i]], dtype=np.float64) - x_mean) / x_std
        U[i] = u; PRE[i] = u @ We.T + be; Z[i] = topk_relu(PRE[i], args.k)
    for j in donors:                           # donors are reduced to their prototypes (package-4 rule)
        zj = Z[j] if j in Z else topk_relu(((np.asarray(obj["delta"][sl[j]], dtype=np.float64) - x_mean) / x_std) @ We.T + be, args.k)
        PROTO[(j, "selected")] = donor_feature_prototype(zj, S); PROTO[(j, "control")] = donor_feature_prototype(zj, C)
    del obj
    # encoder fidelity: the pipeline's float32 torch encoder against the float64 numpy encoder
    from sae_core import TopKSAE
    sae = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])); sae.load_state_dict(sd); sae.eval()
    agree = []
    with torch.no_grad():
        for i in receivers[:40]:
            zt = sae.encode_sparse(torch.from_numpy(U[i].astype(np.float32))).numpy()
            agree.append(float(((zt > 0) == (Z[i] > 0)).all(axis=1).mean()))
    def near_tie(i):
        srt = np.sort(np.maximum(PRE[i], 0), axis=1)
        kth, nxt = srt[:, -args.k], srt[:, -args.k - 1]
        return np.mean((kth > 0) & (kth - nxt < 1e-6))
    near_tie = float(np.mean([near_tie(i) for i in receivers]))
    positive_k = float(np.mean([np.mean((Z[i] > 0).sum(axis=1) == args.k) for i in receivers]))
    fidelity = {"support_agreement_torch_float32_vs_numpy_float64": float(np.mean(agree)), "share_tokens_near_tie_kth": near_tie,
                "share_tokens_with_exactly_k_positive": positive_k}
    print(f"[phase2] encoder fidelity {fidelity} ({time.time() - t0:.0f}s)", flush=True)

    recon_raw = {i: np.linalg.norm((U[i] - Z[i] @ D.T) * x_std, axis=1) for i in receivers}
    g_inv = 1.0 / x_std ** 2
    results, token_rows, qp_candidates, real_candidates = {}, [], defaultdict(list), defaultdict(list)
    zero_effect = defaultdict(int)
    for pop, pairs in pops.items():
        recs = defaultdict(list)
        for i, j in pairs:
            for arm, X in (("selected", S), ("control", C)):
                proto = PROTO[(j, arm)]
                for rule in ("union", "own"):
                    z_req = edited_codes(Z[i], X, proto, rule)
                    rows = np.flatnonzero((z_req != Z[i]).any(axis=1))
                    if rows.size == 0:
                        zero_effect[f"{pop}|{arm}|{rule}"] += 1
                        continue
                    z0, zr = Z[i][rows], z_req[rows]
                    dz = zr - z0
                    # decoder edit
                    u_new = U[i][rows] + dz @ D.T
                    z_dec = topk_relu(u_new @ We.T + be, args.k)
                    m_dec = edit_metrics(z0, z_dec, zr, X, args.k, x_std, D, recon_raw[i][rows], np.linalg.norm((dz @ D.T) * x_std, axis=1))
                    m_dec["user"] = user[i]; recs[f"{arm}|{rule}|decoder_edit"].append(m_dec)
                    # encoder projection: targeted preactivations set to the requested values
                    z_proj = np.zeros_like(z0); move_proj = np.zeros(len(rows))
                    for r_, t_row in enumerate(rows):
                        T = [f for f in X if zr[r_, f] != z0[r_, f]]
                        c = np.array([zr[r_, f] - PRE[i][t_row, f] for f in T])
                        v = min_norm_equality(We[T], c, g_inv)
                        z_proj[r_] = topk_relu(((U[i][t_row] + v) @ We.T + be)[None, :], args.k)[0]
                        move_proj[r_] = np.linalg.norm(v * x_std)
                    m_proj = edit_metrics(z0, z_proj, zr, X, args.k, x_std, D, recon_raw[i][rows], move_proj)
                    m_proj["user"] = user[i]; recs[f"{arm}|{rule}|encoder_projection"].append(m_proj)
                    for r_, t_row in enumerate(rows):
                        key = f"{pop}|{arm}|{rule}"
                        tag = hashlib.sha256(f"{key}|{i}|{j}|{t_row}".encode()).hexdigest()
                        real_candidates[key].append((tag, i, t_row, zr[r_]))
                        if rule == "own" and (z0[r_] > 0).sum() == args.k:
                            qp_candidates[f"{pop}|{arm}"].append((tag, i, j, t_row, {f: float(zr[r_, f] - z0[r_, f]) for f in X if zr[r_, f] != z0[r_, f]},
                                                                  float(m_dec["raw_move"][r_]), float(m_dec["realized_frac"][r_]), float(m_dec["collateral_rel"][r_])))
        results[pop] = {k: summarize(v) for k, v in recs.items()}
        print(f"[phase2] {pop} done ({time.time() - t0:.0f}s)", flush=True)
    results["zero_effect_edits"] = dict(zero_effect)
    results["encoder_fidelity"] = fidelity
    (out / "phase2_edit_metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"[phase2] edit metrics written ({time.time() - t0:.0f}s)", flush=True)

    # exact realizability of requested codes (LP), bounded hash-ordered sample
    real, t_stage = {}, time.time()
    for key, cands in sorted(real_candidates.items()):
        cands.sort(key=lambda c: c[0])
        counts, done = defaultdict(int), 0
        for _, i, t_row, zr in cands[: args.realizability_sample]:
            if time.time() - t_stage > args.stage_budget_s:
                counts["not_run_budget"] += 1
                continue
            counts[exact_code_realizable(PRE[i][t_row], We, zr, args.k, args.margin)] += 1
            done += 1
        real[key] = {"n": min(len(cands), args.realizability_sample), **counts}
        print(f"[phase2] realizability {key}: {dict(counts)} ({time.time() - t0:.0f}s)", flush=True)
    results["exact_realizability_sample"] = real
    print(f"[phase2] realizability done ({time.time() - t0:.0f}s)", flush=True)

    # fixed-cell minimum-movement QP on own-support requests at exactly-k-positive tokens
    eps_grid = [float(e) for e in args.eps_grid.split(",")]
    qp, t_stage = {}, time.time()
    for key, cands in sorted(qp_candidates.items()):
        cands.sort(key=lambda c: c[0])
        per_eps = {e: defaultdict(list) for e in eps_grid}
        for _, i, j, t_row, targets, dec_move, dec_frac, dec_coll in cands[: args.qp_cells]:
            if time.time() - t_stage > args.stage_budget_s:
                for e in eps_grid:
                    per_eps[e]["status"].append("not_run_budget")
                continue
            pre = PRE[i][t_row]; z0 = Z[i][t_row]
            winners = [int(f) for f in np.flatnonzero(z0 > 0)]
            for e in eps_grid:
                eps = {p: e * z0[p] for p in winners if p not in targets}
                res = fixed_cell_edit(pre, We, x_std, winners, targets, eps, args.margin)
                per_eps[e]["status"].append(res["status"])
                if res["status"] == "feasible":
                    v = res["w"] / x_std
                    z_new = topk_relu(((U[i][t_row] + v) @ We.T + be)[None, :], args.k)[0]
                    tgt = list(targets)
                    req = np.array([targets[f] for f in tgt]); got = (z_new - z0)[tgt]
                    per_eps[e]["target_err_after_reencode"].append(float(np.abs(got - req).max()))
                    per_eps[e]["support_kept"].append(bool(set(np.flatnonzero(z_new > 0)) == set(winners)))
                    per_eps[e]["move_ratio_qp_over_decoder"].append(res["raw_movement"] / max(dec_move, 1e-12))
                per_eps[e]["decoder_realized_frac"].append(dec_frac); per_eps[e]["decoder_collateral_rel"].append(dec_coll)
        qp[key] = {str(e): {"n_cells": len(v["status"]), "status_counts": {s: v["status"].count(s) for s in set(v["status"])},
                            "median_move_ratio_qp_over_decoder": float(np.median(v["move_ratio_qp_over_decoder"])) if v["move_ratio_qp_over_decoder"] else None,
                            "max_target_err_after_reencode": float(max(v["target_err_after_reencode"])) if v["target_err_after_reencode"] else None,
                            "share_support_kept": float(np.mean(v["support_kept"])) if v["support_kept"] else None,
                            "decoder_realized_frac_median_same_cells": float(np.median(v["decoder_realized_frac"])),
                            "decoder_collateral_rel_median_same_cells": float(np.median(v["decoder_collateral_rel"]))}
                   for e, v in per_eps.items()}
        print(f"[phase2] fixed-cell QP {key}: " + json.dumps({e: q_["status_counts"] for e, q_ in qp[key].items()}) + f" ({time.time() - t0:.0f}s)", flush=True)
    results["fixed_cell_qp"] = qp
    results["encoder_fidelity"] = fidelity
    results["seconds"] = round(time.time() - t0)
    (out / "phase2_results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k in ("zero_effect_edits", "exact_realizability_sample", "fixed_cell_qp", "encoder_fidelity")}, indent=1)[:6000], flush=True)
    print(f"PHASE2_DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
