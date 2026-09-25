#!/usr/bin/env python3
"""H8 (EXPLORATORY, 2026-09-25): is the sparse dictionary necessary?

Same frozen model, receivers, matched benign days, donor policy, positions and
per-token edit sizes as Pilot 1's residual-preserving selected edit (rpU_S).
Only the DIRECTION of the edit changes:

  zero      no edit (base for every condition, same batches)
  rpU_S     the SAE selected set, residual-preserving, union support (reference)
  randI_S   isotropic random direction per token, norm- and position-matched
  randD_S   random other-dictionary direction per token, matched
  pcaTop5   the 5 leading principal directions of the standardized deltas
  pcaGap5   the 5 principal directions (of the leading 64) with the largest
            malicious-minus-benign gap on the ranking population
  pcaCtrl5  the 5 (of the leading 64) with the smallest gap
  meanDiff  the malicious-minus-benign mean-difference direction

For a direction set U (d x k, orthonormal in standardized coordinates), a
receiver token t at a union-support position with standardized delta
z~_t = (delta_t - x_mean) / x_std is moved toward the donor prototype p
(mean of U^T z~ over donor tokens where any selected feature is active):
    edit_t = x_std * U (p - U^T z~_t),   then rescaled to |rpU_S edit_t|.
So every condition edits the same tokens by the same amount; what differs is
where in the residual stream the edit points. A scalar loss change therefore
compares directions, not decompositions as detectors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from pilot_intervention_specificity import (VIEWS, edit_stats, load_delta_cache, shift_random,  # noqa: E402
                                            shift_residual, view_means)


MAX_GAIN = 100.0   # a row is edited only if rescaling amplifies its natural edit by at most this factor


def projection_edit_unscaled(x_rows: np.ndarray, U: np.ndarray, proto: np.ndarray, x_mean: np.ndarray,
                             x_std: np.ndarray) -> np.ndarray:
    """Residual-unit edit that moves each row's projection on span(U) exactly to the prototype."""
    zt = (x_rows - x_mean) / x_std
    return (((proto[None, :] - zt @ U) @ U.T) * x_std).astype(np.float32)


def shift_projection(x_rows: np.ndarray, U: np.ndarray, proto: np.ndarray, x_mean: np.ndarray,
                     x_std: np.ndarray, ref_norms: np.ndarray, max_gain: float = MAX_GAIN) -> np.ndarray:
    """The unscaled edit rescaled per row to ref_norms; rows needing more than max_gain amplification stay zero."""
    unscaled = projection_edit_unscaled(x_rows, U, proto, x_mean, x_std)
    norms = np.linalg.norm(unscaled, axis=1)
    out = np.zeros_like(unscaled, dtype=np.float32)
    ok = norms * max_gain >= ref_norms
    ok &= norms > 0
    out[ok] = unscaled[ok] * (ref_norms[ok] / norms[ok])[:, None]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--delta-cache", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--populations", type=Path, required=True)
    ap.add_argument("--directions", type=Path, required=True, help="npz from pilot_fit_baseline_directions.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--latent-mult", type=int, default=2)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--selected", default="4596,7693,2302,3673,3455")
    ap.add_argument("--control", default="6596,8017,6608,2765,886")
    ap.add_argument("--exclude-users", default="JJM0203")
    ap.add_argument("--pc-pool", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--loss-batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from remote_common import ensure_dir, load_yaml
    from sae_core import TopKSAE, load_ranking
    import token_class_decomposition as tcd
    from eval_token_delta_sae_causal import (build_example_slices, build_sparse_cache_for_examples,
                                             decode_sparse_tokens, donor_feature_prototype, load_eval_examples,
                                             per_example_nll, score_with_token_patches)

    t0 = time.time()
    out = ensure_dir(args.out_dir)
    cfg = load_yaml(args.config)
    S = [int(v) for v in args.selected.split(",")]
    C = [int(v) for v in args.control.split(",")]
    pops = json.loads(args.populations.read_text())
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    dept_of = lambda i: dict(kv.split("=", 1) for kv in meta.loc[i, "text"].split("\n", 1)[0][4:].split())["dept"]
    excluded = set(args.exclude_users.split(",")) if args.exclude_users else set()
    pairs = [(a, b) for a, b in pops["receiver_pairs"] if meta.loc[a, "user_id"] not in excluded]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    receivers = [i for pair in pairs for i in pair]
    kind = {a: "malicious" for a, _ in pairs} | {b: "benign" for _, b in pairs}
    pair_of = {a: n for n, (a, b) in enumerate(pairs)} | {b: n for n, (a, b) in enumerate(pairs)}
    depts = sorted({dept_of(i) for i in receivers})
    donors = {d: [int(v) for v in pops["donor_policy_by_dept"][d]] for d in depts}

    # frozen directions and their provenance
    dirs_npz = np.load(args.directions)
    prov = json.loads(args.directions.with_suffix(".json").read_text())
    U_all = dirs_npz["U"][:, : args.pc_pool].astype(np.float32)
    gap = dirs_npz["gap"][: args.pc_pool].astype(np.float32)
    dvec = dirs_npz["d"].astype(np.float32)
    receiver_users = {meta.loc[i, "user_id"] for i in receivers}
    if set(prov["confirmation_users_in_pos"]) & receiver_users:
        raise SystemExit("the mean-difference direction used a receiver user's positive rows")
    by_gap = np.argsort(-np.abs(gap))
    sets = {"pcaTop5": list(range(5)), "pcaGap5": [int(j) for j in by_gap[:5]],
            "pcaCtrl5": [int(j) for j in by_gap[-5:]]}
    U_of = {name: U_all[:, cols] for name, cols in sets.items()}
    U_of["meanDiff"] = (dvec / np.linalg.norm(dvec))[:, None]

    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    ranking = load_ranking(cfg_dir)
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    active = ranking.set_index("feature_id")["row_active_frac"]
    pool = [int(f) for f in active.index if active[f] >= 0.002 and f not in set(S) | set(C)]
    manifest = {
        "exploratory": True, "written_before_scoring": True, "hypothesis": "H8 (ledger)",
        "adapter_sha256": hashlib.sha256((args.adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "sae_sha256": hashlib.sha256((cfg_dir / "delta_sae_model.pt").read_bytes()).hexdigest(),
        "ranking_sha256": hashlib.sha256((cfg_dir / "delta_sae_top_features.csv").read_bytes()).hexdigest(),
        "populations_sha256": hashlib.sha256(args.populations.read_bytes()).hexdigest(),
        "delta_cache_sha256": hashlib.sha256(args.delta_cache.read_bytes()).hexdigest(),
        "directions_sha256": hashlib.sha256(args.directions.read_bytes()).hexdigest(),
        "directions_provenance": {k: prov[k] for k in ("pca_rows", "pca_users", "pca_population", "ranking_population",
                                                       "pos_rows", "pos_users", "neg_rows", "neg_users", "stride")},
        "layer": args.layer, "selected": S, "alpha": 1.0, "excluded_users": sorted(excluded),
        "n_pairs": len(pairs), "n_users": len(receiver_users), "pc_sets": sets,
        "pc_gaps": [float(v) for v in gap], "variance_explained": [float(v) for v in dirs_npz["evals"][: args.pc_pool] / dirs_npz["total_variance"]],
        "random_dictionary_pool_size": len(pool), "seed": args.seed,
        "conditions": ["zero", "rpU_S", "randI_S", "randD_S", "pcaTop5", "pcaGap5", "pcaCtrl5", "meanDiff"],
        "support_and_size": "every condition edits exactly the tokens rpU_S edits, with rpU_S's per-token norm",
        "endpoints_declared": {
            "E8a": "baseline minus rpU_S, behavior_only, malicious receivers, for each of pcaTop5, pcaGap5, pcaCtrl5, meanDiff (mean of user means; user-clustered bootstrap, 10,000 and 5,000 draws)",
            "E8b": "baseline minus randI_S (and randD_S), same view and receivers",
            "E8c": "the same on matched benign receivers, and the malicious-minus-benign paired difference per user",
            "reading": "if a baseline's E8a interval covers 0 while both beat random, the SAE directions add nothing beyond that baseline at these positions and sizes; if rpU_S exceeds every baseline with intervals excluding 0, the SAE directions are more potent per unit edit; if a baseline exceeds rpU_S, the SAE directions are not the most potent directions of the delta space",
        },
    }
    (out / "pilot4_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[pilot4] {len(pairs)} pairs, {manifest['n_users']} users; PC sets {sets}", flush=True)

    needed = set(receivers) | {i for v in donors.values() for i in v}
    x, token_idx = load_delta_cache(args.delta_cache, needed)
    slices = build_example_slices(token_idx.astype(np.int64, copy=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    sae.load_state_dict(bundle["state_dict"]); sae.eval()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32).reshape(-1)
    x_std = np.asarray(bundle["x_std"], dtype=np.float32).reshape(-1)
    Z = build_sparse_cache_for_examples(sae, x, slices, sorted(needed), x_mean=x_mean, x_std=x_std, device=device, batch_size=2048)
    dec = lambda codes: decode_sparse_tokens(sae, np.asarray(codes, dtype=np.float32), x_mean=x_mean, x_std=x_std,
                                             device=device, batch_size=2048)
    W = sae.decoder.weight.detach().cpu().numpy() * x_std.reshape(-1, 1)
    rdirs = W[:, pool].T
    rdirs = (rdirs / np.linalg.norm(rdirs, axis=1, keepdims=True)).astype(np.float32)

    # prototypes: SAE (package-4 rule) and projections (same donor tokens)
    proto_S, proto_U = {}, {}
    for d_, idxs in donors.items():
        proto_S[d_] = np.mean([donor_feature_prototype(Z[i], S) for i in idxs], axis=0).astype(np.float32)
        proto_U[d_] = {}
        for name, U in U_of.items():
            per_donor = []
            for i in idxs:
                act = Z[i][:, S].sum(axis=1) > 0
                zt = (x[slices[i]][act] - x_mean) / x_std if act.any() else (x[slices[i]] - x_mean) / x_std
                per_donor.append((zt @ U).mean(axis=0))
            proto_U[d_][name] = np.mean(per_donor, axis=0).astype(np.float32)
    manifest["prototypes"] = {d_: {"S": [float(v) for v in proto_S[d_]], **{n: [float(v) for v in p] for n, p in proto_U[d_].items()}}
                              for d_ in donors}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "right":
        raise SystemExit("patching assumes right padding")
    max_len = int(cfg["training"]["max_seq_len"])
    classes, misaligned = {}, []
    for i in receivers:
        text = meta.loc[i, "text"]
        enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
        cls, _ = tcd.classify_tokens([tuple(o) for o in enc["offset_mapping"]], tcd.cert_class_spans(text))
        classes[i] = cls
        if slices[i].stop - slices[i].start != len(cls):
            misaligned.append(int(i))
    if misaligned:
        raise SystemExit(f"delta rows do not align with tokenization for {len(misaligned)} receivers")

    rng_for = lambda i, tag: np.random.default_rng([args.seed, int(i), int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)])
    ref_cache: Dict[int, np.ndarray] = {}

    def ref(i: int) -> np.ndarray:
        if i not in ref_cache:
            ref_cache[i] = shift_residual(Z[i], S, proto_S[dept_of(i)], dec, "union", x.shape[1])
        return ref_cache[i]

    zero_fallbacks = {name: 0 for name in U_of}

    def build(cond: str, i: int) -> np.ndarray:
        delta = x[slices[i]]
        if cond == "zero":
            return np.zeros_like(delta, dtype=np.float32)
        r = ref(i)
        if cond == "rpU_S":
            return r
        if cond == "randI_S":
            return shift_random(r, rng_for(i, cond))
        if cond == "randD_S":
            return shift_random(r, rng_for(i, cond), rdirs)
        U = U_of[cond]
        norms = np.linalg.norm(r, axis=1)
        rows = np.flatnonzero(norms > 0)
        outp = np.zeros_like(delta, dtype=np.float32)
        if rows.size:
            edit = shift_projection(delta[rows], U, proto_U[dept_of(i)][cond], x_mean, x_std, norms[rows])
            zero_fallbacks[cond] += int((np.linalg.norm(edit, axis=1) == 0).sum())
            outp[rows] = edit
        return outp

    # balance (norms and positions are matched by construction; verified here) and validity
    balance_rows, worst_norm_dev, gains = [], 0.0, {name: [] for name in U_of}
    for cond in manifest["conditions"][1:]:
        for i in receivers:
            sh = build(cond, i)
            balance_rows.append({"condition": cond, "receiver_idx": int(i), "kind": kind[i], **edit_stats(sh, classes[i])})
            if cond not in ("zero", "rpU_S"):
                a, b = np.linalg.norm(sh, axis=1), np.linalg.norm(ref(i), axis=1)
                m = (b > 0) & (a > 0)
                if m.any():
                    worst_norm_dev = max(worst_norm_dev, float(np.abs(a[m] / b[m] - 1).max()))
            if cond in U_of:
                rows = np.flatnonzero(np.linalg.norm(ref(i), axis=1) > 0)
                nat = np.linalg.norm(projection_edit_unscaled(x[slices[i]][rows], U_of[cond], proto_U[dept_of(i)][cond], x_mean, x_std), axis=1)
                gains[cond].extend((np.linalg.norm(ref(i), axis=1)[rows] / np.maximum(nat, 1e-12)).tolist())
    gain_stats = {c: {"median": float(np.median(g)), "p95": float(np.quantile(g, 0.95)), "max": float(np.max(g)),
                      "share_above_max_gain": float(np.mean(np.asarray(g) > MAX_GAIN)), "n_tokens": len(g)}
                  for c, g in gains.items() if g}
    bal = pd.DataFrame(balance_rows)
    bal.to_csv(out / "edit_balance_rows.csv", index=False)
    summary = bal.assign(any_edit=bal["n_edited"] > 0).groupby(["condition", "kind"])[
        ["any_edit", "n_tokens", "n_edited", "norm_l2_total", "norm_mean_edited", "edited_DAY", "edited_PSY", "edited_SESCOUNT", "edited_SES"]].mean()
    (out / "edit_balance.json").write_text(summary.reset_index().to_json(orient="records", indent=1))
    # cosine between each baseline edit and the SAE edit, per edited token (how different are the directions?)
    cos = {}
    for cond in U_of:
        vals = []
        for i in receivers[:60]:
            a, b = build(cond, i), ref(i)
            m = np.linalg.norm(b, axis=1) > 0
            vals.extend((np.sum(a[m] * b[m], axis=1) / (np.linalg.norm(a[m], axis=1) * np.linalg.norm(b[m], axis=1) + 1e-12)).tolist())
        cos[cond] = {"mean": float(np.mean(vals)), "abs_mean": float(np.mean(np.abs(vals))), "n_tokens": len(vals)}
    validity = {"token_rows_equal_tokenizer_length": True, "n_receivers": len(receivers),
                "max_relative_norm_deviation_from_rpU_S": worst_norm_dev,
                "rescaling_gain_reference_over_natural_edit": gain_stats, "max_gain": MAX_GAIN,
                "zero_norm_fallbacks_over_balance_and_cosine_builds": zero_fallbacks,
                "cosine_with_rpU_S_edit_first_60_receivers": cos,
                "fit_users_disjoint_from_receivers_for_positive_rows": True}
    (out / "validity.json").write_text(json.dumps(validity, indent=1) + "\n")
    (out / "pilot4_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[pilot4] balance written ({time.time() - t0:.0f}s); validity {json.dumps(validity)[:600]}", flush=True)
    if args.dry_run:
        print("[pilot4] dry run: stopping before the language model", flush=True)
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    q = cfg["quantization"]
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=bool(q["load_in_4bit"]), bnb_4bit_quant_type=str(q["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=getattr(torch, str(q["bnb_4bit_compute_dtype"])),
            bnb_4bit_use_double_quant=bool(q["bnb_4bit_use_double_quant"])),
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.config.use_cache = False
    model.eval()
    texts = [meta.loc[i, "text"] for i in receivers]
    kw = dict(layer=args.layer, max_seq_len=max_len, batch_size=args.batch_size, loss_batch_size=args.loss_batch_size)
    n_chk = min(24, len(texts))
    plain = []
    for s0 in range(0, n_chk, args.batch_size):
        tok = tokenizer(texts[s0:min(s0 + args.batch_size, n_chk)], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        tok = {k: v.to(model.device) for k, v in tok.items()}
        with torch.no_grad():
            o = model(**tok, return_dict=True)
        plain.append(per_example_nll(o.logits, tok["input_ids"], tok["attention_mask"], loss_batch_size=args.loss_batch_size).float().cpu().numpy())
    plain = np.concatenate(plain)
    zero_chk = score_with_token_patches(model, tokenizer, texts[:n_chk], [build("zero", i) for i in receivers[:n_chk]], class_schema="cert", **kw)[0]
    validity["zero_edit_vs_unhooked_max_abs"] = float(np.abs(np.asarray(zero_chk) - plain).max())
    (out / "validity.json").write_text(json.dumps(validity, indent=1) + "\n")
    if validity["zero_edit_vs_unhooked_max_abs"] > 1e-5:
        raise SystemExit("zero edit does not reproduce the unhooked model")

    rows_path = out / "pilot4_rows.csv"
    writer, fh, base_views = None, rows_path.open("w", newline=""), None
    for cond in manifest["conditions"]:
        shifts = [build(cond, i) for i in receivers]
        sc, sums, counts, names = score_with_token_patches(model, tokenizer, texts, shifts, class_schema="cert", **kw)
        vm = view_means(np.asarray(sums), np.asarray(counts), names)
        if cond == "zero":
            base_views = vm
        for j, i in enumerate(receivers):
            row = {"condition": cond, "receiver_idx": int(i), "receiver_id": meta.loc[i, "example_id"],
                   "user": meta.loc[i, "user_id"], "kind": kind[i], "pair": pair_of[i], "dept": dept_of(i)}
            for v in VIEWS:
                row[f"base_{v}"] = float(base_views[v][j]); row[f"patched_{v}"] = float(vm[v][j])
                row[f"delta_{v}"] = float(vm[v][j] - base_views[v][j])
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
        fh.flush(); del shifts
        print(f"[pilot4] {cond} done ({time.time() - t0:.0f}s)", flush=True)
    fh.close()
    print("PILOT4_DONE", flush=True)


if __name__ == "__main__":
    main()
