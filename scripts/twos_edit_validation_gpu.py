#!/usr/bin/env python3
"""Phase 3 of the 2026-09-26 TWOS feasibility pilot (EXPLORATORY; small GPU validation, one 8 GB GPU).

Run only if the cached-activation diagnostics (Phase 2) justify it. Qwen2.5-3B,
existing NF4 configuration and corrected seed-42 adapter; SAE layer 24, m = 4,
k = 8; selected set fixed by the corrected discovery ranking.

Populations (metadata only, salt 'twos-phase3-2026-09-26', fixed before scoring):
  malicious receivers  up to --per-user positive windows per confirmation user, sha256 order
  matched benign       the same user's benign window nearest in (day, window time), earlier on a
                       tie, each used once; duplicates and exclusions recorded
  donors               per team, up to --donors-per-team benign windows of that team's
                       never-malicious users (distinct users first), sha256 order
  prototype            per team, mean over its donors of the package-4 prototype

Conditions (patched minus unedited, same batches):
  zero      no edit
  recon     reconstruction only, dec(z) - delta at every token when any selected feature fires
  rpU       residual-preserving union edit            x_std * D (z_union - z)
  rpO       residual-preserving own-support edit      x_std * D (z_own - z)
  randI_U / randI_O   isotropic random direction per edited token, norm-matched to rpU / rpO
  projO     minimum raw-movement edit that sets each targeted preactivation to its own-support
            requested value (the equality-only solution of the constrained problem; TopK-cell
            inequalities are not enforced, and no rescaling is applied after solving). Phase 2
            showed it realizes the same target changes as rpO with about 2.4x less movement
  randI_P / randR_P   random controls matched to projO
  randR_U / randR_O   Gram-preserving random rotation of the rpU / rpO edit: the edit matrix
                      U S V^T becomes U S V'^T with a random orthonormal frame V', so per-token
                      norms and all cross-token inner products (directional coherence) are kept
Diagnostic labelled "fresh recomputed" rebuilds the request from fresh deltas; it is NOT a check
of the applied edit. The applied-edit validation (added 2026-09-27) captures the patched hidden
state inside the scoring call (forward pre-hook on the next block, i.e. after the patch hook and
its bf16 cast), subtracts the adapter-off state from an identically batched run of the same code
path, re-encodes it, and compares it with the explicitly recorded intended code (built from the
cached codes, as the scored shifts were).
Common-budget conditions (added 2026-09-27):
  rpO_atP   decoder own-support edit rescaled per token to projO's raw norm (magnitude fixed at the
            projection's budget; the realized target change shrinks accordingly)
  projO_atD projection rescaled per token to rpO's raw norm (magnitude fixed at the decoder's
            budget; overshoots the target)
The common-target comparison (rpO versus projO, same requested code change) is kept separately.

  --smoke N   run only the first N receiver pairs with batch size 1, time and measure memory, stop
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
from pilot_intervention_specificity import VIEWS, edited_codes, shift_random, view_means  # noqa: E402
from sae_edit_geometry import min_norm_equality, topk_relu  # noqa: E402
from twos_edit_realizability import edit_metrics  # noqa: E402

SALT = "twos-phase3-2026-09-26"
H = lambda *parts: hashlib.sha256("|".join((SALT,) + tuple(str(p) for p in parts)).encode()).hexdigest()


def gram_preserving_rotation(E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random directions with the same Gram matrix as the rows of E (norms and cross-token cosines kept)."""
    rows = np.flatnonzero(np.linalg.norm(E, axis=1) > 0)
    out = np.zeros_like(E, dtype=np.float32)
    if rows.size == 0:
        return out
    U, s, Vt = np.linalg.svd(E[rows].astype(np.float64), full_matrices=False)
    Q, _ = np.linalg.qr(rng.standard_normal((E.shape[1], len(s))))
    out[rows] = (U * s) @ Q.T
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--latent-mult", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--selected", default="6036,7375,1197,7420,3218")
    ap.add_argument("--per-user", type=int, default=4)
    ap.add_argument("--donors-per-team", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    import pandas as pd
    import torch
    import transformers, peft, bitsandbytes
    from remote_common import load_yaml
    from eval_token_delta_sae_causal import (build_example_slices, donor_feature_prototype, get_layer_module,
                                             load_eval_examples, per_example_nll, score_with_token_patches)
    import token_class_decomposition as tcd

    t0 = time.time()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    S = [int(v) for v in args.selected.split(",")]
    cfg = load_yaml(args.config)
    conf = (args.split_dir / "twos_confirmation_users.txt").read_text().split()
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    user = meta["user_id"].astype(str).to_numpy(); y = meta["y"].astype(int).to_numpy(); team = meta["team"].astype(str).to_numpy()
    rows_json = [json.loads(l) for l in (args.data_dir / "eval.jsonl").open()]
    win_min = np.array([int(r["win"][:2]) * 60 + int(r["win"][3:5]) for r in rows_json])
    day = meta["day_index"].astype(int).to_numpy()
    pos_users = set(user[y == 1])

    # populations (metadata only)
    mal, ben, dup_log = [], [], []
    for u in sorted(conf):
        attacks = sorted(np.flatnonzero((user == u) & (y == 1)), key=lambda i: H("recv", meta.loc[i, "example_id"]))[: args.per_user]
        benign = list(np.flatnonzero((user == u) & (y == 0)))
        used = set()
        for a in attacks:
            cands = sorted((b for b in benign if b not in used), key=lambda b: (abs(day[b] - day[a]) * 1440 + abs(win_min[b] - win_min[a]), day[b] * 1440 + win_min[b] > day[a] * 1440 + win_min[a], b))
            if not cands:
                dup_log.append({"receiver": meta.loc[a, "example_id"], "reason": "no unused benign window"}); continue
            used.add(cands[0]); mal.append(int(a)); ben.append(int(cands[0]))
    donors = {}
    for tm in sorted({team[i] for i in mal}):
        pool = [j for j in np.flatnonzero((team == tm) & (y == 0)) if user[j] not in pos_users]
        by_user = defaultdict(list)
        for j in sorted(pool, key=lambda j: H("donor", meta.loc[j, "example_id"])):
            by_user[user[j]].append(int(j))
        picks, r = [], 0
        while len(picks) < args.donors_per_team and any(len(v) > r for v in by_user.values()):
            for v in by_user.values():
                if len(v) > r and len(picks) < args.donors_per_team:
                    picks.append(v[r])
            r += 1
        donors[tm] = picks
    pairs = list(zip(mal, ben))
    if args.smoke:
        pairs = pairs[: args.smoke]
    receivers = [i for p in pairs for i in p]
    kind = {a: "malicious" for a, _ in pairs} | {b: "benign" for _, b in pairs}
    manifest = {"exploratory": True, "written_before_scoring": True, "salt": SALT, "selected": S, "layer": args.layer, "k": args.k,
                "n_pairs": len(pairs), "users": sorted({user[i] for i in receivers}), "exclusions": dup_log,
                "receiver_ids": [[meta.loc[a, "example_id"], meta.loc[b, "example_id"]] for a, b in pairs],
                "donors_by_team": {tm: [meta.loc[j, "example_id"] for j in v] for tm, v in donors.items()},
                "conditions": ["zero", "recon", "rpU", "rpO", "projO", "randI_U", "randR_U", "randI_O", "randR_O", "randI_P", "randR_P",
                               "rpO_atP", "projO_atD"],
                "validated_conditions": ["rpU", "rpO", "projO", "rpO_atP", "projO_atD"],
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                             "bitsandbytes": bitsandbytes.__version__, "numpy": np.__version__},
                "smoke": bool(args.smoke)}
    (out / "phase3_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[phase3] {len(pairs)} pairs, users {manifest['users']}, donors {manifest['donors_by_team']}", flush=True)

    # SAE and cached deltas
    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    We = sd["encoder.weight"].double().numpy(); be = sd["encoder.bias"].double().numpy(); D = sd["decoder.weight"].double().numpy()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float64).reshape(-1); x_std = np.asarray(bundle["x_std"], dtype=np.float64).reshape(-1)
    obj = torch.load(args.extract_dir / f"layer_{args.layer}" / "chunk_00000.pt", map_location="cpu", weights_only=False, mmap=True)
    ex = np.asarray(obj["example_idx"], dtype=np.int64); sl = build_example_slices(ex)
    need = set(receivers) | {j for v in donors.values() for j in v}
    delta_c = {i: np.asarray(obj["delta"][sl[i]], dtype=np.float64) for i in need}
    del obj
    enc = lambda d: topk_relu(((d - x_mean) / x_std) @ We.T + be, args.k)
    proto = {tm: np.mean([donor_feature_prototype(enc(delta_c[j]), S) for j in v], axis=0) for tm, v in donors.items()}

    # model
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    q = cfg["quantization"]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "right":
        raise SystemExit("patching assumes right padding")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], quantization_config=BitsAndBytesConfig(
            load_in_4bit=bool(q["load_in_4bit"]), bnb_4bit_quant_type=str(q["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=getattr(torch, str(q["bnb_4bit_compute_dtype"])),
            bnb_4bit_use_double_quant=bool(q["bnb_4bit_use_double_quant"])),
        dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter_dir); model.config.use_cache = False; model.eval()
    load_s = time.time() - t_load
    max_len = int(cfg["training"]["max_seq_len"])
    texts = [meta.loc[i, "text"] for i in receivers]
    bs = 1 if args.smoke else args.batch_size

    # validity 1: tokenization length equals cached rows, no truncation
    classes = {}
    for i, text in zip(receivers, texts):
        enc_t = tokenizer(text, truncation=False, return_offsets_mapping=True)
        if len(enc_t["input_ids"]) != delta_c[i].shape[0] or len(enc_t["input_ids"]) > max_len:
            raise SystemExit(f"token length mismatch for {meta.loc[i, 'example_id']}")
        classes[i] = tcd.classify_tokens([tuple(o) for o in enc_t["offset_mapping"]], tcd.cert_class_spans(text))[0]

    # fresh deltas at layer 24 (adapter on minus off) and their codes
    layer_module = get_layer_module(model, args.layer)
    cap = {}
    hook = lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().float())
    fresh = {}
    t_f = time.time()
    with torch.inference_mode():
        for s0 in range(0, len(receivers), bs):
            batch = receivers[s0:s0 + bs]
            tok = tokenizer([meta.loc[i, "text"] for i in batch], return_tensors="pt", padding=True).to("cuda:0")
            hd = layer_module.register_forward_hook(hook)
            try:
                model(**tok); ha = cap["h"].clone()
                with model.disable_adapter():
                    model(**tok); hb = cap["h"].clone()
            finally:
                hd.remove()
            for b, i in enumerate(batch):
                n = int(tok["attention_mask"][b].sum())
                fresh[i] = (ha[b, :n] - hb[b, :n]).double().cpu().numpy()
    fresh_s = time.time() - t_f
    rel = np.concatenate([np.linalg.norm(fresh[i] - delta_c[i], axis=1) / np.maximum(np.linalg.norm(delta_c[i], axis=1), 1e-9) for i in receivers])
    zc = {i: enc(delta_c[i]) for i in receivers}; zf = {i: enc(fresh[i]) for i in receivers}
    jac = np.concatenate([((zc[i] > 0) & (zf[i] > 0)).sum(1) / np.maximum(((zc[i] > 0) | (zf[i] > 0)).sum(1), 1) for i in receivers])
    sel_agree = float(np.mean([np.array_equal(zc[i][:, S] > 0, zf[i][:, S] > 0) for i in receivers]))
    validity = {"token_lengths_equal_cache": True, "fresh_vs_cache_rel_err_median": float(np.median(rel)),
                "fresh_vs_cache_rel_err_p95": float(np.quantile(rel, 0.95)), "active_set_jaccard_median": float(np.median(jac)),
                "active_set_jaccard_mean": float(jac.mean()), "selected_support_identical_share_of_receivers": sel_agree,
                "dtype": "bfloat16 compute, NF4 weights, float64 SAE arithmetic"}

    # edits built from the CACHED deltas (the objects scored in the corrected run), diagnostics on both
    def shifts_for(i, rng_tag):
        d = delta_c[i]; z = zc[i]; p = proto[team[i]]
        any_s = (z[:, S] > 0).any()
        recon = ((z @ D.T) * x_std + x_mean - d) if any_s else np.zeros_like(d)
        out_s = {"zero": np.zeros_like(d), "recon": recon}
        for tag, rule in (("U", "union"), ("O", "own")):
            zr = edited_codes(z, S, p, rule)
            e = ((zr - z) @ D.T) * x_std
            out_s[f"rp{tag}"] = e
            rng = np.random.default_rng([args.seed, int(i), int(H(rng_tag, tag)[:8], 16)])
            out_s[f"randI_{tag}"] = shift_random(e.astype(np.float32), rng).astype(np.float64)
            out_s[f"randR_{tag}"] = gram_preserving_rotation(e, rng).astype(np.float64)
        out_s["projO"] = projection_shift(i)
        n_o = np.linalg.norm(out_s["rpO"], axis=1); n_p = np.linalg.norm(out_s["projO"], axis=1)
        both = (n_o > 0) & (n_p > 0)
        out_s["rpO_atP"] = np.zeros_like(out_s["rpO"]); out_s["rpO_atP"][both] = out_s["rpO"][both] * (n_p[both] / n_o[both])[:, None]
        out_s["projO_atD"] = np.zeros_like(out_s["projO"]); out_s["projO_atD"][both] = out_s["projO"][both] * (n_o[both] / n_p[both])[:, None]
        rng = np.random.default_rng([args.seed, int(i), int(H(rng_tag, "P")[:8], 16)])
        out_s["randI_P"] = shift_random(out_s["projO"].astype(np.float32), rng).astype(np.float64)
        out_s["randR_P"] = gram_preserving_rotation(out_s["projO"], rng).astype(np.float64)
        return out_s

    def projection_shift(i, src="cached"):
        d = delta_c[i] if src == "cached" else fresh[i]
        z = zc[i] if src == "cached" else zf[i]
        u = (d - x_mean) / x_std; pre = u @ We.T + be
        zr = edited_codes(z, S, proto[team[i]], "own")
        e = np.zeros_like(d)
        for t_row in np.flatnonzero((zr != z).any(axis=1)):
            T = [f for f in S if zr[t_row, f] != z[t_row, f]]
            v = min_norm_equality(We[T], np.array([zr[t_row, f] - pre[t_row, f] for f in T]), 1.0 / x_std ** 2)
            e[t_row] = v * x_std
        return e

    diag = defaultdict(list)
    for i in receivers:
        for src, dd, zz in (("cached", delta_c[i], zc[i]), ("fresh", fresh[i], zf[i])):
            u = (dd - x_mean) / x_std
            for rule in ("union", "own"):
                zr = edited_codes(zz, S, proto[team[i]], rule)
                rows = np.flatnonzero((zr != zz).any(axis=1))
                if rows.size == 0:
                    diag[f"{src}|{rule}|no_edit_receivers"].append(1); continue
                z_new = topk_relu((u[rows] + (zr[rows] - zz[rows]) @ D.T) @ We.T + be, args.k)
                m = edit_metrics(zz[rows], z_new, zr[rows], S, args.k, x_std, D, np.zeros(len(rows)), np.linalg.norm(((zr[rows] - zz[rows]) @ D.T) * x_std, axis=1))
                diag[f"{src}|{rule}|realized_frac"].extend(m["realized_frac"].tolist()); diag[f"{src}|{rule}|collateral_rel"].extend(m["collateral_rel"].tolist())
                diag[f"{src}|{rule}|over_k"].extend(m["over_k"].tolist())
        # the projection's realized code change, from the same source
        for src, dd, zz in (("cached", delta_c[i], zc[i]), ("fresh", fresh[i], zf[i])):
            zr = edited_codes(zz, S, proto[team[i]], "own")
            rows = np.flatnonzero((zr != zz).any(axis=1))
            if rows.size == 0:
                continue
            e = projection_shift(i, src)
            u = (dd - x_mean) / x_std
            z_new = topk_relu((u[rows] + e[rows] / x_std) @ We.T + be, args.k)
            m = edit_metrics(zz[rows], z_new, zr[rows], S, args.k, x_std, D, np.zeros(len(rows)), np.linalg.norm(e[rows], axis=1))
            diag[f"{src}|projection|realized_frac"].extend(m["realized_frac"].tolist()); diag[f"{src}|projection|collateral_rel"].extend(m["collateral_rel"].tolist())
    diagnostics = {("fresh recomputed request (not the applied edit)|" if k.startswith("fresh|") else "") + k:
                   (float(np.median(v)) if "no_edit" not in k else int(sum(v))) for k, v in diag.items()}

    # validity 2: a true zero edit reproduces the unhooked model
    kw = dict(layer=args.layer, max_seq_len=max_len, batch_size=bs, loss_batch_size=bs)
    plain = []
    with torch.inference_mode():
        for s0 in range(0, len(texts), bs):
            tok = tokenizer(texts[s0:s0 + bs], return_tensors="pt", padding=True).to("cuda:0")
            o = model(**tok, return_dict=True)
            plain.append(per_example_nll(o.logits, tok["input_ids"], tok["attention_mask"]).float().cpu().numpy())
    plain = np.concatenate(plain)
    zero = np.asarray(score_with_token_patches(model, tokenizer, texts, [np.zeros_like(delta_c[i], dtype=np.float32) for i in receivers], **kw))
    validity["zero_edit_vs_unhooked_max_abs"] = float(np.abs(zero - plain).max())
    (out / "phase3_validity.json").write_text(json.dumps(validity, indent=1) + "\n")
    print(f"[phase3] validity {json.dumps(validity)}", flush=True)
    if validity["zero_edit_vs_unhooked_max_abs"] > 1e-5:
        raise SystemExit("zero edit does not reproduce the unhooked model")

    # scoring
    torch.cuda.reset_peak_memory_stats(0)
    t_s = time.time()
    all_shifts = {i: shifts_for(i, "phase3") for i in receivers}
    rows_out, base = [], None
    next_block = get_layer_module(model, args.layer + 1)       # its input is the patched output of the patched block
    captured = {}

    def capture_into(store):
        def pre(_m, a, kw_):
            store.append((a[0] if a else kw_["hidden_states"]).detach().float().cpu())
        return next_block.register_forward_pre_hook(pre, with_kwargs=True)

    def per_receiver(store):
        out_c, j = {}, 0
        for chunk in store:
            for b in range(chunk.shape[0]):
                i = receivers[j]; out_c[i] = chunk[b, : delta_c[i].shape[0]].double().numpy(); j += 1
        return out_c

    # unpatched and adapter-off states in the same batch configuration, through the same scoring code path
    zeros = [np.zeros_like(delta_c[i], dtype=np.float32) for i in receivers]
    st = []; hd = capture_into(st)
    try:
        with torch.inference_mode():
            plain_unhooked = []
            for s0 in range(0, len(texts), bs):
                tk = tokenizer(texts[s0:s0 + bs], return_tensors="pt", padding=True).to("cuda:0")
                o = model(**tk, return_dict=True)
                plain_unhooked.append(per_example_nll(o.logits, tk["input_ids"], tk["attention_mask"]).float().cpu().numpy())
    finally:
        hd.remove()
    h_unhooked = per_receiver(st)
    st = []; hd = capture_into(st)
    try:
        with model.disable_adapter():
            score_with_token_patches(model, tokenizer, texts, zeros, class_schema="cert", **kw)
    finally:
        hd.remove()
    h_base_b = per_receiver(st)
    for cond in manifest["conditions"]:
        sh = [all_shifts[i][cond].astype(np.float32) for i in receivers]
        st = []
        hd = capture_into(st) if cond in manifest["validated_conditions"] or cond == "zero" else None
        try:
            sc, sums, counts, names = score_with_token_patches(model, tokenizer, texts, sh, class_schema="cert", **kw)
        finally:
            if hd is not None:
                hd.remove()
        if hd is not None:
            captured[cond] = per_receiver(st)
        vm = view_means(np.asarray(sums), np.asarray(counts), names)
        if cond == "zero":
            base = vm
        for j, i in enumerate(receivers):
            e = all_shifts[i][cond]
            row = {"condition": cond, "receiver_id": meta.loc[i, "example_id"], "user": user[i], "kind": kind[i], "team": team[i],
                   "edit_tokens": int((np.linalg.norm(e, axis=1) > 0).sum()), "edit_raw_l2": float(np.linalg.norm(e))}
            for v in VIEWS:
                row[f"base_{v}"] = float(base[v][j]); row[f"delta_{v}"] = float(vm[v][j] - base[v][j])
            rows_out.append(row)
        print(f"[phase3] {cond} done ({time.time() - t0:.0f}s)", flush=True)
    score_s = time.time() - t_s
    # ---- applied-edit validation (2026-09-27)
    val = defaultdict(list)
    plain_unhooked = np.concatenate(plain_unhooked)
    zero_rows = [r_ for r_ in rows_out if r_["condition"] == "zero"]
    val_summary = {"zero_vs_unhooked_hidden_max_abs": float(max(np.abs(captured["zero"][i] - h_unhooked[i]).max() for i in receivers)),
                   "zero_vs_unhooked_loss_max_abs": float(np.abs(np.asarray([r_["base_full"] for r_ in zero_rows]) - plain_unhooked).max()),
                   "batch_size": bs}
    Sarr = np.asarray(S)
    notS = np.setdiff1d(np.arange(We.shape[0]), Sarr)
    for cond in manifest["validated_conditions"]:
        rule = "union" if cond == "rpU" else "own"
        for i in receivers:
            z_req = edited_codes(zc[i], S, proto[team[i]], rule)          # the intended code, from the cached codes
            rows = np.flatnonzero((z_req != zc[i]).any(axis=1))
            if rows.size == 0:
                val[f"{cond}|no_op_receivers"].append(1); continue
            hp, hu, hb = captured[cond][i][rows], captured["zero"][i][rows], h_base_b[i][rows]
            if not (np.isfinite(hp).all() and np.isfinite(hb).all()):
                val[f"{cond}|numerical_failures"].append(1); continue
            z_real = enc(hp - hb); z_fresh = enc(hu - hb)
            req = (z_req - zc[i])[rows][:, Sarr]
            real_vs_target = (z_real - z_req[rows])[:, Sarr]
            real_change = (z_real - z_fresh)[:, Sarr]
            rn = np.linalg.norm(req, axis=1)
            val[f"{cond}|coord_abs_target_err"].extend(np.abs(real_vs_target)[req != 0].tolist())
            val[f"{cond}|norm_target_err"].extend((np.linalg.norm(real_vs_target, axis=1) / np.maximum(rn, 1e-12)).tolist())
            val[f"{cond}|realized_change_over_requested_projection"].extend((np.sum(real_change * req, axis=1) / np.maximum(rn ** 2, 1e-24)).tolist())
            off = (z_real - z_fresh)[:, notS]
            val[f"{cond}|offtarget_l2_vs_fresh_unpatched"].extend(np.linalg.norm(off, axis=1).tolist())
            val[f"{cond}|offtarget_changed_coords_gt_1e-4"].extend((np.abs(off) > 1e-4).sum(axis=1).tolist())
            a_, b_ = z_real > 0, z_req[rows] > 0
            val[f"{cond}|support_jaccard_realized_vs_intended(den=union)"].extend(((a_ & b_).sum(1) / np.maximum((a_ | b_).sum(1), 1)).tolist())
            val[f"{cond}|share_intended_active_absent(den=intended support)"].extend(((b_ & ~a_).sum(1) / np.maximum(b_.sum(1), 1)).tolist())
            val[f"{cond}|intended_raw_norm"].extend(np.linalg.norm(all_shifts[i][cond][rows], axis=1).tolist())
            val[f"{cond}|applied_raw_norm_over_intended"].extend((np.linalg.norm(hp - hu, axis=1) / np.maximum(np.linalg.norm(all_shifts[i][cond][rows], axis=1), 1e-12)).tolist())
            if cond == "rpO":
                f0, fc = zc[i][rows] > 0, z_fresh > 0
                val["baseline|support_jaccard_fresh_unpatched_vs_cached(den=union)"].extend(((f0 & fc).sum(1) / np.maximum((f0 | fc).sum(1), 1)).tolist())
                val["baseline|features_replaced_per_token(cached support absent in fresh)"].extend((f0 & ~fc).sum(1).tolist())
    q = lambda v: {"n": len(v), "q10": float(np.quantile(v, 0.1)), "median": float(np.median(v)), "q90": float(np.quantile(v, 0.9)),
                   "q99": float(np.quantile(v, 0.99)), "max": float(np.max(v)), "mean": float(np.mean(v))}
    val_summary.update({k: (q(v) if not k.endswith(("receivers", "failures")) else int(sum(v))) for k, v in sorted(val.items())})
    (out / "phase3_applied_validation.json").write_text(json.dumps(val_summary, indent=1) + "\n")
    print(f"[phase3] applied validation: zero/unhooked {val_summary['zero_vs_unhooked_hidden_max_abs']:.2e} hidden, "
          f"{val_summary['zero_vs_unhooked_loss_max_abs']:.2e} loss", flush=True)
    with (out / "phase3_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    timing = {"model_load_s": round(load_s, 1), "fresh_delta_s": round(fresh_s, 1), "scoring_s": round(score_s, 1),
              "forward_passes_scored": len(receivers) * len(manifest["conditions"]), "batch_size": bs,
              "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2), "total_s": round(time.time() - t0, 1),
              "gpu": torch.cuda.get_device_name(0)}
    (out / "phase3_timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    (out / "phase3_diagnostics.json").write_text(json.dumps(diagnostics, indent=1) + "\n")
    print(f"[phase3] timing {json.dumps(timing)}", flush=True)
    print("PHASE3_DONE", flush=True)


if __name__ == "__main__":
    main()
