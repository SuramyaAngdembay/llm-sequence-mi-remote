#!/usr/bin/env python3
"""Pilot 1 (EXPLORATORY, 2026-09-25): intervention specificity and implementation.

Question (docs/HYPOTHESIS_LEDGER_2026-09-25.md, H1 and H2): do the selected SAE
directions change behaviour-token loss beyond what the actual perturbation size,
the edited positions and the SAE reconstruction explain, and is the package-4
effect a property of the model or of the patch implementation?

Everything is frozen before scoring: checkpoint, SAE, feature ids, receivers,
matched benign receivers, donor policy (mean prototype over fixed benign
same-department colleagues), target-token views, conditions, alpha = 1.

Conditions (per receiver day; shifts are added to the adapted model's
layer-26 hidden state, exactly as in eval_token_delta_sae_causal.py):
  zero       no edit (defines the base for every condition, same batches)
  recon_X    reconstruction only: dec(z) - delta at every token when any feature
             of X is active in the receiver (package-4 alpha 0), so that
             orig_X - rpU_X - recon_X isolates the reconstruction-edit interaction
  orig_X     the package-4 procedure: reconstruct every token, then set every
             coordinate of set X to the policy prototype at every position where
             ANY feature of X is active (shift = dec(z') - delta)
  rpU_X      residual-preserving, union support: shift = dec(z') - dec(z)
  rpO_X      residual-preserving, own support: a coordinate is edited only where
             it is itself active (no writing into zero coordinates)
  randI_X    isotropic random direction per token, norm and positions matched to rpU_X
  randD_X    random other-dictionary direction per token (activity >= 0.002,
             outside both sets), norm and positions matched to rpU_X
  single_f   rpO with only feature f of the selected set
X is the selected set (S) or the control set (C).

Writes, in order: pilot1_manifest.json, validity.json (alignment and zero-edit
checks), edit_balance.json (measured decoded edit norms and supports, before any
edited scoring), then pilot1_rows.csv (appended per condition).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

VIEWS = ("full", "profile_only", "behavior_only", "behavior_ses_only", "psy_only", "day_only")


# ------------------------------------------------------------------ pure helpers (unit-tested)
def edited_codes(Z: np.ndarray, feats: Sequence[int], proto: np.ndarray, mode: str) -> np.ndarray:
    """Return z' with the listed coordinates moved to the prototype (alpha = 1)."""
    Zp = np.array(Z, copy=True)
    feats = list(feats)
    if mode == "union":
        active = Z[:, feats].sum(axis=1) > 0
        for j, f in enumerate(feats):
            Zp[active, f] = proto[j]
    elif mode == "own":
        for j, f in enumerate(feats):
            rows = Z[:, f] > 0
            Zp[rows, f] = proto[j]
    else:
        raise ValueError(mode)
    return Zp


def shift_orig(Z, delta, feats, proto, decode: Callable) -> np.ndarray:
    """Package-4 shift: dec(z') - delta at every token if any listed feature is active."""
    if not (Z[:, list(feats)].sum(axis=1) > 0).any():
        return np.zeros_like(delta, dtype=np.float32)
    return (decode(edited_codes(Z, feats, proto, "union")) - delta).astype(np.float32)


def shift_recon(Z, delta, feats, decode: Callable) -> np.ndarray:
    """Package-4 alpha 0: dec(z) - delta at every token if any listed feature is active."""
    if not (Z[:, list(feats)].sum(axis=1) > 0).any():
        return np.zeros_like(delta, dtype=np.float32)
    return (decode(Z) - delta).astype(np.float32)


def load_delta_cache(path: Path, keep) -> tuple:
    """Rows of the compact cache written by pilot_cache_deltas.py, as float32, ordered by example."""
    d = np.load(path)
    idx = d["example_idx"]
    mask = np.isin(idx, np.fromiter(keep, dtype=np.int64))
    return d["delta"][mask].astype(np.float32), idx[mask]


def shift_residual(Z, feats, proto, decode: Callable, mode: str, d: int) -> np.ndarray:
    """h + dec(z') - dec(z): computed only on rows whose code changed (exact zero elsewhere)."""
    Zp = edited_codes(Z, feats, proto, mode)
    out = np.zeros((Z.shape[0], d), dtype=np.float32)
    rows = np.flatnonzero((Zp != Z).any(axis=1))
    if rows.size:
        out[rows] = decode(Zp[rows]) - decode(Z[rows])
    return out


def shift_random(ref: np.ndarray, rng: np.random.Generator, dirs: np.ndarray | None = None) -> np.ndarray:
    """Same per-token norms and positions as `ref`; direction isotropic or from `dirs` (unit rows)."""
    norms = np.linalg.norm(ref, axis=1)
    rows = np.flatnonzero(norms > 0)
    out = np.zeros_like(ref, dtype=np.float32)
    if rows.size == 0:
        return out
    if dirs is None:
        g = rng.standard_normal((rows.size, ref.shape[1])).astype(np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
    else:
        g = dirs[rng.integers(0, len(dirs), size=rows.size)]
    out[rows] = g * norms[rows, None]
    return out


def edit_stats(shift: np.ndarray, classes: Sequence[str]) -> Dict[str, float]:
    norms = np.linalg.norm(shift, axis=1)
    rows = norms > 0
    stats = {
        "n_tokens": int(len(norms)),
        "n_edited": int(rows.sum()),
        "norm_l2_total": float(np.sqrt((norms ** 2).sum())),
        "norm_mean_edited": float(norms[rows].mean()) if rows.any() else 0.0,
    }
    for c in ("DAY", "PSY", "SESCOUNT", "SES"):
        stats[f"edited_{c}"] = int(sum(1 for k, e in zip(classes, rows) if e and k == c))
    return stats


def view_means(sums: np.ndarray, counts: np.ndarray, names: Sequence[str]) -> Dict[str, np.ndarray]:
    import token_class_decomposition as tcd
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for v in VIEWS:
        cols = [idx[c] for c in tcd.CERT_VIEWS[v]]
        out[v] = sums[:, cols].sum(axis=1) / np.maximum(counts[:, cols].sum(axis=1), 1)
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--delta-cache", type=Path, default=None, help="npz from pilot_cache_deltas.py")
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--populations", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--latent-mult", type=int, default=2)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--selected", default="4596,7693,2302,3673,3455")
    ap.add_argument("--control", default="6596,8017,6608,2765,886")
    ap.add_argument("--exclude-users", default="JJM0203", help="users with no same-department colleague")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--loss-batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--dry-run", action="store_true", help="stop before loading the language model")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from remote_common import ensure_dir, load_yaml
    from sae_core import TopKSAE, load_ranking
    import token_class_decomposition as tcd
    from eval_token_delta_sae_causal import (
        build_example_slices, build_sparse_cache_for_examples, decode_sparse_tokens,
        donor_feature_prototype, load_eval_examples, load_token_layer_vectors, per_example_nll,
        score_with_token_patches,
    )

    t0 = time.time()
    out = ensure_dir(args.out_dir)
    cfg = load_yaml(args.config)
    S = [int(x) for x in args.selected.split(",")]
    C = [int(x) for x in args.control.split(",")]
    pops = json.loads(args.populations.read_text())
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    dept_of = lambda i: dict(kv.split("=", 1) for kv in meta.loc[i, "text"].split("\n", 1)[0][4:].split())["dept"]
    excluded = set(args.exclude_users.split(",")) if args.exclude_users else set()
    pairs = [(a, b) for a, b in pops["receiver_pairs"] if meta.loc[a, "user_id"] not in excluded]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    receivers = [i for pair in pairs for i in pair]            # (malicious, benign) adjacent: same batches
    kind = {a: "malicious" for a, _ in pairs} | {b: "benign" for _, b in pairs}
    pair_of = {a: n for n, (a, b) in enumerate(pairs)} | {b: n for n, (a, b) in enumerate(pairs)}
    for a, b in pairs:
        assert meta.loc[a, "user_id"] == meta.loc[b, "user_id"] and meta.loc[a, "y"] == 1 and meta.loc[b, "y"] == 0
    depts = sorted({dept_of(i) for i in receivers})
    donors = {d: [int(x) for x in pops["donor_policy_by_dept"][d]] for d in depts}
    if any(len(v) == 0 for v in donors.values()):
        raise SystemExit(f"a receiver department has no donor: {donors}")

    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    ranking = load_ranking(cfg_dir)
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    active = ranking.set_index("feature_id")["row_active_frac"]
    pool = [int(f) for f in active.index if active[f] >= 0.002 and f not in set(S) | set(C)]
    manifest = {
        "exploratory": True,
        "written_before_scoring": True,
        "adapter_dir": str(args.adapter_dir),
        "adapter_sha256": hashlib.sha256((args.adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "sae": str(cfg_dir / "delta_sae_model.pt"),
        "sae_sha256": hashlib.sha256((cfg_dir / "delta_sae_model.pt").read_bytes()).hexdigest(),
        "ranking_sha256": hashlib.sha256((cfg_dir / "delta_sae_top_features.csv").read_bytes()).hexdigest(),
        "populations_sha256": hashlib.sha256(args.populations.read_bytes()).hexdigest(),
        "layer": args.layer, "selected": S, "control": C, "alpha": 1.0,
        "excluded_users": sorted(excluded), "n_pairs": len(pairs), "n_users": len({meta.loc[a, 'user_id'] for a, _ in pairs}),
        "donor_policy": {d: {"donor_idx": v} for d, v in donors.items()},
        "random_dictionary_pool_size": len(pool),
        "views": list(VIEWS),
        # scored in this order, so that a timeout loses the least important conditions
        "conditions": ["zero", "rpU_S", "randI_S", "randD_S", "rpU_C", "randI_C", "randD_C",
                       "recon_S", "recon_C", "orig_S", "orig_C", "rpO_S", "rpO_C"] + [f"single_{f}" for f in S],
        "donor_prototype_rule": "per department, mean over the fixed donors of the package-4 prototype (mean over donor "
                                "tokens where any feature of the set is active); the same prototype for every support",
        "aggregation": "per receiver: mean patched-minus-zero loss over the view's target tokens; receivers whose edit is "
                       "zero contribute 0 (the policy edits only where features are active) and the share edited is "
                       "reported; per user: mean over receivers; estimate: mean of user means; 95% interval: user-"
                       "clustered percentile bootstrap, 10,000 and 5,000 draws",
        "endpoints_declared": {
            "E1": "malicious receivers, behavior_only delta: rpU_S minus randI_S (and randD_S), mean of user means, user-clustered bootstrap",
            "E2": "same for the control set: rpU_C minus randI_C (and randD_C)",
            "E3": "rpU_S minus rpU_C (selected versus control without reconstruction)",
            "E4": "orig_X minus rpU_X minus recon_X, X in {S, C} (reconstruction-edit interaction)",
            "E5": "rpU_S minus rpO_S (writing into zero coordinates)",
            "E6": "[rpU_S minus rpU_C](malicious) minus [same](matched benign), per user",
            "E7": "rpO_S minus the sum of single-feature effects (interaction)",
        },
        "seed": args.seed,
    }
    (out / "pilot1_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[pilot1] {len(pairs)} pairs, {manifest['n_users']} users, depts {depts}", flush=True)

    needed = set(receivers) | {i for v in donors.values() for i in v}
    if args.delta_cache is not None:
        x, token_idx = load_delta_cache(args.delta_cache, needed)
    else:
        x, token_idx, _ = load_token_layer_vectors(args.extract_dir, args.layer, keep_examples=needed, delta_dtype="float32")
    missing = needed - set(np.unique(token_idx).tolist())
    if missing:
        raise SystemExit(f"{len(missing)} needed examples have no delta rows, e.g. {sorted(missing)[:5]}")
    slices = build_example_slices(token_idx.astype(np.int64, copy=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    sae.load_state_dict(bundle["state_dict"]); sae.eval()
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32); x_std = np.asarray(bundle["x_std"], dtype=np.float32)
    Z = build_sparse_cache_for_examples(sae, x, slices, sorted(needed), x_mean=x_mean, x_std=x_std, device=device, batch_size=2048)
    dec = lambda codes: decode_sparse_tokens(sae, np.asarray(codes, dtype=np.float32), x_mean=x_mean, x_std=x_std,
                                             device=device, batch_size=2048)
    W = sae.decoder.weight.detach().cpu().numpy() * x_std.reshape(-1, 1)          # residual-unit decoder columns
    dirs = W[:, pool].T
    dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(np.float32)

    proto = {}
    for d, idxs in donors.items():
        proto[d] = {"S": np.mean([donor_feature_prototype(Z[i], S) for i in idxs], axis=0).astype(np.float32),
                    "C": np.mean([donor_feature_prototype(Z[i], C) for i in idxs], axis=0).astype(np.float32)}
    manifest["donor_prototypes"] = {d: {k: [float(v) for v in p[k]] for k in p} for d, p in proto.items()}

    # validity 1: tokenizer positions align with the cached delta rows
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "right":
        raise SystemExit(f"patching assumes right padding, tokenizer pads {tokenizer.padding_side}")
    max_len = int(cfg["training"]["max_seq_len"])
    classes = {}
    misaligned = []
    for i in receivers:
        text = meta.loc[i, "text"]
        enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
        cls, _ = tcd.classify_tokens([tuple(o) for o in enc["offset_mapping"]], tcd.cert_class_spans(text))
        classes[i] = cls
        n_rows = slices[i].stop - slices[i].start
        if n_rows != len(cls):
            misaligned.append((int(i), n_rows, len(cls)))
    validity = {"token_rows_equal_tokenizer_length": not misaligned, "misaligned": misaligned[:10],
                "n_receivers": len(receivers), "tokenizer_padding_side": tokenizer.padding_side}
    if misaligned:
        (out / "validity.json").write_text(json.dumps(validity, indent=1) + "\n")
        raise SystemExit(f"delta rows do not align with tokenization for {len(misaligned)} receivers")

    rng_for = lambda i, tag: np.random.default_rng([args.seed, int(i), int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)])

    def build(cond: str, i: int) -> np.ndarray:
        z, delta = Z[i], x[slices[i]]
        d = dept_of(i)
        if cond == "zero":
            return np.zeros_like(delta, dtype=np.float32)
        base, arm = cond.rsplit("_", 1)
        if base == "single":
            f = int(arm)
            return shift_residual(z, [f], proto[d]["S"][[S.index(f)]], dec, "own", delta.shape[1])
        feats, p = (S, proto[d]["S"]) if arm == "S" else (C, proto[d]["C"])
        if base == "orig":
            return shift_orig(z, delta, feats, p, dec)
        if base == "recon":
            return shift_recon(z, delta, feats, dec)
        if base == "rpU":
            return shift_residual(z, feats, p, dec, "union", delta.shape[1])
        if base == "rpO":
            return shift_residual(z, feats, p, dec, "own", delta.shape[1])
        ref = shift_residual(z, feats, p, dec, "union", delta.shape[1])
        if base == "randI":
            return shift_random(ref, rng_for(i, cond))
        if base == "randD":
            return shift_random(ref, rng_for(i, cond), dirs)
        raise ValueError(cond)

    # validity 2: orig = rpU + reconstruction error, on the first receivers
    worst = 0.0
    for i in receivers[:8]:
        z, delta = Z[i], x[slices[i]]
        if (z[:, S].sum(axis=1) > 0).any():
            recon = dec(z) - delta
            worst = max(worst, float(np.abs(build("orig_S", i) - (build("rpU_S", i) + recon)).max()))
    validity["orig_equals_residual_plus_reconstruction_max_abs"] = worst

    # balance: measured decoded edit sizes and supports, written before any edited scoring
    balance_rows = []
    for cond in manifest["conditions"][1:]:
        for i in receivers:
            st = edit_stats(build(cond, i), classes[i])
            balance_rows.append({"condition": cond, "receiver_idx": int(i), "kind": kind[i], **st})
    bal = pd.DataFrame(balance_rows)
    bal.to_csv(out / "edit_balance_rows.csv", index=False)
    summary = bal.assign(any_edit=bal["n_edited"] > 0).groupby(["condition", "kind"])[
        ["any_edit", "n_tokens", "n_edited", "norm_l2_total", "norm_mean_edited",
         "edited_DAY", "edited_PSY", "edited_SESCOUNT", "edited_SES"]].mean()
    (out / "edit_balance.json").write_text(summary.reset_index().to_json(orient="records", indent=1))

    # donor-minus-receiver coefficient changes per feature, own and union support (alpha 1)
    coef = []
    for arm, feats in (("S", S), ("C", C)):
        for j, f in enumerate(feats):
            own_z, own_dz, uni_dz, n_rec = [], [], [], 0
            for i in receivers:
                z = Z[i]
                p = proto[dept_of(i)][arm][j]
                own = z[:, f] > 0
                uni = z[:, feats].sum(axis=1) > 0
                own_z.extend(z[own, f].tolist()); own_dz.extend((p - z[own, f]).tolist())
                uni_dz.extend((p - z[uni, f]).tolist()); n_rec += bool(own.any())
            coef.append({"arm": arm, "feature": f, "receivers_active": n_rec,
                         "prototype_mean_over_depts": float(np.mean([proto[d][arm][j] for d in proto])),
                         "receiver_own_active_mean": float(np.mean(own_z)) if own_z else 0.0,
                         "dz_own_support_mean": float(np.mean(own_dz)) if own_dz else 0.0,
                         "dz_own_support_frac_negative": float(np.mean(np.asarray(own_dz) < 0)) if own_dz else 0.0,
                         "dz_union_support_mean": float(np.mean(uni_dz)) if uni_dz else 0.0,
                         "decoder_col_norm_residual_units": float(np.linalg.norm(W[:, f]))})
    (out / "coefficient_changes.json").write_text(json.dumps(coef, indent=1) + "\n")
    (out / "pilot1_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[pilot1] balance written ({time.time() - t0:.0f}s)", flush=True)
    if args.dry_run:
        (out / "validity.json").write_text(json.dumps(validity, indent=1) + "\n")
        print("[pilot1] dry run: stopping before the language model", flush=True)
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

    # validity 3: a true zero edit reproduces the unhooked model
    n_chk = min(24, len(texts))
    plain = []
    for s0 in range(0, n_chk, args.batch_size):
        tok = tokenizer(texts[s0:min(s0 + args.batch_size, n_chk)], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        tok = {k: v.to(model.device) for k, v in tok.items()}
        with torch.no_grad():
            o = model(**tok, return_dict=True)
        plain.append(per_example_nll(o.logits, tok["input_ids"], tok["attention_mask"],
                                     loss_batch_size=args.loss_batch_size).float().cpu().numpy())
    plain = np.concatenate(plain)
    zero_chk = score_with_token_patches(model, tokenizer, texts[:n_chk], [build("zero", i) for i in receivers[:n_chk]],
                                        class_schema="cert", **kw)[0]
    validity["zero_edit_vs_unhooked_max_abs"] = float(np.abs(np.asarray(zero_chk) - plain).max())
    (out / "validity.json").write_text(json.dumps(validity, indent=1) + "\n")
    print(f"[pilot1] validity {json.dumps(validity)}", flush=True)
    if validity["zero_edit_vs_unhooked_max_abs"] > 1e-5:
        raise SystemExit("zero edit does not reproduce the unhooked model")

    rows_path = out / "pilot1_rows.csv"
    writer, fh = None, rows_path.open("w", newline="")
    base_views = None
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
                row[f"base_{v}"] = float(base_views[v][j])
                row[f"patched_{v}"] = float(vm[v][j])
                row[f"delta_{v}"] = float(vm[v][j] - base_views[v][j])
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
        fh.flush()
        del shifts
        print(f"[pilot1] {cond} done ({time.time() - t0:.0f}s)", flush=True)
    fh.close()
    print("PILOT1_DONE", flush=True)


if __name__ == "__main__":
    main()
