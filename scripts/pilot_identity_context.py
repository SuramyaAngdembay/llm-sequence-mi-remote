#!/usr/bin/env python3
"""Pilot 2 (EXPLORATORY, 2026-09-25): identity context in behaviour prediction.

Question (docs/HYPOTHESIS_LEDGER_2026-09-25.md, H3; data for H4): does
fine-tuning make behaviour-token predictions depend on the profile (DAY/PSY
lines) more than the base model's do, and do the selected SAE features read
the profile when they fire on session tokens?

Inputs per receiver day (frozen in the populations manifest): the original
text; the same text with its DAY and PSY lines replaced by a same-department
never-malicious eval user's (keeping this day's `week=` value and every session
line); the same with a different-department eval user's lines; and (added
2026-09-25, before scoring) the same with a same-department TRAINING user's
lines. Every eval user is absent from the adapter's training split, so the
training-user swap is the only familiar-identity contrast. Causal order is
respected: the swap changes only context that precedes every behaviour token.
The swap is an input contrast for measuring what the predictions depend on; a
swapped record is not assumed to keep its attack label.

Per input and per model (adapter on / off): per-token target NLL and the
behaviour/profile view means. At layer 26: delta = h_adapted - h_base, encoded
by the frozen SAE; selected and control feature activations on SES tokens.
For the original inputs, per-token arrays are saved for Pilot 3.

Validity (written before any analysis): the on-the-fly delta is compared with
the cached delta for the first receivers; a hook or environment mismatch
would show as a large relative error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def swap_profile(receiver_text: str, partner_text: str) -> str:
    """Replace the receiver's DAY and PSY lines with the partner's, keeping the receiver's week."""
    r = receiver_text.split("\n")
    p = partner_text.split("\n") if isinstance(partner_text, str) else list(partner_text)
    if not (r[0].startswith("DAY ") and r[1].startswith("PSY ") and p[0].startswith("DAY ") and p[1].startswith("PSY ")):
        raise ValueError("unexpected profile layout")
    week = re.search(r"\bweek=(\S+)", r[0])
    day = re.sub(r"\bweek=\S+", f"week={week.group(1)}", p[0]) if week else p[0]
    return "\n".join([day, p[1]] + r[2:])


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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-check", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    import torch
    import torch.nn.functional as F
    from remote_common import ensure_dir, load_yaml
    from sae_core import TopKSAE
    import token_class_decomposition as tcd
    from eval_token_delta_sae_causal import (build_example_slices, get_layer_module, load_eval_examples,
                                             load_token_layer_vectors)

    t0 = time.time()
    out = ensure_dir(args.out_dir)
    cfg = load_yaml(args.config)
    S = [int(v) for v in args.selected.split(",")]
    C = [int(v) for v in args.control.split(",")]
    pops = json.loads(args.populations.read_text())
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    pairs = [tuple(p) for p in pops["receiver_pairs"]]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    receivers = [i for p in pairs for i in p]
    kind = {a: "malicious" for a, _ in pairs} | {b: "benign" for _, b in pairs}
    partners = pops["swap_partners_by_user"]
    train_partners = pops.get("train_partners_by_user", {})
    if not train_partners:
        raise SystemExit("populations manifest has no train_partners_by_user (run the selector with --train-jsonl)")

    items = []                        # (receiver_idx, variant, text)
    for i in receivers:
        text = meta.loc[i, "text"]
        u = meta.loc[i, "user_id"]
        items.append((i, "orig", text))
        swaps = [("swap_same", None if partners[u]["same_dept_day"] is None
                  else meta.loc[int(partners[u]["same_dept_day"]), "text"]),
                 ("swap_other", meta.loc[int(partners[u]["other_dept_day"]), "text"]),
                 ("swap_train", None if not train_partners.get(u) else train_partners[u]["profile_lines"])]
        for variant, partner in swaps:
            if partner is None:
                continue
            new = swap_profile(text, partner)
            assert new.split("\n")[2:] == text.split("\n")[2:], "session lines changed"
            items.append((i, variant, new))
    manifest = {
        "exploratory": True, "written_before_scoring": True, "layer": args.layer, "selected": S, "control": C,
        "n_pairs": len(pairs), "n_items": len(items), "variants": ["orig", "swap_same", "swap_other", "swap_train"],
        "populations_sha256": __import__("hashlib").sha256(args.populations.read_bytes()).hexdigest(),
        "models": ["adapted", "base (adapter disabled)"],
        "endpoints_declared": {
            "F1": "behavior_only loss change under swap_other (and swap_same), adapted minus base, per user; malicious and benign reported separately",
            "F2": "adapted swap sensitivity on malicious minus matched benign days, per user",
            "F3": "mean selected-feature activation change on SES tokens under swap, minus the same for control features",
            "F4": "identity familiarity: [swap_train minus swap_same] behavior_only loss, adapted minus base, per user "
                  "(both partners share the receiver's department; only the training user is familiar to the adapter)",
            "aggregation": "per receiver: mean loss over the view's target tokens; per user: mean over receivers; "
                           "estimate: mean of user means; 95% interval: user-clustered percentile bootstrap",
            "H4 data": "orig inputs: per-token adapted and base target NLL and S/C activations, saved for Pilot 3",
        },
    }
    (out / "pilot2_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[pilot2] {len(pairs)} pairs, {len(items)} inputs", flush=True)
    if args.dry_run:
        print("[pilot2] dry run: inputs built, stopping before the language model", flush=True)
        return

    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    sae = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    sae.load_state_dict(bundle["state_dict"]); sae.eval()
    x_mean = torch.from_numpy(np.asarray(bundle["x_mean"], dtype=np.float32)).to(device)
    x_std = torch.from_numpy(np.asarray(bundle["x_std"], dtype=np.float32)).to(device)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    q = cfg["quantization"]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "right":
        raise SystemExit(f"per-token alignment assumes right padding, tokenizer pads {tokenizer.padding_side}")
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
    layer_module = get_layer_module(model, args.layer)
    max_len = int(cfg["training"]["max_seq_len"])

    captured: Dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    @torch.no_grad()
    def run(texts: Sequence[str], adapted: bool):
        tok = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                        return_offsets_mapping=True)
        offsets = tok.pop("offset_mapping")
        tok = {k: v.to(device) for k, v in tok.items()}
        handle = layer_module.register_forward_hook(hook)
        try:
            if adapted:
                o = model(**tok, return_dict=True)
            else:
                with model.disable_adapter():
                    o = model(**tok, return_dict=True)
        finally:
            handle.remove()
        res = []
        for b in range(len(texts)):
            n = int(tok["attention_mask"][b].sum())
            logp = F.log_softmax(o.logits[b, : n - 1].float(), dim=-1)
            nll = -logp.gather(1, tok["input_ids"][b, 1:n].unsqueeze(1)).squeeze(1)
            res.append({"nll": nll.cpu().numpy(), "h": captured["h"][b, :n].float(), "offsets": offsets[b, :n].tolist(),
                        "ids": tok["input_ids"][b, :n].cpu().numpy().astype(np.int32)})
        del o
        return res

    # validity: on-the-fly delta versus the cached delta
    chk = receivers[: args.n_check]
    if args.delta_cache is not None:
        from pilot_intervention_specificity import load_delta_cache
        x_cache, idx_cache = load_delta_cache(args.delta_cache, set(chk))
    else:
        x_cache, idx_cache, _ = load_token_layer_vectors(args.extract_dir, args.layer, keep_examples=set(chk),
                                                         delta_dtype="float32")
    sl = build_example_slices(idx_cache.astype(np.int64, copy=False))
    rel, jac, shape_mismatch = [], [], 0
    for s0 in range(0, len(chk), args.batch_size):
        batch = chk[s0 : s0 + args.batch_size]
        texts = [meta.loc[i, "text"] for i in batch]
        ra, rb = run(texts, True), run(texts, False)
        for i, a, b in zip(batch, ra, rb):
            d_new = a["h"] - b["h"]
            d_old = torch.from_numpy(x_cache[sl[i]]).to(device)
            if d_new.shape != d_old.shape:
                shape_mismatch += 1; continue
            rel.extend((torch.linalg.norm(d_new - d_old, dim=1) / torch.linalg.norm(d_old, dim=1).clamp_min(1e-6)).tolist())
            z_new = sae.encode_sparse((d_new - x_mean) / x_std) > 0
            z_old = sae.encode_sparse((d_old - x_mean) / x_std) > 0
            jac.extend(((z_new & z_old).sum(1) / (z_new | z_old).sum(1).clamp_min(1)).tolist())
    rel, jac = np.asarray(rel), np.asarray(jac)
    validity = {"n_examples_checked": len(chk), "shape_mismatches": shape_mismatch, "n_tokens_checked": int(len(rel)),
                "delta_rel_error_median": float(np.median(rel)), "delta_rel_error_p95": float(np.quantile(rel, 0.95)),
                "active_set_jaccard_median": float(np.median(jac)), "active_set_jaccard_mean": float(jac.mean()),
                "abort_rule": "shape mismatch, or median relative error > 0.5 (a wrong layer or misalignment, not bf16 noise)"}
    (out / "validity.json").write_text(json.dumps(validity, indent=1) + "\n")
    print(f"[pilot2] validity {json.dumps(validity)}", flush=True)
    if shape_mismatch or not np.isfinite(validity["delta_rel_error_median"]) or validity["delta_rel_error_median"] > 0.5:
        raise SystemExit("on-the-fly delta does not match the cached delta")

    feats = S + C
    records, per_token = [], {}
    for s0 in range(0, len(items), args.batch_size):
        batch = items[s0 : s0 + args.batch_size]
        texts = [t for _, _, t in batch]
        ra, rb = run(texts, True), run(texts, False)
        for (i, variant, text), a, b in zip(batch, ra, rb):
            cls, _ = tcd.classify_tokens([tuple(o) for o in a["offsets"]], tcd.cert_class_spans(text))
            tgt = np.asarray(cls[1:])                      # class of each predicted token
            delta = (a["h"] - b["h"])
            with torch.no_grad():
                z = sae.encode_sparse((delta - x_mean) / x_std)[:, feats].detach().cpu().numpy()
            ses = np.asarray(cls) == "SES"
            rec = {"receiver_idx": int(i), "receiver_id": meta.loc[i, "example_id"], "user": meta.loc[i, "user_id"],
                   "kind": kind[i], "variant": variant, "n_tokens": int(len(cls)), "n_ses_tokens": int(ses.sum())}
            for m, r in (("adapted", a), ("base", b)):
                for v in ("full", "profile_only", "behavior_only", "behavior_ses_only"):
                    mask = np.isin(tgt, tcd.CERT_VIEWS[v])
                    rec[f"{m}_{v}"] = float(r["nll"][mask].mean()) if mask.any() else float("nan")
            for j, f in enumerate(feats):
                col = z[ses, j] if ses.any() else np.zeros(1)
                rec[f"mean_act_{f}"] = float(col.mean())
                rec[f"frac_active_{f}"] = float((col > 0).mean())
            records.append(rec)
            if variant == "orig":
                per_token[str(i)] = {"classes": np.asarray(cls), "act": z.astype(np.float32), "ids": a["ids"],
                                     "nll_adapted": a["nll"].astype(np.float32), "nll_base": b["nll"].astype(np.float32)}
        if (s0 // args.batch_size) % 20 == 0:
            print(f"[pilot2] {s0 + len(batch)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(records).to_csv(out / "pilot2_rows.csv", index=False)
    np.savez_compressed(out / "pilot2_orig_tokens.npz", feats=np.asarray(feats), receivers=np.asarray(list(per_token)),
                        **{f"{k}__{name}": v for k, d in per_token.items() for name, v in d.items()})
    print("PILOT2_DONE", flush=True)


if __name__ == "__main__":
    main()
