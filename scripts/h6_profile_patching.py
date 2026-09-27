#!/usr/bin/env python3
"""H6 pilot (EXPLORATORY, 2026-09-27): where does profile-dependent behaviour prediction arise?

Protocol: docs/H6_PROFILE_PATCHING_PROTOCOL_2026-09-27.md (frozen before scoring).
Qwen3-8B r4.2 adapter; discovery-user receivers; profile substitutions R (both
lines), D (DAY only), P (PSY only), T (second partner, control source).
Activation patching (residual stream) at aligned profile or session positions,
component patching (attention / MLP output) at session positions, and
selected-feature mediation at the SAE layer with realized-code verification.

  python scripts/h6_profile_patching.py --config C --data-dir D --adapter-dir A --extract-dir X \\
      --frontier-dir F --split-dir S --out-dir O [--smoke-pairs 2] [--wall-budget-s 1500]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

SALT = "h6-2026-09-27"
H = lambda *p: hashlib.sha256("|".join((SALT,) + tuple(str(x) for x in p)).encode()).hexdigest()
LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34)
SAE_LAYER = 26


# ------------------------------------------------------------------ pure helpers (unit-tested)
def char_parts(text: str):
    """Per character: (line, field, part, index-within-part) with part in {'k', 'v', 'nl'}."""
    out = [None] * len(text)
    pos = 0
    for li, line in enumerate(text.split("\n")):
        f, part, start = 0, "k", pos
        for ci, ch in enumerate(line):
            if ch == " ":
                f += 1; part = "k"
            out[pos + ci] = [li, f, part]
            if ch == "=" and part == "k":
                part = "v"                       # characters after '=' belong to the value
        pos += len(line)
        if pos < len(text):
            out[pos] = [li, -1, "nl"]; pos += 1
    return out


def token_keys(text: str, offsets):
    """Alignment key per token: (line, field, part, rank from the end of its part); None for empty spans."""
    cp = char_parts(text)
    base = [tuple(cp[a]) if b > a and a < len(cp) and cp[a] is not None else None for a, b in offsets]
    groups = defaultdict(list)
    for i, k in enumerate(base):
        if k is not None:
            groups[k].append(i)
    keys = [None] * len(offsets)
    for k, idx in groups.items():
        for r, i in enumerate(reversed(idx)):
            keys[i] = k + (r,)
    return keys


def align(keys_src, keys_dst):
    """Map of aligned (src position, dst position) pairs with equal keys."""
    where = {k: i for i, k in enumerate(keys_src) if k is not None}
    return [(where[k], j) for j, k in enumerate(keys_dst) if k is not None and k in where]


def swap_lines(text: str, day_line=None, psy_line=None) -> str:
    lines = text.split("\n")
    wk = re.search(r"\bweek=(\S+)", lines[0])
    if day_line is not None:
        lines[0] = re.sub(r"\bweek=\S+", f"week={wk.group(1)}", day_line) if wk else day_line
    if psy_line is not None:
        lines[1] = psy_line
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-user", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--smoke-pairs", type=int, default=2)
    ap.add_argument("--wall-budget-s", type=float, default=1500.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20260927)
    # defaults are the CERT r4.2 settings of the protocol; overridable only to test the code path on another dataset
    ap.add_argument("--sae-layer", type=int, default=26)
    ap.add_argument("--sae-subdir", default="layer_26/m02_k04")
    ap.add_argument("--selected", default="4596,7693,2302,3673,3455")
    ap.add_argument("--control", default="6596,8017,6608,2765,886")
    ap.add_argument("--context-field", default="dept")
    ap.add_argument("--discovery-file", default="discovery_users.txt")
    ap.add_argument("--layers", default="2,6,10,14,18,22,26,30,34")
    args = ap.parse_args()
    global LAYERS, SAE_LAYER
    LAYERS = tuple(int(v) for v in args.layers.split(",")); SAE_LAYER = args.sae_layer
    if SAE_LAYER not in LAYERS:
        raise SystemExit("the SAE layer must be one of the patched layers")

    import pandas as pd
    from remote_common import load_yaml
    import token_class_decomposition as tcd
    from eval_token_delta_sae_causal import get_layer_module, load_eval_examples

    t0 = time.time()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    S = [int(v) for v in args.selected.split(",")]; C = [int(v) for v in args.control.split(",")]
    disc = (args.split_dir / args.discovery_file).read_text().split()
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    user = meta["user_id"].astype(str).to_numpy(); y = meta["y"].astype(int).to_numpy(); day = meta["day_index"].astype(int).to_numpy()
    dept = meta["text"].str.extract(args.context_field + r"=(\S+)")[0].to_numpy()
    mal_users = set(user[y == 1])
    benign_users = sorted(set(user) - mal_users)

    # populations (metadata only)
    pairs, excl = [], []
    for u in sorted(disc):
        att = sorted(np.flatnonzero((user == u) & (y == 1)), key=lambda i: H("recv", meta.loc[i, "example_id"]))[: args.per_user]
        ben = list(np.flatnonzero((user == u) & (y == 0))); used = set()
        for a in att:
            cands = sorted((b for b in ben if b not in used), key=lambda b: (abs(day[b] - day[a]), day[b] > day[a], b))
            if not cands:
                excl.append({"receiver": meta.loc[a, "example_id"], "reason": "no benign day"}); continue
            used.add(cands[0]); pairs.append((int(a), int(cands[0])))
    partner_day = {v: sorted(np.flatnonzero(user == v), key=lambda i: H("pday", meta.loc[i, "example_id"]))[0] for v in benign_users}

    def partners(i):
        others = sorted((v for v in benign_users if dept[partner_day[v]] != dept[i]), key=lambda v: H("partner", user[i], v))
        return others[0], others[1]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.padding_side != "right":
        raise SystemExit("patching assumes right padding")

    def encode(text):
        e = tok(text, return_offsets_mapping=True)
        cls, _ = tcd.classify_tokens([tuple(o) for o in e["offset_mapping"]], tcd.cert_class_spans(text))
        return {"ids": e["input_ids"], "cls": np.asarray(cls), "keys": token_keys(text, e["offset_mapping"])}

    recs, bad = [], []
    for a, b in pairs:
        for i, kind in ((a, "malicious"), (b, "benign")):
            pR, pT = partners(i)
            txt = meta.loc[i, "text"]
            dR = meta.loc[partner_day[pR], "text"].split("\n"); dT = meta.loc[partner_day[pT], "text"].split("\n")
            v = {"O": txt, "R": swap_lines(txt, dR[0], dR[1]), "D": swap_lines(txt, dR[0], None),
                 "P": swap_lines(txt, None, dR[1]), "T": swap_lines(txt, dT[0], dT[1])}
            enc = {k: encode(t) for k, t in v.items()}
            maps = {k: align(enc["O"]["keys"], enc[k]["keys"]) for k in v}             # O -> variant
            maps["T->R"] = align(enc["T"]["keys"], enc["R"]["keys"])
            ok = True
            for k in ("R", "D", "P", "T"):
                sess_o = [p for p in range(len(enc["O"]["cls"])) if enc["O"]["cls"][p] in ("SES", "SESCOUNT")]
                sess_v = [p for p in range(len(enc[k]["cls"])) if enc[k]["cls"][p] in ("SES", "SESCOUNT")]
                m = dict(maps[k])
                if len(sess_o) != len(sess_v) or any(m.get(p) is None for p in sess_o) or \
                        [enc["O"]["ids"][p] for p in sess_o] != [enc[k]["ids"][m[p]] for p in sess_o]:
                    ok = False
            if not ok:
                bad.append(meta.loc[i, "example_id"]); continue
            recs.append({"idx": i, "id": meta.loc[i, "example_id"], "user": user[i], "kind": kind, "pair": len(recs) // 2,
                         "partner_R": pR, "partner_T": pT, "texts": v, "enc": enc, "maps": maps})
    n_prof = [sum(1 for c in r["enc"]["R"]["cls"] if c in ("DAY", "PSY")) for r in recs]
    n_prof_al = [sum(1 for s, d in r["maps"]["R"] if r["enc"]["R"]["cls"][d] in ("DAY", "PSY")) for r in recs]
    manifest = {"exploratory": True, "written_before_scoring": True, "protocol": "docs/H6_PROFILE_PATCHING_PROTOCOL_2026-09-27.md",
                "salt": SALT, "layers": LAYERS, "sae_layer": SAE_LAYER, "selected": S, "control": C,
                "n_receivers": len(recs), "n_users": len({r["user"] for r in recs}), "exclusions": excl,
                "session_alignment_failures": bad, "share_alignment_failures": len(bad) / max(1, len(bad) + len(recs)),
                "profile_token_alignment_coverage_median": float(np.median(np.asarray(n_prof_al) / np.maximum(n_prof, 1))),
                "receivers": [{"id": r["id"], "user": r["user"], "kind": r["kind"], "partner_R": r["partner_R"], "partner_T": r["partner_T"]} for r in recs]}
    (out / "h6_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[h6] {len(recs)} receivers, {manifest['n_users']} users, alignment failures {len(bad)}, "
          f"profile alignment coverage median {manifest['profile_token_alignment_coverage_median']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    if manifest["share_alignment_failures"] > 0.2:
        raise SystemExit("more than 20% of receivers fail session alignment")
    if args.dry_run:
        return

    # ---------------------------------------------------------------- model and SAE
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    q = cfg["quantization"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], quantization_config=BitsAndBytesConfig(
        load_in_4bit=bool(q["load_in_4bit"]), bnb_4bit_quant_type=str(q["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(torch, str(q["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(q["bnb_4bit_use_double_quant"])), dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter_dir); model.config.use_cache = False; model.eval()
    dev = torch.device("cuda:0")
    blocks = {l: get_layer_module(model, l) for l in LAYERS}
    cfg_dir = args.frontier_dir / args.sae_subdir
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    We = sd["encoder.weight"].float().to(dev); be = sd["encoder.bias"].float().to(dev); Dd = sd["decoder.weight"].float().to(dev)
    xm = torch.tensor(np.asarray(bundle["x_mean"], dtype=np.float32).reshape(-1), device=dev)
    xs = torch.tensor(np.asarray(bundle["x_std"], dtype=np.float32).reshape(-1), device=dev)
    kk = int(bundle["k"])

    def sae_code(delta):                     # delta (n, d) float32 -> TopK code (n, L)
        pre = torch.relu(((delta - xm) / xs) @ We.T + be)
        v, i = torch.topk(pre, kk, dim=-1)
        return torch.zeros_like(pre).scatter_(1, i, v)

    beh_classes = set(tcd.CERT_VIEWS["behavior_only"]); ses_classes = set(tcd.CERT_VIEWS["behavior_ses_only"])

    def losses(logits, ids, lens, cls_list):
        out_b, out_s = [], []
        for b in range(len(lens)):
            n = lens[b]
            lp = F.log_softmax(logits[b, : n - 1].float(), dim=-1)
            nll = -lp.gather(1, ids[b, 1:n].unsqueeze(1)).squeeze(1).cpu().numpy()
            tgt = cls_list[b][1:n]
            mb = np.isin(tgt, list(beh_classes)); ms = np.isin(tgt, list(ses_classes))
            out_b.append(float(nll[mb].mean())); out_s.append(float(nll[ms].mean()))
        return out_b, out_s

    def run(texts, cls_list, adapter=True, patches=(), captures=()):
        """patches: list of (module, fn(output)->output); captures: list of (name, module, kind)."""
        b = tok(texts, return_tensors="pt", padding=True).to(dev)
        lens = b["attention_mask"].sum(1).tolist()
        cap, handles = {}, []
        for mod, fn in patches:
            handles.append(mod.register_forward_hook(lambda m, i, o, fn=fn: fn(o)))
        for name, mod in captures:
            def h(m, i, o, name=name):
                cap[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()
            handles.append(mod.register_forward_hook(h))
        try:
            with torch.inference_mode():
                if adapter:
                    lo = model(**b).logits
                else:
                    with model.disable_adapter():
                        lo = model(**b).logits
                lb, ls = losses(lo, b["input_ids"], lens, cls_list)
        finally:
            for hd in handles:
                hd.remove()
        return lb, ls, cap

    def put(o, b_idx_dst, src, b_idx_src_pos, add=False):
        """Replace (or add to) rows of a module output at given positions, preserving tuple outputs."""
        hs = o[0] if isinstance(o, tuple) else o
        for b, (dst, vals) in enumerate(zip(b_idx_dst, src)):
            if len(dst):
                idx = torch.as_tensor(dst, device=hs.device)
                v = vals.to(hs.dtype)
                hs[b, idx] = (hs[b, idx] + v) if add else v
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    def positions(r, variant, group):
        cls = r["enc"][variant]["cls"]
        want = ("DAY", "PSY") if group == "prof" else ("SES", "SESCOUNT")
        return [p for p in range(len(cls)) if cls[p] in want]

    def pair_idx(r, src_variant, dst_variant, group):
        m = r["maps"][dst_variant] if src_variant == "O" else r["maps"]["T->R"] if src_variant == "T" else [(p, p) for p in range(len(r["enc"][dst_variant]["ids"]))]
        dstset = set(positions(r, dst_variant, group))
        pr = [(s, d) for s, d in m if d in dstset]
        return [s for s, _ in pr], [d for _, d in pr]

    rows, verif, token_codes = [], defaultdict(list), {}

    def do_batch(batch, tag):
        texts = {v: [r["texts"][v] for r in batch] for v in "ORDPT"}
        cls = {v: [r["enc"][v]["cls"] for r in batch] for v in "ORDPT"}
        res = {}
        caps_mods = [(f"res{l}", blocks[l]) for l in LAYERS] + [(f"att{l}", blocks[l].self_attn) for l in LAYERS] + \
                    [(f"mlp{l}", blocks[l].mlp) for l in LAYERS]
        lb, ls, capO = run(texts["O"], cls["O"], True, captures=caps_mods); res["O_on"] = (lb, ls)
        lb, ls, capOb = run(texts["O"], cls["O"], False, captures=[(f"res{SAE_LAYER}", blocks[SAE_LAYER])]); res["O_off"] = (lb, ls)
        lb, ls, capR = run(texts["R"], cls["R"], True, captures=[(f"res{l}", blocks[l]) for l in sorted({LAYERS[2], SAE_LAYER})]); res["R_on"] = (lb, ls)
        lb, ls, capRb = run(texts["R"], cls["R"], False, captures=[(f"res{SAE_LAYER}", blocks[SAE_LAYER])]); res["R_off"] = (lb, ls)
        lb, ls, capT = run(texts["T"], cls["T"], True, captures=[(f"res{l}", blocks[l]) for l in LAYERS]); res["T_on"] = (lb, ls)
        res["T_off"] = run(texts["T"], cls["T"], False)[:2]
        for v in "DP":
            res[f"{v}_on"] = run(texts[v], cls[v], True)[:2]; res[f"{v}_off"] = run(texts[v], cls[v], False)[:2]
        # implementation checks
        res["zero"] = run(texts["R"], cls["R"], True, patches=[(blocks[SAE_LAYER], lambda o: o)])[:2]
        for l, g in ((SAE_LAYER, "sess"), (LAYERS[2], "prof")):
            idx = [pair_idx(r, "R", "R", g) for r in batch]
            res[f"self_{g}@{l}"] = run(texts["R"], cls["R"], True, patches=[(blocks[l], lambda o, idx=idx, l=l: put(
                o, [d for _, d in idx], [capR[f"res{l}"][b, torch.as_tensor(s, device=dev)] for b, (s, _) in enumerate(idx)], None))])[:2]
        # activation and component patches from O (and T)
        for l in LAYERS:
            for g in ("prof", "sess"):
                for src, capS in (("O", capO), ("T", capT)):
                    idx = [pair_idx(r, src, "R", g) for r in batch]
                    res[f"resid_{g}@{l}<-{src}"] = run(texts["R"], cls["R"], True, patches=[(blocks[l], lambda o, idx=idx, capS=capS, l=l: put(
                        o, [d for _, d in idx], [capS[f"res{l}"][b, torch.as_tensor(s, device=dev)] for b, (s, _) in enumerate(idx)], None))])[:2]
            idx = [pair_idx(r, "O", "R", "sess") for r in batch]
            for comp, mod in (("attn", blocks[l].self_attn), ("mlp", blocks[l].mlp)):
                key = f"att{l}" if comp == "attn" else f"mlp{l}"
                res[f"{comp}_sess@{l}"] = run(texts["R"], cls["R"], True, patches=[(mod, lambda o, idx=idx, key=key: put(
                    o, [d for _, d in idx], [capO[key][b, torch.as_tensor(s, device=dev)] for b, (s, _) in enumerate(idx)], None))])[:2]
        # SAE mediation at layer 26 (codes from the same-input adapter-off states)
        dO = [(capO[f"res{SAE_LAYER}"][b] - capOb[f"res{SAE_LAYER}"][b]).float() for b in range(len(batch))]
        dR = [(capR[f"res{SAE_LAYER}"][b] - capRb[f"res{SAE_LAYER}"][b]).float() for b in range(len(batch))]
        edits = defaultdict(list)
        g_rng = torch.Generator(device="cpu").manual_seed(args.seed)
        for b, r in enumerate(batch):
            zO_all, zR_all = sae_code(dO[b]), sae_code(dR[b])
            for g in ("sess", "all"):
                groups = ("sess",) if g == "sess" else ("sess", "prof")
                s_idx, d_idx = [], []
                for gg in groups:
                    s_, d_ = pair_idx(r, "O", "R", gg); s_idx += s_; d_idx += d_
                sI = torch.as_tensor(s_idx, device=dev); dI = torch.as_tensor(d_idx, device=dev)
                for name, feats in (("sel", S), ("ctrl", C)):
                    fI = torch.as_tensor(feats, device=dev)
                    dz = zO_all[sI][:, fI] - zR_all[dI][:, fI]
                    e = (dz @ Dd[:, fI].T) * xs
                    edits[f"{name}_{g}"].append((d_idx, e, zR_all[dI], zO_all[sI], feats))
                edits[f"fulldelta_{g}"].append((d_idx, dO[b][sI] - dR[b][dI], None, None, None))
                if g == "sess":
                    e = edits["sel_sess"][-1][1]
                    rnd = torch.randn(e.shape, generator=g_rng).to(dev)
                    rnd = rnd / rnd.norm(dim=1, keepdim=True).clamp_min(1e-12) * e.norm(dim=1, keepdim=True)
                    edits["rand_sess"].append((d_idx, rnd, None, None, None))
            token_codes[r["id"]] = {"O_sess": zO_all[torch.as_tensor(positions(r, "O", "sess"), device=dev)][:, S + C].cpu().numpy(),
                                    "R_sess": zR_all[torch.as_tensor(positions(r, "R", "sess"), device=dev)][:, S + C].cpu().numpy(),
                                    "O_prof": zO_all[torch.as_tensor(positions(r, "O", "prof"), device=dev)][:, S + C].cpu().numpy(),
                                    "R_prof": zR_all[torch.as_tensor(positions(r, "R", "prof"), device=dev)][:, S + C].cpu().numpy()}
        for name, lst in edits.items():
            lb, ls, capP = run(texts["R"], cls["R"], True, patches=[(blocks[SAE_LAYER], lambda o, lst=lst: put(
                o, [x[0] for x in lst], [x[1] for x in lst], None, add=True))], captures=[("after", blocks[SAE_LAYER])])
            res[name] = (lb, ls)
            # realized-code verification after the hook and bf16 conversion
            for b, (d_idx, e, zR_t, zO_t, feats) in enumerate(lst):
                nzm = (e.norm(dim=1) > 0) if len(d_idx) else None
                if not len(d_idx) or not bool(nzm.any()):
                    verif[f"{name}|no_op_receivers"].append(1); continue
                verif[f"{name}|zero_edit_tokens"].append(int((~nzm).sum()))
                d_idx = [d for d, m in zip(d_idx, nzm.tolist()) if m]
                e = e[nzm]
                zR_t = zR_t[nzm] if zR_t is not None else None
                zO_t = zO_t[nzm] if zO_t is not None else None
                dI = torch.as_tensor(d_idx, device=dev)
                applied = (capP["after"][b, dI].float() - capR[f"res{SAE_LAYER}"][b, dI].float())
                verif[f"{name}|applied_over_intended_norm"].extend((applied.norm(dim=1) / e.norm(dim=1).clamp_min(1e-12)).cpu().tolist())
                verif[f"{name}|raw_edit_norm"].extend(e.norm(dim=1).cpu().tolist())
                if feats is None:
                    continue
                z_real = sae_code(capP["after"][b, dI].float() - capRb[f"res{SAE_LAYER}"][b, dI].float())
                fI = torch.as_tensor(feats, device=dev)
                target = zR_t.clone(); target[:, fI] = zO_t[:, fI]
                req = (zO_t[:, fI] - zR_t[:, fI]); err = (z_real[:, fI] - target[:, fI])
                moved = req.norm(dim=1) > 0
                verif[f"{name}|coord_abs_err"].extend(err.abs().flatten().cpu().tolist())
                if moved.any():
                    verif[f"{name}|norm_target_err"].extend((err[moved].norm(dim=1) / req[moved].norm(dim=1)).cpu().tolist())
                off = torch.ones(z_real.shape[1], dtype=torch.bool, device=dev); off[fI] = False
                verif[f"{name}|offtarget_l2"].extend((z_real[:, off] - zR_t[:, off]).norm(dim=1).cpu().tolist())
                a_, b_ = z_real > 0, target > 0
                verif[f"{name}|support_jaccard_realized_vs_intended"].extend(((a_ & b_).sum(1) / (a_ | b_).sum(1).clamp_min(1)).cpu().tolist())
                verif[f"{name}|share_intended_active_missing"].extend(((b_ & ~a_).sum(1) / b_.sum(1).clamp_min(1)).cpu().tolist())
                verif[f"{name}|nonfinite"].append(int(~torch.isfinite(z_real).all()))
        for cond, (lb, ls) in res.items():
            for b, r in enumerate(batch):
                rows.append({"stage": tag, "condition": cond, "receiver_id": r["id"], "user": r["user"], "kind": r["kind"],
                             "pair": r["pair"], "behavior_only": lb[b], "behavior_ses_only": ls[b]})

    # smoke: first pairs, timed; then the rest within the wall budget
    smoke = recs[: 2 * args.smoke_pairs]
    ts = time.time()
    for s0 in range(0, len(smoke), 2):
        do_batch(smoke[s0:s0 + 2], "smoke")
    per_rec = (time.time() - ts) / max(1, len(smoke))
    chk = {c: max(abs(a["behavior_only"] - b["behavior_only"]) for a, b in zip([x for x in rows if x["condition"] == c], [x for x in rows if x["condition"] == "R_on"]))
           for c in ("zero", f"self_sess@{SAE_LAYER}", f"self_prof@{LAYERS[2]}")}
    smoke_info = {"receivers": len(smoke), "seconds_per_receiver_batch2": per_rec, "checks_max_abs_vs_R_on": chk,
                  "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2)}
    (out / "h6_smoke.json").write_text(json.dumps(smoke_info, indent=1) + "\n")
    print(f"[h6] smoke {json.dumps(smoke_info)} ({time.time() - t0:.0f}s)", flush=True)
    if max(chk.values()) > 1e-5:
        raise SystemExit("zero or self patch does not reproduce the unpatched swapped run")
    rest = recs[2 * args.smoke_pairs:]
    remaining = args.wall_budget_s - (time.time() - t0) - 120
    cap_n = max(0, int(remaining / max(per_rec * 0.6, 1e-3)))   # batching amortises; 0.6 is conservative
    if cap_n < len(rest):
        keep_users = sorted({r["user"] for r in rest}, key=lambda u: H("trim", u))
        while sum(1 for r in rest if r["user"] in keep_users) > cap_n and keep_users:
            keep_users.pop()
        rest = [r for r in rest if r["user"] in keep_users]
        print(f"[h6] trimmed to {len(rest)} further receivers to fit the budget", flush=True)
    for s0 in range(0, len(rest), args.batch_size):
        do_batch(rest[s0:s0 + args.batch_size], "main")
        if s0 // args.batch_size % 4 == 0:
            print(f"[h6] {s0 + args.batch_size}/{len(rest)} ({time.time() - t0:.0f}s)", flush=True)
    with (out / "h6_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    vsum = {k: ({"n": len(v), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "p99": float(np.quantile(v, 0.99)), "max": float(np.max(v))}
                if not k.endswith(("no_op_receivers", "nonfinite", "zero_edit_tokens")) else int(sum(v))) for k, v in verif.items()}
    (out / "h6_verification.json").write_text(json.dumps(vsum, indent=1) + "\n")
    np.savez_compressed(out / "h6_token_codes.npz", feats=np.asarray(S + C), **{f"{k}__{n}": a for k, d in token_codes.items() for n, a in d.items()})
    print(f"[h6] peak GPU {torch.cuda.max_memory_allocated(0) / 2 ** 30:.2f} GiB; total {time.time() - t0:.0f}s", flush=True)
    print("H6_DONE", flush=True)


if __name__ == "__main__":
    main()
