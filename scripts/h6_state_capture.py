#!/usr/bin/env python3
"""Capture the actual H6 SAE-layer states behind the selected-feature edit (EXPLORATORY, 2026-09-27).

No behavioural outcome is computed. The script recreates the H6 population, alignment and
batches exactly (smoke: the first 2 * smoke_pairs receivers in batches of 2; then the rest in
batches of batch_size). It runs O and R with the adapter on and off, stopping each forward pass
after the SAE layer. For every receiver and aligned session token it records:

  * full TopK supports (indices and values) of the O and R codes under two arithmetic paths
      A (H6 intended codes):     delta = (h_adapted - h_base) subtracted in bf16, then cast to float32
      B (H6 verification codes): delta = h_adapted.float() - h_base.float()
    then u = (delta - x_mean) / x_std, preactivation u @ W_enc^T + b_enc, ReLU, TopK(k), in float32;
  * the H6 selected-feature decoder edit e = ((z_O,S - z_R,S) @ D_S^T) * x_std from path-A codes,
    exactly as H6 built it, applied with H6's put(add=True) at the aligned R positions;
  * the patched state captured immediately after the hook (bf16), the vector execution error
    applied - e, whether applied equals the bf16 sum h_R + bf16(e) exactly, and the realized codes
    of the patched state under paths A and B;
  * zero-edit diagnostics in the same batch configuration: an identity hook, and adding zeros
    through the same put() arithmetic.
It also checks the path-A selected/control codes against H6's saved h6_token_codes.npz.

States (bf16 bit patterns) are saved for every token with a nonzero selected or control edit.

  python scripts/h6_state_capture.py --config C --data-dir D --adapter-dir A --extract-dir X \\
      --frontier-dir F --split-dir S --h6-dir H6_OUT --out-dir O [--smoke-pairs 2 --batch-size 6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from h6_profile_patching import H, align, swap_lines, token_keys  # noqa: E402  (same helpers and salt as H6)


class _Stop(Exception):
    pass


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--config", "--data-dir", "--adapter-dir", "--extract-dir", "--frontier-dir", "--split-dir", "--h6-dir", "--out-dir"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--per-user", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--smoke-pairs", type=int, default=2)
    ap.add_argument("--sae-layer", type=int, default=26)
    ap.add_argument("--sae-subdir", default="layer_26/m02_k04")
    ap.add_argument("--selected", default="4596,7693,2302,3673,3455")
    ap.add_argument("--control", default="6596,8017,6608,2765,886")
    ap.add_argument("--context-field", default="dept")
    ap.add_argument("--discovery-file", default="discovery_users.txt")
    args = ap.parse_args()

    import pandas as pd
    from remote_common import load_yaml
    import token_class_decomposition as tcd
    from eval_token_delta_sae_causal import get_layer_module, load_eval_examples

    t0 = time.time(); cpu0 = time.process_time()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    S = [int(v) for v in args.selected.split(",")]; C = [int(v) for v in args.control.split(",")]
    SAE_LAYER = args.sae_layer

    # ---------------------------------------------------------------- population: verbatim from h6_profile_patching.py
    disc = (args.split_dir / args.discovery_file).read_text().split()
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    user = meta["user_id"].astype(str).to_numpy(); y = meta["y"].astype(int).to_numpy(); day = meta["day_index"].astype(int).to_numpy()
    dept = meta["text"].str.extract(args.context_field + r"=(\S+)")[0].to_numpy()
    mal_users = set(user[y == 1])
    benign_users = sorted(set(user) - mal_users)
    pairs = []
    for u in sorted(disc):
        att = sorted(np.flatnonzero((user == u) & (y == 1)), key=lambda i: H("recv", meta.loc[i, "example_id"]))[: args.per_user]
        ben = list(np.flatnonzero((user == u) & (y == 0))); used = set()
        for a in att:
            cands = sorted((b for b in ben if b not in used), key=lambda b: (abs(day[b] - day[a]), day[b] > day[a], b))
            if not cands:
                continue
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

    recs = []
    for a, b in pairs:
        for i, kind in ((a, "malicious"), (b, "benign")):
            pR, pT = partners(i)
            txt = meta.loc[i, "text"]
            dR = meta.loc[partner_day[pR], "text"].split("\n")
            v = {"O": txt, "R": swap_lines(txt, dR[0], dR[1])}
            enc = {k: encode(t) for k, t in v.items()}
            maps = {"R": align(enc["O"]["keys"], enc["R"]["keys"])}
            sess_o = [p for p in range(len(enc["O"]["cls"])) if enc["O"]["cls"][p] in ("SES", "SESCOUNT")]
            sess_v = [p for p in range(len(enc["R"]["cls"])) if enc["R"]["cls"][p] in ("SES", "SESCOUNT")]
            m = dict(maps["R"])
            if len(sess_o) != len(sess_v) or any(m.get(p) is None for p in sess_o) or \
                    [enc["O"]["ids"][p] for p in sess_o] != [enc["R"]["ids"][m[p]] for p in sess_o]:
                continue   # H6 excluded none; the manifest comparison below would catch any difference
            recs.append({"idx": i, "id": meta.loc[i, "example_id"], "user": user[i], "kind": kind, "partner_R": pR,
                         "partner_T": pT, "texts": v, "enc": enc, "maps": maps})
    h6_man = json.loads((args.h6_dir / "h6_manifest.json").read_text())
    mine = [{"id": r["id"], "user": r["user"], "kind": r["kind"], "partner_R": r["partner_R"], "partner_T": r["partner_T"]} for r in recs]
    if mine != h6_man["receivers"]:
        raise SystemExit("population differs from the H6 manifest")
    print(f"[capture] population identical to H6: {len(recs)} receivers ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- model and SAE (as H6)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    q = cfg["quantization"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], quantization_config=BitsAndBytesConfig(
        load_in_4bit=bool(q["load_in_4bit"]), bnb_4bit_quant_type=str(q["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(torch, str(q["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(q["bnb_4bit_use_double_quant"])), dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter_dir); model.config.use_cache = False; model.eval()
    dev = torch.device("cuda:0")
    block = get_layer_module(model, SAE_LAYER)
    sae_path = args.frontier_dir / args.sae_subdir / "delta_sae_model.pt"
    bundle = torch.load(sae_path, map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    We = sd["encoder.weight"].float().to(dev); be = sd["encoder.bias"].float().to(dev); Dd = sd["decoder.weight"].float().to(dev)
    xm = torch.tensor(np.asarray(bundle["x_mean"], dtype=np.float32).reshape(-1), device=dev)
    xs = torch.tensor(np.asarray(bundle["x_std"], dtype=np.float32).reshape(-1), device=dev)
    kk = int(bundle["k"])

    def sae_code(delta):
        pre = torch.relu(((delta - xm) / xs) @ We.T + be)
        v, i = torch.topk(pre, kk, dim=-1)
        return torch.zeros_like(pre).scatter_(1, i, v)

    def topk_of(delta):
        pre = torch.relu(((delta - xm) / xs) @ We.T + be)
        v, i = torch.topk(pre, kk, dim=-1)
        return i.to(torch.int16).cpu().numpy(), v.cpu().numpy()

    def run(texts, adapter=True, hook=None):
        """Forward pass stopped right after the SAE layer; returns that layer's (possibly patched) output."""
        b = tok(texts, return_tensors="pt", padding=True).to(dev)
        cap = {}

        def h(m, i, o):
            if hook is not None:
                o = hook(o)
            cap["x"] = (o[0] if isinstance(o, tuple) else o).detach().clone()
            raise _Stop

        hd = block.register_forward_hook(h)
        try:
            with torch.inference_mode():
                try:
                    if adapter:
                        model(**b)
                    else:
                        with model.disable_adapter():
                            model(**b)
                except _Stop:
                    pass
        finally:
            hd.remove()
        return cap["x"]

    def put(o, b_idx_dst, src, add=False):            # identical to H6's put()
        hs = o[0] if isinstance(o, tuple) else o
        for b, (dst, vals) in enumerate(zip(b_idx_dst, src)):
            if len(dst):
                idx = torch.as_tensor(dst, device=hs.device)
                v = vals.to(hs.dtype)
                hs[b, idx] = (hs[b, idx] + v) if add else v
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    def positions(r, variant):
        cls = r["enc"][variant]["cls"]
        return [p for p in range(len(cls)) if cls[p] in ("SES", "SESCOUNT")]

    def pair_idx(r):
        dstset = set(positions(r, "R"))
        pr = [(s, d) for s, d in r["maps"]["R"] if d in dstset]
        return [s for s, _ in pr], [d for _, d in pr]

    saved = np.load(args.h6_dir / "h6_token_codes.npz")
    bits = lambda t: t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    tokrows = {k: [] for k in ("recv", "src", "dst", "edited_sel", "edited_ctl", "e_sel_norm",
                               "O_A_idx", "O_A_val", "O_B_idx", "O_B_val", "R_A_idx", "R_A_val", "R_B_idx", "R_B_val",
                               "P_A_idx", "P_A_val", "P_B_idx", "P_B_val", "exec_err", "exec_rel", "exec_cos", "exec_bf16_exact")}
    states = {k: [] for k in ("recv", "dst", "xR", "bR", "xO", "bO", "e_sel", "e_ctl")}
    recv_rows, code_check = [], {"max_abs_O": 0.0, "max_abs_R": 0.0, "receivers_checked": 0}

    def do_batch(batch):
        texts = {v: [r["texts"][v] for r in batch] for v in "OR"}
        capO, capOb = run(texts["O"], True), run(texts["O"], False)
        capR, capRb = run(texts["R"], True), run(texts["R"], False)
        idx = [pair_idx(r) for r in batch]
        e_sel, e_ctl, zA = [], [], []
        for b, r in enumerate(batch):
            sI = torch.as_tensor(idx[b][0], device=dev); dI = torch.as_tensor(idx[b][1], device=dev)
            zO_A = sae_code((capO[b] - capOb[b]).float()); zR_A = sae_code((capR[b] - capRb[b]).float())
            po, pr = torch.as_tensor(positions(r, "O"), device=dev), torch.as_tensor(positions(r, "R"), device=dev)
            so, sr = saved[f"{r['id']}__O_sess"], saved[f"{r['id']}__R_sess"]
            code_check["max_abs_O"] = max(code_check["max_abs_O"], float(np.abs(zO_A[po][:, S + C].cpu().numpy() - so).max()) if so.size else 0.0)
            code_check["max_abs_R"] = max(code_check["max_abs_R"], float(np.abs(zR_A[pr][:, S + C].cpu().numpy() - sr).max()) if sr.size else 0.0)
            code_check["receivers_checked"] += 1
            fS, fC = torch.as_tensor(S, device=dev), torch.as_tensor(C, device=dev)
            e_sel.append(((zO_A[sI][:, fS] - zR_A[dI][:, fS]) @ Dd[:, fS].T) * xs)
            e_ctl.append(((zO_A[sI][:, fC] - zR_A[dI][:, fC]) @ Dd[:, fC].T) * xs)
            zA.append((zO_A, zR_A))
        capP = run(texts["R"], True, hook=lambda o: put(o, [d for _, d in idx], e_sel, add=True))
        capZ = run(texts["R"], True, hook=lambda o: o)
        capZ2 = run(texts["R"], True, hook=lambda o: put(o, [d for _, d in idx], [e * 0 for e in e_sel], add=True))
        for b, r in enumerate(batch):
            s_, d_ = idx[b]
            sI = torch.as_tensor(s_, device=dev); dI = torch.as_tensor(d_, device=dev)
            valid = torch.as_tensor(tok(r["texts"]["R"])["input_ids"], device=dev).numel()
            recv_rows.append({"recv": len(recv_rows), "id": r["id"], "user": r["user"], "kind": r["kind"],
                              "n_session_tokens": len(d_),
                              "zero_identity_equal": bool(torch.equal(capZ[b, :valid], capR[b, :valid])),
                              "zero_addzero_equal": bool(torch.equal(capZ2[b, :valid], capR[b, :valid])),
                              "zero_addzero_max_abs": float((capZ2[b, :valid].float() - capR[b, :valid].float()).abs().max())})
            if not len(d_):
                continue
            oa, ov = topk_of((capO[b, sI] - capOb[b, sI]).float()); ob, obv = topk_of(capO[b, sI].float() - capOb[b, sI].float())
            ra, rv = topk_of((capR[b, dI] - capRb[b, dI]).float()); rb, rbv = topk_of(capR[b, dI].float() - capRb[b, dI].float())
            pa, pv = topk_of((capP[b, dI] - capRb[b, dI]).float()); pb, pbv = topk_of(capP[b, dI].float() - capRb[b, dI].float())
            es, ec = e_sel[b], e_ctl[b]
            applied = capP[b, dI].float() - capR[b, dI].float()
            err = applied - es
            en = es.norm(dim=1)
            expected = capR[b, dI] + es.to(torch.bfloat16)
            n = len(d_)
            for k_, v_ in (("recv", [len(recv_rows) - 1] * n), ("src", s_), ("dst", d_), ("edited_sel", (en > 0).tolist()),
                           ("edited_ctl", (ec.norm(dim=1) > 0).tolist()), ("e_sel_norm", en.tolist()),
                           ("O_A_idx", oa), ("O_A_val", ov), ("O_B_idx", ob), ("O_B_val", obv),
                           ("R_A_idx", ra), ("R_A_val", rv), ("R_B_idx", rb), ("R_B_val", rbv),
                           ("P_A_idx", pa), ("P_A_val", pv), ("P_B_idx", pb), ("P_B_val", pbv),
                           ("exec_err", err.norm(dim=1).tolist()),
                           ("exec_rel", (err.norm(dim=1) / en.clamp_min(1e-12)).tolist()),
                           ("exec_cos", torch.nn.functional.cosine_similarity(applied, es, dim=1).nan_to_num(0.0).tolist()),
                           ("exec_bf16_exact", (capP[b, dI] == expected).all(dim=1).tolist())):
                tokrows[k_].extend(list(v_))
            keep = torch.as_tensor(((en > 0) | (ec.norm(dim=1) > 0)).cpu().numpy())
            if keep.any():
                kI = torch.nonzero(keep).flatten().to(dev)
                states["recv"].extend([len(recv_rows) - 1] * int(keep.sum())); states["dst"].extend(np.asarray(d_)[keep.numpy()].tolist())
                states["xR"].append(bits(capR[b, dI[kI]])); states["bR"].append(bits(capRb[b, dI[kI]]))
                states["xO"].append(bits(capO[b, sI[kI]])); states["bO"].append(bits(capOb[b, sI[kI]]))
                states["e_sel"].append(es[kI].cpu().numpy()); states["e_ctl"].append(ec[kI].cpu().numpy())

    smoke = recs[: 2 * args.smoke_pairs]
    for s0 in range(0, len(smoke), 2):
        do_batch(smoke[s0:s0 + 2])
    rest = recs[2 * args.smoke_pairs:]
    for s0 in range(0, len(rest), args.batch_size):
        do_batch(rest[s0:s0 + args.batch_size])
        print(f"[capture] {s0 + args.batch_size}/{len(rest)} ({time.time() - t0:.0f}s)", flush=True)

    arr = {k: np.asarray(v) for k, v in tokrows.items()}
    np.savez_compressed(out / "h6_capture_tokens.npz", **arr)
    np.savez_compressed(out / "h6_capture_states.npz", recv=np.asarray(states["recv"]), dst=np.asarray(states["dst"]),
                        **{k: np.concatenate(states[k]) for k in ("xR", "bR", "xO", "bO", "e_sel", "e_ctl")})
    (out / "h6_capture_receivers.json").write_text(json.dumps(recv_rows, indent=1) + "\n")
    metad = {
        "exploratory": True, "behavioural_outcomes_computed": False,
        "definitions": {
            "site": f"forward-hook output of get_layer_module(model, {SAE_LAYER}) = transformer block index {SAE_LAYER - 1} (0-based), hidden state {SAE_LAYER}",
            "positions": "aligned session positions (SES and SESCOUNT classes); O source and R destination from H6's key alignment",
            "adapted_state": "adapter enabled; bf16 residual output of that block",
            "base_state": "same input with model.disable_adapter(); bf16 residual output of that block",
            "path_A": "delta = (adapted - base) computed in bf16, then .float(); used by H6 to build intended codes and the edit",
            "path_B": "delta = adapted.float() - base.float(); used by H6's post-patch verification",
            "normalization": "u = (delta - x_mean) / x_std; pre = u @ W_enc^T + b_enc; ReLU; TopK(k); float32",
            "edit": "e = ((z_O,S - z_R,S) @ D_S^T) * x_std from path-A codes; applied with put(add=True): hs + e.to(bf16) in bf16",
            "patched_state": "block output captured after the hook (bf16)",
            "batches": f"first {2 * args.smoke_pairs} receivers in batches of 2, then batches of {args.batch_size}, as in H6"},
        "sae_checkpoint": str(sae_path), "sae_sha256": sha256(sae_path), "k": kk, "sae_width": int(We.shape[0]), "d_model": int(We.shape[1]),
        "adapter_sha256": sha256(args.adapter_dir / "adapter_model.safetensors"),
        "selected": S, "control": C, "receivers": len(recs),
        "saved_code_check_vs_h6": code_check,
        "gpu": torch.cuda.get_device_name(0), "wall_s": time.time() - t0, "cpu_s": time.process_time() - cpu0,
        "versions": {"torch": torch.__version__}}
    (out / "h6_capture_meta.json").write_text(json.dumps(metad, indent=1) + "\n")
    print(json.dumps({"saved_code_check_vs_h6": code_check, "edited_sel_tokens": int(arr["edited_sel"].sum()),
                      "zero_identity_all_equal": all(r["zero_identity_equal"] for r in recv_rows),
                      "zero_addzero_all_equal": all(r["zero_addzero_equal"] for r in recv_rows)}), flush=True)
    print("CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
