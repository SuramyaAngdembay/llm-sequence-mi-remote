#!/usr/bin/env python3
"""Bounded behavioural pilot of declared feasible feature interventions on the H6 population (EXPLORATORY).

Protocol: research/imperfect_interventions_2026-09-27/h6_states/PRECHECK_PROTOCOL.md, Part 3b.
Runs only on the output of a passed precheck. It recreates the H6 population and batches, then checks
that the SAE-layer O and R states equal the capture bit for bit, and stops otherwise. Conditions:

  on R: R, zero_R, dec (H6 decoder edit, path A, added), I2_sel, I1_sel, I2_ctl, rand_I2
  on O: O, zero_O, noise_I2, rand_noise

Projected states are written with put(add=False). Every patched SAE-layer state is re-encoded with
path B on the GPU, and its TopK support and values are saved for verification against the targets.

  python scripts/h6_intervention_pilot.py [H6 args] --h6-dir H6 --capture CAP --projection PROJ --out-dir OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from h6_profile_patching import H, align, swap_lines, token_keys  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--config", "--data-dir", "--adapter-dir", "--extract-dir", "--frontier-dir", "--split-dir", "--h6-dir",
              "--capture", "--projection", "--out-dir"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--per-user", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--smoke-pairs", type=int, default=2)
    ap.add_argument("--sae-layer", type=int, default=26)
    ap.add_argument("--sae-subdir", default="layer_26/m02_k04")
    ap.add_argument("--selected", default="4596,7693,2302,3673,3455")
    ap.add_argument("--context-field", default="dept")
    ap.add_argument("--discovery-file", default="discovery_users.txt")
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    import pandas as pd
    from remote_common import load_yaml
    import token_class_decomposition as tcd
    from eval_token_delta_sae_causal import get_layer_module, load_eval_examples

    t0 = time.time(); cpu0 = time.process_time()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    S = [int(v) for v in args.selected.split(",")]
    SAE_LAYER = args.sae_layer

    # ---------------------------------------------------------------- population: verbatim from h6_profile_patching.py
    disc = (args.split_dir / args.discovery_file).read_text().split()
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    meta = load_eval_examples(args.data_dir, scores)
    user = meta["user_id"].astype(str).to_numpy(); y = meta["y"].astype(int).to_numpy(); day = meta["day_index"].astype(int).to_numpy()
    dept = meta["text"].str.extract(args.context_field + r"=(\S+)")[0].to_numpy()
    mal_users = set(user[y == 1]); benign_users = sorted(set(user) - mal_users)
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

    def encode(text):
        e = tok(text, return_offsets_mapping=True)
        cls, _ = tcd.classify_tokens([tuple(o) for o in e["offset_mapping"]], tcd.cert_class_spans(text))
        return {"ids": e["input_ids"], "cls": np.asarray(cls), "keys": token_keys(text, e["offset_mapping"])}

    recs = []
    for a, b in pairs:
        for i, kind in ((a, "malicious"), (b, "benign")):
            pR, pT = partners(i)
            txt = meta.loc[i, "text"]; dR = meta.loc[partner_day[pR], "text"].split("\n")
            v = {"O": txt, "R": swap_lines(txt, dR[0], dR[1])}
            enc = {k: encode(t) for k, t in v.items()}
            recs.append({"idx": i, "id": meta.loc[i, "example_id"], "user": user[i], "kind": kind, "partner_R": pR, "partner_T": pT,
                         "texts": v, "enc": enc, "maps": {"R": align(enc["O"]["keys"], enc["R"]["keys"])}})
    mine = [{"id": r["id"], "user": r["user"], "kind": r["kind"], "partner_R": r["partner_R"], "partner_T": r["partner_T"]} for r in recs]
    if mine != json.loads((args.h6_dir / "h6_manifest.json").read_text())["receivers"]:
        raise SystemExit("population differs from the H6 manifest")

    # ---------------------------------------------------------------- interventions from the precheck
    st = np.load(args.capture / "h6_capture_states.npz")
    proj = np.load(args.projection / "projected_states.npz")
    precs = json.loads((args.projection / "projection_records.json").read_text())
    tk = np.load(args.capture / "h6_capture_tokens.npz")
    src_of = {(int(r), int(d)): int(s) for r, d, s in zip(tk["recv"], tk["dst"], tk["src"])}
    # per condition: receiver -> list of (position, float32 vector, mode) ; mode 'set' or 'add'
    conds = {c: {} for c in ("I2_sel", "I1_sel", "I2_ctl", "rand_I2", "noise_I2", "rand_noise")}
    rng = np.random.default_rng(args.seed)
    xR_all = (st["xR"].astype(np.uint32) << 16).view(np.float32); xO_all = (st["xO"].astype(np.uint32) << 16).view(np.float32)
    for rec in precs:
        i, rv, d = rec["i"], rec["recv"], rec["dst"]
        s_ = src_of[(rv, d)]
        for cname, key, pos, ctx in (("I2_sel", f"restore_sel__I2_decoder__{i}", d, xR_all[i]), ("I1_sel", f"restore_sel__I1_min_change__{i}", d, xR_all[i]),
                                     ("I2_ctl", f"restore_ctl__I2_decoder__{i}", d, xR_all[i]), ("noise_I2", f"noise_sel__I2_decoder__{i}", s_, xO_all[i])):
            if key in proj.files:
                x = proj[key]
                conds[cname].setdefault(rv, []).append((pos, x, "set"))
                if cname in ("I2_sel", "noise_I2"):
                    r_ = rng.standard_normal(x.size).astype(np.float32)
                    r_ *= np.linalg.norm(x.astype(np.float64) - ctx.astype(np.float64)) / np.linalg.norm(r_)
                    conds["rand_I2" if cname == "I2_sel" else "rand_noise"].setdefault(rv, []).append((pos, r_, "add"))
    ctx_of = {"I2_sel": "R", "I1_sel": "R", "I2_ctl": "R", "rand_I2": "R", "noise_I2": "O", "rand_noise": "O"}

    # ---------------------------------------------------------------- model and SAE (as H6)
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
    block = get_layer_module(model, SAE_LAYER)
    bundle = torch.load(args.frontier_dir / args.sae_subdir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    We = sd["encoder.weight"].float().to(dev); be = sd["encoder.bias"].float().to(dev); Dd = sd["decoder.weight"].float().to(dev)
    xm = torch.tensor(np.asarray(bundle["x_mean"], dtype=np.float32).reshape(-1), device=dev)
    xs = torch.tensor(np.asarray(bundle["x_std"], dtype=np.float32).reshape(-1), device=dev)
    kk = int(bundle["k"])

    def sae_code(delta):
        pre = torch.relu(((delta - xm) / xs) @ We.T + be)
        v, i = torch.topk(pre, kk, dim=-1)
        return torch.zeros_like(pre).scatter_(1, i, v)

    beh = set(tcd.CERT_VIEWS["behavior_only"])

    def losses(logits, ids, lens, cls_list):
        out_b = []
        for b in range(len(lens)):
            n = lens[b]
            lp = F.log_softmax(logits[b, : n - 1].float(), dim=-1)
            nll = -lp.gather(1, ids[b, 1:n].unsqueeze(1)).squeeze(1).cpu().numpy()
            out_b.append(float(nll[np.isin(cls_list[b][1:n], list(beh))].mean()))
        return out_b

    def run(texts, cls_list, adapter=True, hook=None):
        b = tok(texts, return_tensors="pt", padding=True).to(dev)
        lens = b["attention_mask"].sum(1).tolist(); cap = {}

        def h(m, i, o):
            if hook is not None:
                o = hook(o)
            cap["x"] = (o[0] if isinstance(o, tuple) else o).detach().clone()
            return o
        hd = block.register_forward_hook(h)
        try:
            with torch.inference_mode():
                if adapter:
                    lo = model(**b).logits
                else:
                    with model.disable_adapter():
                        lo = model(**b).logits
                lb = losses(lo, b["input_ids"], lens, cls_list)
        finally:
            hd.remove()
        return lb, cap["x"]

    def put(o, b_idx_dst, src, add=False):            # identical to H6's put()
        hs = o[0] if isinstance(o, tuple) else o
        for b, (dst, vals) in enumerate(zip(b_idx_dst, src)):
            if len(dst):
                idx = torch.as_tensor(dst, device=hs.device)
                v = vals.to(hs.dtype)
                hs[b, idx] = (hs[b, idx] + v) if add else v
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    def put_mixed(o, items):
        """items[b] = list of (pos, vec, mode)."""
        hs = o[0] if isinstance(o, tuple) else o
        for b, lst in enumerate(items):
            for pos, vec, mode in lst:
                v = torch.as_tensor(vec, device=hs.device).to(hs.dtype)
                hs[b, pos] = (hs[b, pos] + v) if mode == "add" else v
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    def positions(r, variant):
        cls = r["enc"][variant]["cls"]
        return [p for p in range(len(cls)) if cls[p] in ("SES", "SESCOUNT")]

    def pair_idx(r):
        dstset = set(positions(r, "R"))
        pr = [(s, d) for s, d in r["maps"]["R"] if d in dstset]
        return [s for s, _ in pr], [d for _, d in pr]

    state_key = {(int(r), int(d)): i for i, (r, d) in enumerate(zip(st["recv"], st["dst"]))}
    rows, verif, checks = [], [], {"state_bitwise_mismatch_tokens": 0, "state_tokens_checked": 0, "zero_R_max_abs": 0.0, "zero_O_max_abs": 0.0}
    h6_rows = {(r["condition"], r["receiver_id"]): float(r["behavior_only"]) for r in csv.DictReader((args.h6_dir / "h6_rows.csv").open())}

    def do_batch(batch, offset):
        texts = {v: [r["texts"][v] for r in batch] for v in "OR"}
        cls = {v: [r["enc"][v]["cls"] for r in batch] for v in "OR"}
        res = {}
        res["O"], capO = run(texts["O"], cls["O"], True)
        _, capOb = run(texts["O"], cls["O"], False)
        res["R"], capR = run(texts["R"], cls["R"], True)
        _, capRb = run(texts["R"], cls["R"], False)
        idx = [pair_idx(r) for r in batch]
        # bitwise check against the capture
        for b in range(len(batch)):
            rv = offset + b
            for s_, d_ in zip(*idx[b]):
                i = state_key.get((rv, d_))
                if i is None:
                    continue
                checks["state_tokens_checked"] += 1
                a = capR[b, d_].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                o_ = capO[b, s_].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                checks["state_bitwise_mismatch_tokens"] += int(not (np.array_equal(a, st["xR"][i]) and np.array_equal(o_, st["xO"][i])))
        if checks["state_bitwise_mismatch_tokens"]:
            raise SystemExit("SAE-layer states differ from the capture; the projections do not apply")
        res["zero_R"], _ = run(texts["R"], cls["R"], True, hook=lambda o: o)
        res["zero_O"], _ = run(texts["O"], cls["O"], True, hook=lambda o: o)
        # H6 decoder edit, path A, exactly as H6
        e_sel = []
        for b in range(len(batch)):
            sI = torch.as_tensor(idx[b][0], device=dev); dI = torch.as_tensor(idx[b][1], device=dev)
            zO = sae_code((capO[b] - capOb[b]).float()); zR = sae_code((capR[b] - capRb[b]).float())
            fS = torch.as_tensor(S, device=dev)
            e_sel.append(((zO[sI][:, fS] - zR[dI][:, fS]) @ Dd[:, fS].T) * xs)
        res["dec"], _ = run(texts["R"], cls["R"], True, hook=lambda o: put(o, [d for _, d in idx], e_sel, add=True))
        for cname, items_all in conds.items():
            ctx = ctx_of[cname]
            items = [items_all.get(offset + b, []) for b in range(len(batch))]
            base_b = capRb if ctx == "R" else capOb
            res[cname], capP = run(texts[ctx], cls[ctx], True, hook=lambda o, items=items: put_mixed(o, items))
            for b, lst in enumerate(items):
                for pos, vec, mode in lst:
                    z = torch.relu(((capP[b, pos].float() - base_b[b, pos].float()) - xm) / xs @ We.T + be)
                    v_, i_ = torch.topk(z, kk)
                    verif.append({"condition": cname, "recv": offset + b, "pos": int(pos), "idx": i_.cpu().tolist(), "val": v_.cpu().tolist()})
        for b, r in enumerate(batch):
            checks["zero_R_max_abs"] = max(checks["zero_R_max_abs"], abs(res["zero_R"][b] - res["R"][b]))
            checks["zero_O_max_abs"] = max(checks["zero_O_max_abs"], abs(res["zero_O"][b] - res["O"][b]))
            for c, lb in res.items():
                rows.append({"condition": c, "receiver_id": r["id"], "user": r["user"], "kind": r["kind"], "behavior_only": lb[b],
                             "n_patched_tokens": len(conds[c].get(offset + b, [])) if c in conds else None})

    smoke = recs[: 2 * args.smoke_pairs]
    for s0 in range(0, len(smoke), 2):
        do_batch(smoke[s0:s0 + 2], s0)
    rest = recs[2 * args.smoke_pairs:]
    for s0 in range(0, len(rest), args.batch_size):
        do_batch(rest[s0:s0 + args.batch_size], 2 * args.smoke_pairs + s0)
        print(f"[pilot] {s0 + args.batch_size}/{len(rest)} ({time.time() - t0:.0f}s)", flush=True)
    # reproduction of H6's selected-feature edit and baselines
    rep = {"dec_vs_h6_sel_sess_max_abs": max(abs(r["behavior_only"] - h6_rows[("sel_sess", r["receiver_id"])]) for r in rows if r["condition"] == "dec"),
           "R_vs_h6_R_on_max_abs": max(abs(r["behavior_only"] - h6_rows[("R_on", r["receiver_id"])]) for r in rows if r["condition"] == "R"),
           "O_vs_h6_O_on_max_abs": max(abs(r["behavior_only"] - h6_rows[("O_on", r["receiver_id"])]) for r in rows if r["condition"] == "O")}
    checks.update(rep)
    with (out / "pilot_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "pilot_verification.json").write_text(json.dumps(verif) + "\n")
    checks.update({"gpu": torch.cuda.get_device_name(0), "wall_s": time.time() - t0, "cpu_s": time.process_time() - cpu0,
                   "patched_tokens": {c: sum(len(v) for v in d.values()) for c, d in conds.items()}, "seed": args.seed})
    (out / "pilot_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks), flush=True)
    print("PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
