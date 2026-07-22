#!/usr/bin/env python3
"""Attribute top token-SAE features to serialization fields.

Tests the profile-confound hypothesis: do the causally implicated features
activate on behavioral session tokens (SES lines) or on static profile tokens
(DAY organizational header / PSY psychometric line)?

For each selected feature, on positive examples (plus a benign contrast set
drawn from the same chunks), reports post-top-k activation mass by field
class, the field's token-share baseline, the enrichment ratio, and the
highest-activating tokens with their surrounding text.

Field classes by serialized line: DAY (line 1), PSY (line 2), SESCOUNT
(line 3), SES (all remaining lines). Token positions follow the extraction
convention: index into the truncated (max_seq_len) unpadded tokenization of
`text` by the adapter tokenizer.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from sae_core import TopKSAE
from eval_token_delta_sae_causal import _select_token_chunks, read_jsonl

CLASSES = ["DAY", "PSY", "SESCOUNT", "SES"]


def token_classes_for_text(text: str, tokenizer, max_seq_len: int) -> tuple[list[str], list[str]]:
    enc = tokenizer(text, truncation=True, max_length=max_seq_len, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    # char offset -> line index
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)
    line_starts_arr = np.asarray(line_starts)
    classes: list[str] = []
    tokens: list[str] = []
    ids = enc["input_ids"]
    for (a, b), tid in zip(offsets, ids):
        line_idx = int(np.searchsorted(line_starts_arr, a, side="right") - 1)
        if line_idx == 0:
            cls = "DAY"
        elif line_idx == 1:
            cls = "PSY"
        elif line_idx == 2:
            cls = "SESCOUNT"
        else:
            cls = "SES"
        classes.append(cls)
        tokens.append(tokenizer.decode([tid]))
    return classes, tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--n-benign-contrast", type=int, default=500)
    ap.add_argument("--sae-batch-size", type=int, default=8192)
    ap.add_argument("--top-tokens", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    model = TopKSAE(d_in=int(bundle["d_in"]), d_latent=int(bundle["d_latent"]), k=int(bundle["k"])).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    x_mean = torch.from_numpy(np.asarray(bundle["x_mean"], dtype=np.float32)).to(device)
    x_std = torch.from_numpy(np.asarray(bundle["x_std"], dtype=np.float32)).to(device)

    feats = pd.read_csv(cfg_dir / "delta_sae_top_features.csv")
    top5 = [int(x) for x in feats.sort_values("row_gap", ascending=False).head(5)["feature_id"]]

    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet")
    pos_examples = set(scores.loc[scores["y"] == 1, "example_idx"].astype(int))
    texts = {}
    for i, ex in enumerate(read_jsonl(args.data_dir / "eval.jsonl")):
        texts[i] = ex["text"]

    layer_dir = args.extract_dir / f"layer_{args.layer}"
    chunk_paths = sorted(layer_dir.glob("chunk_*.pt"))
    keep_arr = np.asarray(sorted(pos_examples), dtype=np.int64)
    selected = _select_token_chunks(args.extract_dir, args.layer, chunk_paths, keep_arr)

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), use_fast=True)

    rng = np.random.default_rng(args.seed)
    benign_pool: set[int] = set()

    # accumulators: mass[feature][group][class]; token records for top activations
    mass = {f: {g: defaultdict(float) for g in ("positive", "benign")} for f in top5}
    active_counts = {f: {g: defaultdict(int) for g in ("positive", "benign")} for f in top5}
    token_share = {g: defaultdict(int) for g in ("positive", "benign")}
    top_records: Dict[int, List[tuple]] = {f: [] for f in top5}
    class_cache: Dict[int, tuple] = {}

    def classes_for(example_idx: int):
        if example_idx not in class_cache:
            class_cache[example_idx] = token_classes_for_text(
                texts[example_idx], tokenizer, args.max_seq_len
            )
        return class_cache[example_idx]

    for path in selected:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        vecs = np.asarray(obj["delta"], dtype=np.float32)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        pos = np.asarray(obj["position"], dtype=np.int64)
        uniq = np.unique(idx)
        benign_here = [int(e) for e in uniq if int(e) not in pos_examples]
        for e in benign_here:
            if len(benign_pool) < args.n_benign_contrast:
                benign_pool.add(e)
        keep_examples = pos_examples | benign_pool
        m = np.isin(idx, np.asarray(sorted(keep_examples), dtype=np.int64))
        vecs, idx, pos = vecs[m], idx[m], pos[m]
        if len(idx) == 0:
            continue
        for start in range(0, len(idx), args.sae_batch_size):
            sl = slice(start, start + args.sae_batch_size)
            xb = torch.from_numpy(vecs[sl]).to(device)
            xb = (xb - x_mean) / x_std
            with torch.no_grad():
                _, z = model(xb)
            z_top = z[:, top5].cpu().numpy()
            for row_i in range(z_top.shape[0]):
                e = int(idx[sl][row_i]); p = int(pos[sl][row_i])
                group = "positive" if e in pos_examples else "benign"
                cls_list, tok_list = classes_for(e)
                if p >= len(cls_list):
                    continue
                cls = cls_list[p]
                token_share[group][cls] += 1
                for fi, f in enumerate(top5):
                    v = float(z_top[row_i, fi])
                    if v > 0.0:
                        mass[f][group][cls] += v
                        active_counts[f][group][cls] += 1
                        if group == "positive":
                            top_records[f].append((v, e, p, cls, tok_list[p]))

    report = {"layer": args.layer, "latent_mult": args.latent_mult, "k": args.k,
              "top5": top5, "n_benign_contrast": len(benign_pool)}
    per_feature = []
    for f in top5:
        row = {"feature_id": f}
        for g in ("positive", "benign"):
            total = sum(mass[f][g].values()) or 1.0
            share_total = sum(token_share[g].values()) or 1
            for c in CLASSES:
                frac = mass[f][g][c] / total
                base = token_share[g][c] / share_total
                row[f"{g}_{c}_mass_frac"] = round(frac, 4)
                row[f"{g}_{c}_token_share"] = round(base, 4)
                row[f"{g}_{c}_enrichment"] = round(frac / base, 3) if base > 0 else None
        recs = sorted(top_records[f], reverse=True)[: args.top_tokens]
        row["top_tokens"] = [
            {"act": round(v, 3), "example": e, "pos": p, "class": c, "token": t}
            for v, e, p, c, t in recs
        ]
        per_feature.append(row)
    report["features"] = per_feature

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "feature_token_attribution.json").write_text(json.dumps(report, indent=2))

    lines = ["# Feature Token Attribution", "",
             f"Config: layer {args.layer}, m={args.latent_mult}, k={args.k}; top5 = {top5}", ""]
    for row in per_feature:
        f = row["feature_id"]
        lines.append(f"## Feature {f}")
        lines.append("")
        lines.append("| class | pos mass frac | pos token share | pos enrichment | benign mass frac |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for c in CLASSES:
            lines.append(
                f"| {c} | {row[f'positive_{c}_mass_frac']} | {row[f'positive_{c}_token_share']} "
                f"| {row[f'positive_{c}_enrichment']} | {row[f'benign_{c}_mass_frac']} |")
        lines.append("")
        lines.append("Top activating tokens (positive examples): " + ", ".join(
            f"`{r['token'].strip() or r['token']!r}`({r['class']},{r['act']})" for r in row["top_tokens"][:12]))
        lines.append("")
    (args.out_dir / "FEATURE_TOKEN_ATTRIBUTION.md").write_text("\n".join(lines))
    print(json.dumps({k: report[k] for k in ("top5", "n_benign_contrast")}, indent=1))
    for row in per_feature:
        print(f"feature {row['feature_id']}: SES mass {row['positive_SES_mass_frac']}, "
              f"DAY {row['positive_DAY_mass_frac']}, PSY {row['positive_PSY_mass_frac']}")


if __name__ == "__main__":
    main()
