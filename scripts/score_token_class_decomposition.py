#!/usr/bin/env python3
"""Score a serialized pool and cache per-example loss sums/counts by token class.

Reproduces the detector score of `extract_adapter_deltas.py` (mean next-token
NLL of the adapted model over attention-valid targets, same tokenizer, same
`max_seq_len` truncation) while additionally accumulating the loss of each
scored target into the class of the token being predicted.

The cache makes every later score view (full / profile-only / behaviour-only)
a table operation: no further inference is needed to change the target set.

Only the adapted model is loaded by default. `--with-base` additionally scores
the frozen base model (doubles the cost) and is not needed for the detector
score, which is `adapted_nll`.

Self-check (`--reference-scores`) compares the recomputed full mean NLL against
a previously cached `example_scores.parquet` on the same pool, and reports
token-count mismatches, which would indicate a tokenizer or truncation drift.

Example
-------
python scripts/score_token_class_decomposition.py \
  --config configs/qwen3_8b_qlora_session_targeted.yaml \
  --data-dir  $SCRATCH/p1_8b/pool_full \
  --adapter-dir $SCRATCH/p1_8b/adapter_full/adapter \
  --output    $SCRATCH/p1_8b/classdecomp_full/class_scores.parquet \
  --schema cert --max-examples 512 \
  --reference-scores $SCRATCH/p1_8b/deltas_full/example_scores.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from remote_common import dump_json, ensure_dir, load_yaml, read_jsonl
from token_class_decomposition import (
    CERT_CLASSES,
    CERT_SUBCLASSES,
    CERT_VIEWS,
    LANL_CLASSES,
    LANL_VIEWS,
    accumulate_class_losses,
    cert_class_spans,
    cert_subclass_spans,
    classify_subclass,
    classify_tokens,
    lanl_class_spans,
)

SCHEMAS = {
    "cert": {
        "classes": CERT_CLASSES,
        "views": CERT_VIEWS,
        "spans": cert_class_spans,
        "subspans": cert_subclass_spans,
        "subclass": "DAY_WEEK",
    },
    "lanl": {
        "classes": LANL_CLASSES,
        "views": LANL_VIEWS,
        "spans": lanl_class_spans,
        "subspans": lambda text: [],
        "subclass": "NONE",
    },
}


def token_losses(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-target cross entropy, shape [B, L-1], in fp32.

    Sequence-chunked exactly as `extract_adapter_deltas.per_example_nll` to
    bound the fp32 logits workspace; `token_loss[b, i]` is the loss of the
    target at position i+1.
    """
    shift_labels = input_ids[:, 1:].contiguous()
    seq_len = shift_labels.shape[1]
    out = torch.empty(shift_labels.shape, device=logits.device, dtype=torch.float32)
    seq_chunk = 256
    for s0 in range(0, seq_len, seq_chunk):
        s1 = min(s0 + seq_chunk, seq_len)
        chunk_logits = logits[:, s0:s1, :].float().contiguous()
        chunk_labels = shift_labels[:, s0:s1]
        out[:, s0:s1] = F.cross_entropy(
            chunk_logits.view(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        ).view(chunk_labels.size())
        del chunk_logits, chunk_labels
    return out


def dir_digest(path: Path, patterns=("*.json", "*.txt", "*.model", "*.safetensors")) -> str:
    """Stable digest of the files that define a checkpoint/tokenizer."""
    h = hashlib.sha256()
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(path.glob(pat)))
    for f in sorted(files):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        if f.suffix in (".json", ".txt"):
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="parquet path for the class cache")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--schema", choices=sorted(SCHEMAS), default="cert")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--length-sorted", action="store_true", help="sort by token length to cut padding")
    ap.add_argument("--with-base", action="store_true", help="also score the frozen base model")
    ap.add_argument("--reference-scores", type=Path, default=None, help="cached example_scores.parquet to self-check against")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    schema = SCHEMAS[args.schema]
    classes: List[str] = list(schema["classes"])
    subclass: str = schema["subclass"]

    cfg = load_yaml(args.config)
    model_name = cfg["model_name_or_path"]
    max_seq_len = int(cfg["training"]["max_seq_len"])

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    split_path = args.data_dir / f"{args.split}.jsonl"
    examples = list(read_jsonl(split_path))
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise RuntimeError(f"No examples found in {split_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(cfg["quantization"]["load_in_4bit"]),
        bnb_4bit_quant_type=str(cfg["quantization"]["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(torch, str(cfg["quantization"]["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(cfg["quantization"]["bnb_4bit_use_double_quant"]),
    )
    common_kwargs = dict(
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if bool(cfg["training"].get("bf16", True)) else torch.float16,
        device_map="auto",
    )
    adapted_backbone = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    adapted_model = PeftModel.from_pretrained(adapted_backbone, str(args.adapter_dir))
    adapted_model.eval()
    base_model = None
    if args.with_base:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
        base_model.eval()

    order = list(range(len(examples)))
    if args.length_sorted:
        approx = [len(examples[i]["text"]) for i in order]
        order.sort(key=lambda i: approx[i])

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    n_done = 0
    peak_mem = 0.0
    for bstart in range(0, len(order), args.batch_size):
        idxs = order[bstart : bstart + args.batch_size]
        batch = [examples[i] for i in idxs]
        texts = [ex["text"] for ex in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_offsets_mapping=True,
        )
        offsets_batch = enc.pop("offset_mapping")
        enc = {k: v.to(adapted_model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = adapted_model(**enc, return_dict=True)
            tl = token_losses(out.logits, enc["input_ids"]).cpu().numpy()
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tl_base = None
            if base_model is not None:
                out_b = base_model(**enc, return_dict=True)
                tl_base = token_losses(out_b.logits, enc["input_ids"]).cpu().numpy()
                del out_b
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        attn = enc["attention_mask"].cpu().numpy()
        shift_mask = attn[:, 1:]

        for bi, (ex, ex_idx) in enumerate(zip(batch, idxs)):
            text = ex["text"]
            n_valid = int(attn[bi].sum())
            offs = [tuple(map(int, o)) for o in offsets_batch[bi].tolist()[:n_valid]]
            spans = schema["spans"](text)
            tok_classes, crosses = classify_tokens(offs, spans)
            subflags = (
                classify_subclass(offs, schema["subspans"](text))
                if subclass != "NONE"
                else [False] * len(offs)
            )
            # pad the per-token lists out to the padded width; padded targets
            # are excluded by shift_mask, so their class never matters.
            pad_to = attn.shape[1]
            tok_classes += ["SPECIAL"] * (pad_to - len(tok_classes))
            crosses += [False] * (pad_to - len(crosses))
            subflags += [False] * (pad_to - len(subflags))

            acc = accumulate_class_losses(
                tok_classes,
                tl[bi],
                shift_mask[bi],
                classes,
                crosses=crosses,
                subclass_flags=subflags,
                subclass_name=subclass if subclass != "NONE" else "DAY_WEEK",
            )
            row: Dict[str, Any] = {
                "example_idx": int(ex_idx),
                "example_id": ex["example_id"],
                "user_id": ex["user_id"],
                "day_index": ex.get("day_index", -1),
                "split": ex["split"],
                "y": int(ex["y"]),
                "n_tokens": n_valid,
                "adapted_nll": float(acc["loss_sum_total"] / max(acc["n_targets"], 1)),
            }
            row.update({k: (float(v) if k.startswith("loss_sum") else int(v)) for k, v in acc.items()})
            if tl_base is not None:
                acc_b = accumulate_class_losses(
                    tok_classes, tl_base[bi], shift_mask[bi], classes,
                    crosses=crosses, subclass_flags=subflags,
                    subclass_name=subclass if subclass != "NONE" else "DAY_WEEK",
                )
                row["base_nll"] = float(acc_b["loss_sum_total"] / max(acc_b["n_targets"], 1))
                row["delta_nll"] = row["adapted_nll"] - row["base_nll"]
                for k, v in acc_b.items():
                    row[f"base_{k}"] = float(v) if k.startswith("loss_sum") else int(v)
            rows.append(row)

        n_done += len(batch)
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)
        if args.progress_every and n_done % args.progress_every < args.batch_size:
            rate = n_done / max(time.time() - t0, 1e-9)
            print(f"[decomp] {n_done}/{len(order)} ex  {rate:.1f} ex/s  peak {peak_mem:.1f} GB", flush=True)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows).sort_values("example_idx").reset_index(drop=True)
    out_path = args.output
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)

    # ---------------------------------------------------------- self-checks
    checks: Dict[str, Any] = {}
    part_sum = df[[f"loss_sum_{c}" for c in classes]].sum(axis=1)
    part_n = df[[f"n_{c}" for c in classes]].sum(axis=1)
    checks["partition_loss_max_abs_err"] = float((part_sum - df["loss_sum_total"]).abs().max())
    checks["partition_count_mismatches"] = int((part_n != df["n_targets"]).sum())
    checks["targets_equal_tokens_minus_one"] = int((df["n_targets"] != df["n_tokens"] - 1).sum())
    recon = df["loss_sum_total"] / df["n_targets"]
    checks["mean_nll_recon_max_abs_err"] = float((recon - df["adapted_nll"]).abs().max())
    checks["boundary_target_frac"] = float(
        df[[f"n_boundary_{c}" for c in classes]].sum(axis=1).sum() / max(df["n_targets"].sum(), 1)
    )
    checks["class_target_share"] = {
        c: float(df[f"n_{c}"].sum() / max(df["n_targets"].sum(), 1)) for c in classes
    }
    if subclass != "NONE":
        checks["subclass_target_share"] = float(
            df[f"n_{subclass}"].sum() / max(df["n_targets"].sum(), 1)
        )

    if args.reference_scores is not None and args.reference_scores.exists():
        ref_all = pd.read_parquet(args.reference_scores)
        # join on example_id when both sides carry it: robust to any difference
        # in row order between this run and the cached one.
        key = "example_id" if "example_id" in ref_all.columns and "example_id" in df.columns else "example_idx"
        ref = ref_all[[key, "adapted_nll", "n_tokens"]]
        m = df.merge(ref, on=key, suffixes=("", "_ref"))
        checks["reference_join_key"] = key
        d = (m["adapted_nll"] - m["adapted_nll_ref"]).abs()
        checks["reference"] = {
            "n_matched": int(len(m)),
            "n_token_mismatches": int((m["n_tokens"] != m["n_tokens_ref"]).sum()),
            "nll_max_abs_diff": float(d.max()),
            "nll_mean_abs_diff": float(d.mean()),
            "nll_p99_abs_diff": float(d.quantile(0.99)),
            "nll_corr": float(np.corrcoef(m["adapted_nll"], m["adapted_nll_ref"])[0, 1]) if len(m) > 2 else float("nan"),
            "spearman_like_rank_corr": float(
                np.corrcoef(m["adapted_nll"].rank(), m["adapted_nll_ref"].rank())[0, 1]
            ) if len(m) > 2 else float("nan"),
        }

    manifest = {
        "script": "score_token_class_decomposition.py",
        "model_name_or_path": model_name,
        "config": str(args.config),
        "adapter_dir": str(args.adapter_dir),
        "adapter_digest": dir_digest(args.adapter_dir),
        "tokenizer_digest": dir_digest(args.adapter_dir, patterns=("tokenizer*.json", "vocab.json", "merges.txt", "special_tokens_map.json")),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "n_examples": int(len(df)),
        "max_seq_len": max_seq_len,
        "batch_size": args.batch_size,
        "length_sorted": bool(args.length_sorted),
        "with_base": bool(args.with_base),
        "schema": args.schema,
        "classes": classes,
        "subclass": subclass,
        "views": {k: list(v) for k, v in schema["views"].items()},
        "scoring_convention": "mean next-token NLL over attention-valid targets; token 0 never a target; loss attributed to the class of the predicted token",
        "elapsed_sec": round(elapsed, 2),
        "examples_per_sec": round(len(df) / max(elapsed, 1e-9), 3),
        "peak_gpu_gb": round(peak_mem, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "checks": checks,
    }
    dump_json(out_path.parent / (out_path.stem + "_manifest.json"), manifest)
    print(json.dumps(checks, indent=2))
    print(f"[decomp] wrote {out_path} ({len(df)} rows) in {elapsed:.1f}s "
          f"({len(df)/max(elapsed,1e-9):.1f} ex/s, peak {peak_mem:.1f} GB)")


if __name__ == "__main__":
    main()
