#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from remote_common import dump_json, ensure_dir, load_yaml, read_jsonl


def per_example_nll(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Sequence-chunked CE: bounds the fp32 logits workspace on small GPUs
    (same approach as score_adapter_examples.per_example_nll)."""
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()
    seq_len = shift_labels.shape[1]
    seq_chunk = 256
    loss_sum = torch.zeros(shift_labels.shape[0], device=logits.device, dtype=torch.float32)
    for s0 in range(0, seq_len, seq_chunk):
        s1 = min(s0 + seq_chunk, seq_len)
        chunk_logits = logits[:, s0:s1, :].float().contiguous()
        chunk_labels = shift_labels[:, s0:s1]
        token_loss = F.cross_entropy(
            chunk_logits.view(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        ).view(chunk_labels.size())
        loss_sum += (token_loss * shift_mask[:, s0:s1]).sum(dim=1)
        del chunk_logits, chunk_labels, token_loss
        if logits.device.type == "cuda":
            torch.cuda.empty_cache()
    denom = shift_mask.sum(dim=1).clamp_min(1.0)
    return loss_sum / denom


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def flush_chunk(path: Path, payload: Dict[str, Any]) -> None:
    torch.save(payload, path, pickle_protocol=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--chunk-examples", type=int, default=4096)
    ap.add_argument("--pool-unit", choices=["mean", "token"], default=None)
    ap.add_argument("--layers", default="", help="optional comma-separated hidden-state layers to extract")
    ap.add_argument("--base-via-disable-adapter", action="store_true",
                    help="score the base model by disabling the adapter on the one loaded model (fits 8 GB GPUs)")
    ap.add_argument("--start-index", type=int, default=0, help="first example (global index) of this shard")
    ap.add_argument("--end-index", type=int, default=0, help="end (exclusive) of this shard; 0 = all")
    ap.add_argument("--max-batch-tokens", type=int, default=0,
                    help="if > 0: deterministic length-aware batches of at most --batch-size examples and at most this many "
                         "padded tokens, never crossing a --chunk-examples boundary (a longer example is scored alone)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies for delta extraction. Install transformers, peft, bitsandbytes."
        ) from exc

    pool_unit = str(args.pool_unit or cfg.get("delta_extraction", {}).get("pool_unit", "mean"))
    if pool_unit not in {"mean", "token"}:
        raise ValueError(f"Unsupported pool_unit={pool_unit}")

    out_dir = ensure_dir(args.output_dir)
    split_path = args.data_dir / f"{args.split}.jsonl"
    examples = list(read_jsonl(split_path))
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    shard_start = int(args.start_index)
    if shard_start % int(args.chunk_examples):
        raise ValueError("--start-index must be a multiple of --chunk-examples so chunk ids stay global")
    examples = examples[shard_start: (args.end_index or len(examples))]
    if not examples:
        raise RuntimeError(f"No examples found in {split_path}")

    model_name = cfg["model_name_or_path"]
    max_seq_len = int(cfg["training"]["max_seq_len"])
    if args.layers.strip():
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layers = [int(x) for x in cfg["delta_extraction"]["layers"]]
    import torch

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(cfg["quantization"]["load_in_4bit"]),
        bnb_4bit_quant_type=str(cfg["quantization"]["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(torch, str(cfg["quantization"]["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(cfg["quantization"]["bnb_4bit_use_double_quant"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = dict(
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if bool(cfg["training"].get("bf16", True)) else torch.float16,
        device_map="auto",
    )
    adapted_backbone = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    adapted_model = PeftModel.from_pretrained(adapted_backbone, args.adapter_dir)
    adapted_model.eval()
    if args.base_via_disable_adapter:
        base_model = adapted_model
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
        base_model.eval()
    from contextlib import nullcontext
    base_context = adapted_model.disable_adapter if args.base_via_disable_adapter else nullcontext

    chunk_meta: List[Dict[str, Any]] = []
    scores_rows: List[Dict[str, Any]] = []
    current: Dict[int, Dict[str, List[np.ndarray]]] = {layer: {"delta": [], "example_idx": [], "position": []} for layer in layers}
    current_example_count = 0
    chunk_id = shard_start // int(args.chunk_examples)

    def maybe_flush() -> None:
        nonlocal chunk_id, current_example_count, current
        if current_example_count == 0:
            return
        for layer in layers:
            layer_dir = ensure_dir(out_dir / f"layer_{layer}")
            delta_arr = np.concatenate(current[layer]["delta"], axis=0).astype(np.float16)
            example_idx = np.concatenate(current[layer]["example_idx"], axis=0).astype(np.int64)
            payload = {"delta": delta_arr, "example_idx": example_idx}
            if current[layer]["position"]:
                payload["position"] = np.concatenate(current[layer]["position"], axis=0).astype(np.int32)
            chunk_path = layer_dir / f"chunk_{chunk_id:05d}.pt"
            flush_chunk(chunk_path, payload)
            chunk_meta.append(
                {
                    "layer": layer,
                    "chunk_id": chunk_id,
                    "path": str(chunk_path),
                    "rows": int(delta_arr.shape[0]),
                    "d_model": int(delta_arr.shape[1]),
                    "unit": pool_unit,
                }
            )
        current = {layer: {"delta": [], "example_idx": [], "position": []} for layer in layers}
        current_example_count = 0
        chunk_id += 1

    if args.max_batch_tokens > 0:
        lengths = [len(ids) for ids in tokenizer([ex["text"] for ex in examples], truncation=True,
                                                   max_length=max_seq_len)["input_ids"]]
        spans, i = [], 0
        while i < len(examples):
            block_end = min(len(examples), ((shard_start + i) // args.chunk_examples + 1) * args.chunk_examples - shard_start)
            j, longest = i, 0
            while j < block_end and j - i < args.batch_size and (j - i + 1) * max(longest, lengths[j]) <= args.max_batch_tokens:
                longest = max(longest, lengths[j]); j += 1
            j = max(j, i + 1)
            spans.append((i, j)); i = j
    else:
        spans = [(st, min(st + args.batch_size, len(examples))) for st in range(0, len(examples), args.batch_size)]
    batch_sizes_used = {}
    for start, stop in spans:
        batch = examples[start:stop]
        batch_sizes_used[len(batch)] = batch_sizes_used.get(len(batch), 0) + 1
        texts = [ex["text"] for ex in batch]
        tok = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        tok = {k: v.to(base_model.device) for k, v in tok.items()}
        # Sequential forwards: never hold two full logits/hidden-state sets at
        # once (8GB-GPU constraint). Harvest NLL + needed layers, then free.
        with torch.no_grad():
            with base_context():
                base_out = base_model(**tok, output_hidden_states=True, return_dict=True)
            base_nll = per_example_nll(base_out.logits, tok["input_ids"], tok["attention_mask"]).cpu().numpy()
            base_hidden = {layer: base_out.hidden_states[layer].detach().clone() for layer in layers}
            del base_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            adapted_out = adapted_model(**tok, output_hidden_states=True, return_dict=True)
            adapted_nll = per_example_nll(adapted_out.logits, tok["input_ids"], tok["attention_mask"]).cpu().numpy()
            adapted_hidden = {layer: adapted_out.hidden_states[layer].detach().clone() for layer in layers}
            del adapted_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        attn = tok["attention_mask"]

        for bi, ex in enumerate(batch):
            example_index = shard_start + start + bi
            scores_rows.append(
                {
                    "example_idx": example_index,
                    "example_id": ex["example_id"],
                    "user_id": ex["user_id"],
                    "day_index": ex["day_index"],
                    "split": ex["split"],
                    "y": ex["y"],
                    "n_sessions_total": ex.get("n_sessions_total", ex.get("n_events", 0)),
                    "n_sessions_kept": ex.get("n_sessions_kept", ex.get("n_events", 0)),
                    "base_nll": float(base_nll[bi]),
                    "adapted_nll": float(adapted_nll[bi]),
                    "delta_nll": float(adapted_nll[bi] - base_nll[bi]),
                    "n_tokens": int(attn[bi].sum().item()),
                }
            )

        for layer in layers:
            base_h = base_hidden[layer]
            adapted_h = adapted_hidden[layer]
            delta = (adapted_h - base_h).float()
            if pool_unit == "mean":
                pooled = mean_pool(delta, attn).cpu().numpy()
                current[layer]["delta"].append(pooled)
                current[layer]["example_idx"].append(np.arange(shard_start + start, shard_start + start + len(batch), dtype=np.int64))
            else:
                for bi in range(delta.shape[0]):
                    valid = attn[bi].bool()
                    vecs = delta[bi, valid].cpu().numpy()
                    pos = np.arange(vecs.shape[0], dtype=np.int32)
                    current[layer]["delta"].append(vecs)
                    current[layer]["example_idx"].append(np.full((vecs.shape[0],), shard_start + start + bi, dtype=np.int64))
                    current[layer]["position"].append(pos)

        del base_hidden, adapted_hidden
        current_example_count += len(batch)
        if current_example_count >= args.chunk_examples:
            maybe_flush()

    maybe_flush()

    scores_df = pd.DataFrame(scores_rows)
    scores_df.to_parquet(out_dir / "example_scores.parquet", index=False)
    scores_df.to_csv(out_dir / "example_scores.csv", index=False)
    chunk_meta_df = pd.DataFrame(chunk_meta)
    chunk_meta_df.to_csv(out_dir / "chunk_manifest.csv", index=False)
    summary = {
        "split": args.split,
        "n_examples": int(len(scores_df)),
        "layers": layers,
        "pool_unit": pool_unit,
        "chunk_examples": int(args.chunk_examples),
        "adapter_dir": str(args.adapter_dir),
        "model_name_or_path": model_name,
        "base_via_disable_adapter": bool(args.base_via_disable_adapter),
        "start_index": shard_start,
        "batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "batch_sizes_used": {str(k): v for k, v in sorted(batch_sizes_used.items())},
    }
    dump_json(out_dir / "extract_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
