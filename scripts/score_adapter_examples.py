#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from remote_common import dump_json, ensure_dir, load_yaml, read_jsonl


def clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def per_example_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    loss_batch_size: int = 0,
) -> torch.Tensor:
    """Compute per-example NLL while bounding the fp32 cross-entropy workspace."""
    batch_size = int(logits.shape[0])
    chunk_size = batch_size if int(loss_batch_size) <= 0 else min(batch_size, int(loss_batch_size))
    out: List[torch.Tensor] = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        shift_logits = logits[start:end, :-1, :].float().contiguous()
        shift_labels = input_ids[start:end, 1:].contiguous()
        shift_mask = attention_mask[start:end, 1:].contiguous().float()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        denom = shift_mask.sum(dim=1).clamp_min(1.0)
        out.append((token_loss * shift_mask).sum(dim=1) / denom)
        del shift_logits, shift_labels, shift_mask, token_loss, denom
        if logits.device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


def batched(rows: Iterable[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_allowed_users(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        arr = json.loads(text)
        return {str(x) for x in arr}
    return {line.strip() for line in text.splitlines() if line.strip()}


def flush_rows(
    writer: Optional[pq.ParquetWriter],
    parquet_path: Path,
    rows: List[Dict[str, Any]],
) -> pq.ParquetWriter:
    if not rows:
        if writer is None:
            raise ValueError("cannot flush empty buffer before writer init")
        return writer
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--jsonl-path", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--loss-batch-size", type=int, default=4, help="Chunk size for fp32 cross-entropy workspace; <=0 disables chunking.")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--flush-rows", type=int, default=4096)
    ap.add_argument("--users-file", type=Path, default=None, help="Optional newline-separated or JSON-array allowlist of user_ids.")
    ap.add_argument("--log-every", type=int, default=4096)
    ap.add_argument("--separate-base-model", action="store_true", help="Load a second base model instead of disabling the adapter on the PEFT model.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies for example scoring. Install transformers, peft, bitsandbytes."
        ) from exc

    allowed_users = load_allowed_users(args.users_file)
    out_dir = ensure_dir(args.output_dir)
    parquet_path = out_dir / "example_scores.parquet"

    model_name = cfg["model_name_or_path"]
    max_seq_len = int(cfg["training"]["max_seq_len"])
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
    adapted_model.config.use_cache = False
    if hasattr(adapted_model, "base_model") and hasattr(adapted_model.base_model, "config"):
        adapted_model.base_model.config.use_cache = False

    use_separate_base_model = bool(args.separate_base_model)
    if not use_separate_base_model and not hasattr(adapted_model, "disable_adapter"):
        print("PeftModel.disable_adapter() is unavailable; loading a separate base model.", flush=True)
        use_separate_base_model = True

    base_model = None
    if use_separate_base_model:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
        base_model.eval()
        base_model.config.use_cache = False

    model_device = adapted_model.device if hasattr(adapted_model, "device") else next(adapted_model.parameters()).device

    def score_nll(model: torch.nn.Module, tok: Dict[str, torch.Tensor]) -> np.ndarray:
        with torch.inference_mode():
            out = model(**tok, return_dict=True, use_cache=False)
        logits = out.logits
        nll_t = per_example_nll(
            logits,
            tok["input_ids"],
            tok["attention_mask"],
            loss_batch_size=args.loss_batch_size,
        )
        nll = nll_t.detach().cpu().numpy()
        del out, logits, nll_t
        clear_cuda(model_device)
        return nll

    def example_iter() -> Iterator[Dict[str, Any]]:
        n_seen = 0
        for ex in read_jsonl(args.jsonl_path):
            if allowed_users is not None and str(ex["user_id"]) not in allowed_users:
                continue
            yield ex
            n_seen += 1
            if args.max_examples > 0 and n_seen >= args.max_examples:
                break

    writer: Optional[pq.ParquetWriter] = None
    buffer: List[Dict[str, Any]] = []
    n_scored = 0
    n_tokens_total = 0
    next_log_at = int(args.log_every) if int(args.log_every) > 0 else 0
    for batch in batched(example_iter(), args.batch_size):
        texts = [ex["text"] for ex in batch]
        tok = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        tok = {k: v.to(model_device) for k, v in tok.items()}
        base_context = nullcontext() if use_separate_base_model else adapted_model.disable_adapter()
        with base_context:
            base_nll = score_nll(base_model if base_model is not None else adapted_model, tok)
        adapted_nll = score_nll(adapted_model, tok)
        n_tokens = tok["attention_mask"].sum(dim=1).cpu().numpy()

        for bi, ex in enumerate(batch):
            row = {
                "example_idx": n_scored + bi,
                "example_id": ex["example_id"],
                "user_id": ex["user_id"],
                "day_index": ex["day_index"],
                "split": ex["split"],
                "y": ex["y"],
                "n_sessions_total": ex["n_sessions_total"],
                "n_sessions_kept": ex["n_sessions_kept"],
                "base_nll": float(base_nll[bi]),
                "adapted_nll": float(adapted_nll[bi]),
                "delta_nll": float(adapted_nll[bi] - base_nll[bi]),
                "n_tokens": int(n_tokens[bi]),
            }
            buffer.append(row)
            n_tokens_total += int(n_tokens[bi])

        n_scored += len(batch)
        if len(buffer) >= args.flush_rows:
            writer = flush_rows(writer, parquet_path, buffer)
            buffer = []
        if next_log_at > 0 and n_scored >= next_log_at:
            print(f"scored_examples={n_scored}", flush=True)
            while next_log_at <= n_scored:
                next_log_at += int(args.log_every)
        del tok, base_nll, adapted_nll, n_tokens
        clear_cuda(model_device)

    if n_scored == 0:
        raise ValueError(f"no examples were scored from {args.jsonl_path}")
    if buffer:
        writer = flush_rows(writer, parquet_path, buffer)
    if writer is not None:
        writer.close()

    summary = {
        "config": str(args.config),
        "jsonl_path": str(args.jsonl_path),
        "adapter_dir": str(args.adapter_dir),
        "output_dir": str(out_dir),
        "model_name_or_path": model_name,
        "n_examples": int(n_scored),
        "batch_size": int(args.batch_size),
        "loss_batch_size": int(args.loss_batch_size),
        "separate_base_model": bool(use_separate_base_model),
        "flush_rows": int(args.flush_rows),
        "max_examples": int(args.max_examples),
        "users_file": str(args.users_file) if args.users_file else "",
        "n_allowed_users": int(len(allowed_users)) if allowed_users is not None else None,
        "mean_tokens": float(n_tokens_total / max(n_scored, 1)),
    }
    dump_json(out_dir / "score_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
