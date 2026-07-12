#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from remote_common import dump_json, ensure_dir, load_yaml, read_jsonl


def per_example_nll(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())
    denom = shift_mask.sum(dim=1).clamp_min(1.0)
    return (token_loss * shift_mask).sum(dim=1) / denom


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
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--flush-rows", type=int, default=4096)
    ap.add_argument("--users-file", type=Path, default=None, help="Optional newline-separated or JSON-array allowlist of user_ids.")
    ap.add_argument("--log-every", type=int, default=4096)
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
    base_model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    adapted_backbone = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    adapted_model = PeftModel.from_pretrained(adapted_backbone, args.adapter_dir)
    base_model.eval()
    adapted_model.eval()

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
    for batch in batched(example_iter(), args.batch_size):
        texts = [ex["text"] for ex in batch]
        tok = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        tok = {k: v.to(base_model.device) for k, v in tok.items()}
        with torch.no_grad():
            base_out = base_model(**tok, return_dict=True)
            adapted_out = adapted_model(**tok, return_dict=True)
        base_nll = per_example_nll(base_out.logits.float(), tok["input_ids"], tok["attention_mask"]).cpu().numpy()
        adapted_nll = per_example_nll(adapted_out.logits.float(), tok["input_ids"], tok["attention_mask"]).cpu().numpy()
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
        if args.log_every > 0 and n_scored % args.log_every == 0:
            print(f"scored_examples={n_scored}", flush=True)

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
