#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from remote_common import ensure_dir, load_yaml


FIELDNAMES = [
    "model",
    "data_dir",
    "micro_batch_size",
    "sample_mode",
    "max_train_examples",
    "warmup_steps",
    "bench_steps",
    "completed_warmup_steps",
    "completed_steps",
    "gradient_checkpointing",
    "attn_impl",
    "status",
    "elapsed_sec",
    "examples_per_sec",
    "tokens_per_sec",
    "avg_tokens_per_example",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "device_total_gib",
    "device_free_after_gib",
    "error",
]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_rows(out_dir: Path, rows: list[dict[str, Any]], target_gib: float, tolerance_gib: float) -> None:
    rows = sorted(rows, key=lambda r: int(r.get("micro_batch_size") or 0))
    csv_path = out_dir / "qwen3_8b_r42_throughput_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "qwen3_8b_r42_throughput_benchmark.json", rows)

    ok_rows = [
        r
        for r in rows
        if r.get("status") == "ok"
        and int(r.get("completed_steps") or 0) > 0
        and r.get("peak_reserved_gib") is not None
    ]
    within_target = [
        r for r in ok_rows if float(r["peak_reserved_gib"]) <= float(target_gib + tolerance_gib)
    ]
    if within_target:
        rec = max(within_target, key=lambda r: float(r["peak_reserved_gib"]))
        reason = f"highest OK peak_reserved_gib <= target+tolerance ({target_gib}+{tolerance_gib})"
    elif ok_rows:
        rec = min(ok_rows, key=lambda r: abs(float(r["peak_reserved_gib"]) - float(target_gib)))
        reason = "closest OK peak_reserved_gib to target; all OK rows were above target+tolerance"
    else:
        rec = None
        reason = "no OK microbatch candidate completed"

    recommendation = {
        "target_gib": float(target_gib),
        "tolerance_gib": float(tolerance_gib),
        "recommended_micro_batch_size": None if rec is None else int(rec["micro_batch_size"]),
        "recommended_peak_reserved_gib": None if rec is None else float(rec["peak_reserved_gib"]),
        "recommended_tokens_per_sec": None if rec is None else float(rec["tokens_per_sec"]),
        "reason": reason,
        "rows_csv": str(csv_path),
    }
    write_json(out_dir / "recommendation.json", recommendation)
    print(json.dumps({"out_dir": str(out_dir), "recommendation": recommendation, "rows": rows}, indent=2))


def run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    row_dir = ensure_dir(out_dir / "rows")
    rows: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()

    for micro_bs in parse_int_list(args.micro_batches):
        row_json = row_dir / f"micro_bs_{micro_bs}.json"
        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(args.config),
            "--data-dir",
            str(args.data_dir),
            "--out-dir",
            str(out_dir),
            "--micro-batch-size",
            str(micro_bs),
            "--row-json",
            str(row_json),
            "--max-train-examples",
            str(args.max_train_examples),
            "--warmup-steps",
            str(args.warmup_steps),
            "--bench-steps",
            str(args.bench_steps),
            "--sample-mode",
            args.sample_mode,
            "--attn-impl",
            args.attn_impl,
            "--dataloader-workers",
            str(args.dataloader_workers),
            "--seed",
            str(args.seed),
        ]
        cmd.append("--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing")
        print(f"[throughput_bench] running micro_batch_size={micro_bs}", flush=True)
        result = subprocess.run(cmd, text=True)
        if row_json.exists():
            rows.append(json.loads(row_json.read_text(encoding="utf-8")))
        else:
            rows.append(
                {
                    "model": "",
                    "data_dir": str(args.data_dir),
                    "micro_batch_size": micro_bs,
                    "sample_mode": args.sample_mode,
                    "max_train_examples": int(args.max_train_examples),
                    "warmup_steps": int(args.warmup_steps),
                    "bench_steps": int(args.bench_steps),
                    "completed_warmup_steps": 0,
                    "completed_steps": 0,
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                    "attn_impl": args.attn_impl,
                    "status": "launcher_error",
                    "elapsed_sec": 0.0,
                    "examples_per_sec": 0.0,
                    "tokens_per_sec": 0.0,
                    "avg_tokens_per_example": 0.0,
                    "peak_allocated_gib": None,
                    "peak_reserved_gib": None,
                    "device_total_gib": None,
                    "device_free_after_gib": None,
                    "error": f"worker exited rc={result.returncode} without row output",
                }
            )

    write_rows(out_dir, rows, target_gib=args.target_gib, tolerance_gib=args.target_tolerance_gib)


def run_worker(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)
    row: dict[str, Any] = {
        "model": str(cfg["model_name_or_path"]),
        "data_dir": str(args.data_dir),
        "micro_batch_size": int(args.micro_batch_size),
        "sample_mode": args.sample_mode,
        "max_train_examples": int(args.max_train_examples),
        "warmup_steps": int(args.warmup_steps),
        "bench_steps": int(args.bench_steps),
        "completed_warmup_steps": 0,
        "completed_steps": 0,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "attn_impl": args.attn_impl,
        "status": "ok",
        "elapsed_sec": 0.0,
        "examples_per_sec": 0.0,
        "tokens_per_sec": 0.0,
        "avg_tokens_per_example": 0.0,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
        "device_total_gib": None,
        "device_free_after_gib": None,
        "error": "",
    }
    torch = None
    start_time = time.time()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            set_seed,
        )
        import torch as torch_mod

        torch = torch_mod
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for throughput benchmarking")

        set_seed(args.seed)
        device = torch.device("cuda:0")
        model_name = cfg["model_name_or_path"]

        train_path = args.data_dir / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train JSONL: {train_path}")
        ds = load_dataset("json", data_files={"train": str(train_path)})["train"]
        if args.max_train_examples > 0:
            ds = ds.select(range(min(args.max_train_examples, len(ds))))

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_seq_len = int(cfg["training"]["max_seq_len"])

        def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
            tok = tokenizer(batch["text"], truncation=True, max_length=max_seq_len)
            tok["length"] = [len(ids) for ids in tok["input_ids"]]
            return tok

        tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        if args.sample_mode == "longest":
            tokenized = tokenized.sort("length", reverse=True)
        elif args.sample_mode == "random":
            tokenized = tokenized.shuffle(seed=args.seed)
        elif args.sample_mode != "first":
            raise ValueError(f"unsupported sample_mode={args.sample_mode}")
        tokenized = tokenized.remove_columns(["length"])

        needed = int(args.micro_batch_size) * max(1, int(args.warmup_steps) + int(args.bench_steps))
        if len(tokenized) > needed:
            tokenized = tokenized.select(range(needed))

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        loader = DataLoader(
            tokenized,
            batch_size=int(args.micro_batch_size),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(args.dataloader_workers),
            pin_memory=True,
        )

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=bool(cfg["quantization"]["load_in_4bit"]),
            bnb_4bit_quant_type=str(cfg["quantization"]["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=getattr(torch, str(cfg["quantization"]["bnb_4bit_compute_dtype"])),
            bnb_4bit_use_double_quant=bool(cfg["quantization"]["bnb_4bit_use_double_quant"]),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16 if bool(cfg["training"].get("bf16", True)) else torch.float16,
            device_map={"": 0},
            attn_implementation=args.attn_impl,
        )
        model.config.use_cache = False
        prep_kwargs = {"use_gradient_checkpointing": bool(args.gradient_checkpointing)}
        if args.gradient_checkpointing:
            prep_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        model = prepare_model_for_kbit_training(model, **prep_kwargs)
        lora_cfg = LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["alpha"]),
            lora_dropout=float(cfg["lora"]["dropout"]),
            target_modules=list(cfg["lora"]["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["training"]["learning_rate"]),
        )

        data_iter = iter(loader)
        for _ in range(int(args.warmup_steps)):
            batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            row["completed_warmup_steps"] += 1
            del batch, loss
            torch.cuda.synchronize(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        bench_start = time.time()
        total_examples = 0
        total_tokens = 0

        for _ in range(int(args.bench_steps)):
            batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            total_examples += int(batch["input_ids"].shape[0])
            total_tokens += int(batch.get("attention_mask", torch.ones_like(batch["input_ids"])).sum().item())
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            row["completed_steps"] += 1
            del batch, loss
            torch.cuda.synchronize(device)

        elapsed = max(time.time() - bench_start, 1e-9)
        row["elapsed_sec"] = round(elapsed, 3)
        row["examples_per_sec"] = round(total_examples / elapsed, 3)
        row["tokens_per_sec"] = round(total_tokens / elapsed, 3)
        row["avg_tokens_per_example"] = round(total_tokens / max(total_examples, 1), 3)

    except Exception as exc:
        msg = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        row["status"] = "oom" if "out of memory" in str(exc).lower() else "error"
        row["error"] = msg
        row["elapsed_sec"] = round(time.time() - start_time, 3)
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    finally:
        if torch is not None and torch.cuda.is_available():
            try:
                device = torch.device("cuda:0")
                torch.cuda.synchronize(device)
            except Exception:
                pass
            try:
                row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
                row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / (1024**3), 3)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                row["device_total_gib"] = round(total_bytes / (1024**3), 3)
                row["device_free_after_gib"] = round(free_bytes / (1024**3), 3)
            except Exception:
                pass
        if args.row_json:
            write_json(args.row_json, row)
        else:
            write_json(out_dir / f"micro_bs_{args.micro_batch_size}.json", row)
        print(json.dumps(row, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--micro-batches", type=str, default="12,13,14")
    ap.add_argument("--micro-batch-size", type=int, default=0)
    ap.add_argument("--row-json", type=Path, default=None)
    ap.add_argument("--max-train-examples", type=int, default=4096)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--bench-steps", type=int, default=8)
    ap.add_argument("--sample-mode", choices=("longest", "first", "random"), default="longest")
    ap.add_argument("--attn-impl", type=str, default="sdpa")
    ap.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--dataloader-workers", type=int, default=4)
    ap.add_argument("--target-gib", type=float, default=70.0)
    ap.add_argument("--target-tolerance-gib", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.micro_batch_size > 0:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
