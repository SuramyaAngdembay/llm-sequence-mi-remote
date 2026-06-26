#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from remote_common import ensure_dir, load_yaml


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--micro-batches", type=str, default="4,8,12,16")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--attn-impl", type=str, default="sdpa")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRAM benchmarking.")

    set_seed(args.seed)
    device = torch.device("cuda:0")
    model_name = cfg["model_name_or_path"]
    model_cfg = AutoConfig.from_pretrained(model_name)
    vocab_size = int(getattr(model_cfg, "vocab_size"))

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg["training"]["learning_rate"]))

    rows: list[dict[str, object]] = []
    for micro_bs in parse_int_list(args.micro_batches):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        status = "ok"
        error = ""
        completed_steps = 0
        try:
            for step in range(int(args.steps)):
                # Full-length synthetic stress batch. Real training may use less memory
                # when dynamic padding sees shorter session sequences.
                input_ids = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=(micro_bs, int(args.seq_len)),
                    device=device,
                    dtype=torch.long,
                )
                attention_mask = torch.ones_like(input_ids, device=device)
                labels = input_ids.clone()
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                loss.backward()
                optimizer.step()
                completed_steps = step + 1
                del input_ids, attention_mask, labels, loss
                torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as exc:
            status = "oom"
            error = str(exc).splitlines()[0]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            status = "runtime_error"
            error = str(exc).splitlines()[0]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        torch.cuda.synchronize(device)
        peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        rows.append(
            {
                "model": model_name,
                "micro_batch_size": micro_bs,
                "seq_len": int(args.seq_len),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "attn_impl": args.attn_impl,
                "status": status,
                "completed_steps": completed_steps,
                "peak_allocated_gb": round(peak_alloc_gb, 3),
                "peak_reserved_gb": round(peak_reserved_gb, 3),
                "device_total_gb": round(total_bytes / (1024**3), 3),
                "device_free_after_gb": round(free_bytes / (1024**3), 3),
                "elapsed_sec": round(time.time() - start, 3),
                "error": error,
            }
        )

    csv_path = out_dir / "qwen3_8b_qlora_vram_benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "qwen3_8b_qlora_vram_benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(json.dumps({"out_dir": str(out_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
