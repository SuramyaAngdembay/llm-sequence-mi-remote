#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

from remote_common import dump_json, ensure_dir, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-train-examples", type=int, default=0)
    ap.add_argument("--max-val-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # --- multi-GPU / utilization overrides (optional; 0/None = fall back to config) ---
    ap.add_argument("--micro-batch-size", type=int, default=0, help="override training.micro_batch_size (0=use config)")
    ap.add_argument("--grad-accum", type=int, default=0, help="override gradient_accumulation_steps (0=use config)")
    ap.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", default=None)
    ap.add_argument("--dataloader-workers", type=int, default=4)
    ap.add_argument("--attn-impl", type=str, default="sdpa")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies for QLoRA training. Install transformers, datasets, peft, bitsandbytes."
        ) from exc

    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)

    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    data_files = {"train": str(train_path), "validation": str(val_path)}
    ds = load_dataset("json", data_files=data_files)
    if args.max_train_examples > 0:
        ds["train"] = ds["train"].select(range(min(args.max_train_examples, len(ds["train"]))))
    if args.max_val_examples > 0:
        ds["validation"] = ds["validation"].select(range(min(args.max_val_examples, len(ds["validation"]))))

    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_seq_len = int(cfg["training"]["max_seq_len"])

    def tokenize(batch):
        tok = tokenizer(batch["text"], truncation=True, max_length=max_seq_len)
        return tok

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(cfg["quantization"]["load_in_4bit"]),
        bnb_4bit_quant_type=str(cfg["quantization"]["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(__import__("torch"), str(cfg["quantization"]["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(cfg["quantization"]["bnb_4bit_use_double_quant"]),
    )

    import torch

    # Distributed (torchrun) aware placement: under DDP each rank loads the full
    # 4-bit model onto its OWN gpu. Single-process keeps device_map="auto".
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if world_size > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if bool(cfg["training"].get("bf16", True)) else torch.float16,
        device_map=device_map,
        attn_implementation=args.attn_impl,
    )
    model.config.use_cache = False
    grad_ckpt = (
        bool(cfg["training"].get("gradient_checkpointing", True))
        if args.gradient_checkpointing is None
        else bool(args.gradient_checkpointing)
    )
    prep_kwargs = {"use_gradient_checkpointing": grad_ckpt}
    if grad_ckpt:
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

    micro_bs = args.micro_batch_size or int(cfg["training"]["micro_batch_size"])
    grad_accum = args.grad_accum or int(cfg["training"]["gradient_accumulation_steps"])

    training_arg_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": micro_bs,
        "per_device_eval_batch_size": micro_bs,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": float(cfg["training"]["num_train_epochs"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "lr_scheduler_type": str(cfg["training"]["lr_scheduler_type"]),
        "warmup_ratio": float(cfg["training"]["warmup_ratio"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "logging_steps": int(cfg["training"]["logging_steps"]),
        "save_steps": int(cfg["training"]["save_steps"]),
        "eval_steps": int(cfg["training"]["save_steps"]),
        "save_strategy": "steps",
        "bf16": bool(cfg["training"].get("bf16", True)),
        "gradient_checkpointing": grad_ckpt,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": args.dataloader_workers,
        "report_to": [],
        "remove_unused_columns": False,
        "logging_first_step": True,
    }
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_key = "eval_strategy" if "eval_strategy" in training_args_params else "evaluation_strategy"
    training_arg_kwargs[eval_strategy_key] = "steps"
    training_args = TrainingArguments(**training_arg_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "adapter"))
        tokenizer.save_pretrained(str(out_dir / "adapter"))

        summary = {
            "config": str(args.config),
            "data_dir": str(args.data_dir),
            "output_dir": str(out_dir),
            "model_name_or_path": model_name,
            "n_train": int(len(tokenized["train"])),
            "n_validation": int(len(tokenized["validation"])),
            "max_seq_len": max_seq_len,
            "seed": int(args.seed),
            "world_size": world_size,
            "micro_batch_size": micro_bs,
            "grad_accum": grad_accum,
            "effective_batch": micro_bs * grad_accum * world_size,
            "gradient_checkpointing": grad_ckpt,
            "env": {
                "HF_HOME": os.environ.get("HF_HOME", ""),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
            },
        }
        dump_json(out_dir / "train_summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
