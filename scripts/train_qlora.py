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
    ap.add_argument("--save-steps", type=int, default=0, help="override training.save_steps (0=use config)")
    ap.add_argument("--eval-steps", type=int, default=0, help="override eval_steps (0=use save_steps)")
    ap.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="steps")
    ap.add_argument(
        "--target-loss-mask",
        choices=("none", "profile"),
        default="none",
        help="'profile' sets labels to -100 at DAY/PSY target tokens: the adapter is not "
             "trained to PREDICT profile tokens, while the profile stays in the input context "
             "(behavioural-token losses can still reward using it). 'none' = unchanged recipe.",
    )
    ap.add_argument(
        "--loss-denominator",
        choices=("scored_targets", "all_targets"),
        default="scored_targets",
        help="scored_targets: mean over non-ignored targets (HF default; with --target-loss-mask "
             "profile this up-weights each behavioural token by 1/(1-p)). all_targets: divide by "
             "the count of non-pad targets, so a retained token keeps exactly its unmasked weight "
             "and masking only drops the profile terms.",
    )
    ap.add_argument("--resume-from-checkpoint", type=Path, default=None)
    ap.add_argument("--ignore-data-skip", action="store_true")
    ap.add_argument(
        "--skip-rng-state-resume",
        action="store_true",
        help="resume checkpoint weights/optimizer/scheduler without restoring rng_state_*.pth",
    )
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
            DataCollatorForSeq2Seq,
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

    mask_profile_targets = args.target_loss_mask == "profile"
    fixed_denominator = args.loss_denominator == "all_targets"

    if not mask_profile_targets and not fixed_denominator:
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)
    else:
        # Profile *targets* are identified by the class of the token being
        # predicted; `input_ids` (the context) is never touched. Classes come
        # from the shared decomposition library, by line PREFIX, so `no_psy` /
        # `no_profile` serializations cannot be mislabelled by line index.
        from token_class_decomposition import (
            CERT_PROFILE_CLASSES,
            cert_class_spans,
            classify_tokens,
        )

        profile_classes = set(CERT_PROFILE_CLASSES)

        def tokenize(batch):
            tok = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_seq_len,
                return_offsets_mapping=True,
            )
            out = {k: v for k, v in tok.items() if k != "offset_mapping"}
            masks, n_masked = [], []
            for text, ids, offs in zip(batch["text"], tok["input_ids"], tok["offset_mapping"]):
                if mask_profile_targets:
                    classes, _ = classify_tokens([tuple(o) for o in offs], cert_class_spans(text))
                    m = [1 if c in profile_classes else 0 for c in classes]
                else:
                    m = [0] * len(ids)
                masks.append(m)
                n_masked.append(sum(m))
            if fixed_denominator:
                # labels stay unmasked (only padding becomes -100), so the
                # trainer's `num_items_in_batch` counts EVERY non-pad target.
                # The profile mask is applied inside compute_loss instead.
                out["profile_mask"] = masks
            else:
                # HF-default semantics: -100 at profile targets, so the trainer
                # normalizes over behaviour targets only.
                out["labels"] = [
                    [(-100 if mm else i) for i, mm in zip(ids, m)]
                    for ids, m in zip(tok["input_ids"], masks)
                ]
            out["n_masked_targets"] = n_masked
            return out

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    masked_target_frac = None
    if mask_profile_targets or fixed_denominator:
        # bounded probe: materializing every input_ids list would cost GBs
        n_probe = min(5000, len(tokenized["train"]))
        probe = tokenized["train"].select(range(n_probe))
        n_masked = sum(probe["n_masked_targets"])
        n_tok = sum(len(x) for x in probe["input_ids"])
        masked_target_frac = float(n_masked) / max(n_tok, 1)
        tokenized = tokenized.remove_columns(["n_masked_targets"])

    if mask_profile_targets and not fixed_denominator:
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, padding=True, return_tensors="pt"
        )
    elif fixed_denominator:
        _lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        class ProfileMaskCollator:
            """Pads `profile_mask` (with 0) alongside the usual LM batch."""

            def __init__(self, base):
                self.base = base

            def __call__(self, features):
                import torch as _t

                masks = [list(f.pop("profile_mask")) for f in features]
                batch = self.base(features)
                width = batch["input_ids"].shape[1]
                pm = _t.zeros((len(masks), width), dtype=_t.long)
                for i, m in enumerate(masks):
                    k = min(len(m), width)
                    if k:
                        pm[i, :k] = _t.tensor(m[:k], dtype=_t.long)
                batch["profile_mask"] = pm
                return batch

        collator = ProfileMaskCollator(_lm_collator)
    else:
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
    save_steps = args.save_steps or int(cfg["training"]["save_steps"])
    eval_steps = args.eval_steps or save_steps

    training_arg_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": micro_bs,
        "per_device_eval_batch_size": micro_bs,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": float(cfg["training"]["num_train_epochs"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "lr_scheduler_type": str(cfg["training"]["lr_scheduler_type"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "logging_steps": int(cfg["training"]["logging_steps"]),
        "save_steps": save_steps,
        "eval_steps": eval_steps,
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
    _wr = float(cfg["training"]["warmup_ratio"])
    if "warmup_ratio" in training_args_params:
        training_arg_kwargs["warmup_ratio"] = _wr
    else:
        import math as _math
        _n = getattr(args, "max_train_examples", None)
        _epochs = float(cfg["training"]["num_train_epochs"])
        if _n:
            _total = _math.ceil(int(_n) / (micro_bs * grad_accum)) * _epochs
            training_arg_kwargs["warmup_steps"] = max(1, round(_wr * _total))
        else:
            training_arg_kwargs["warmup_steps"] = 500
    eval_strategy_key = "eval_strategy" if "eval_strategy" in training_args_params else "evaluation_strategy"
    training_arg_kwargs[eval_strategy_key] = args.eval_strategy
    if "ignore_data_skip" in training_args_params:
        training_arg_kwargs["ignore_data_skip"] = bool(args.ignore_data_skip)
    training_args = TrainingArguments(**training_arg_kwargs)

    trainer_cls = Trainer

    if args.loss_denominator == "all_targets":
        import torch.nn.functional as _F

        class FixedDenominatorMixin:
            """Sum of retained target losses over the count of NON-PAD targets.

            `labels` here are unmasked (only padding is -100), so the trainer's
            `num_items_in_batch` — which it computes over the whole gradient
            accumulation window — counts every non-pad target. Dividing by it
            keeps each retained (behavioural) token at exactly the weight it
            carries in the unmasked recipe: masking then only removes the
            profile terms rather than also rescaling the survivors by 1/(1-p).

            Normalizing per micro-batch instead would be wrong: transformers
            skips its own `/ gradient_accumulation_steps` whenever
            `num_items_in_batch` is supplied, so a micro-batch mean comes out
            `grad_accum` times too large (measured: exactly 4.0x at
            grad_accum=4, job 20810414).

            With nothing masked this is arithmetically identical to the HF
            default, which is what the pre-training gate checks.
            """

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
                labels = inputs.pop("labels")
                pmask = inputs.pop("profile_mask", None)
                outputs = model(**inputs)
                logits = outputs.logits[..., :-1, :].float()
                target = labels[..., 1:]
                tok_loss = _F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(target.shape)
                keep = (target != -100).to(tok_loss.dtype)
                if pmask is not None:
                    keep = keep * (1.0 - pmask[..., 1:].to(tok_loss.dtype))
                loss_sum = (tok_loss * keep).sum()
                if num_items_in_batch is not None:
                    denom = float(num_items_in_batch)
                else:
                    denom = float((target != -100).sum().clamp_min(1))
                loss = loss_sum / max(denom, 1.0)
                return (loss, outputs) if return_outputs else loss

        class FixedDenominatorTrainer(FixedDenominatorMixin, Trainer):
            pass

        trainer_cls = FixedDenominatorTrainer

    if args.skip_rng_state_resume:
        _base = trainer_cls

        class RngSkipTrainer(_base):
            def _load_rng_state(self, checkpoint):
                if self.is_world_process_zero():
                    print(f"skipping_rng_state_restore=1 checkpoint={checkpoint}")

        trainer_cls = RngSkipTrainer

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )
    # transformers v5 renamed Trainer's `tokenizer` kwarg to `processing_class`.
    if "processing_class" in inspect.signature(trainer_cls.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = trainer_cls(**trainer_kwargs)
    resume_from_checkpoint = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
            "target_loss_mask": args.target_loss_mask,
            "loss_denominator": args.loss_denominator,
            "masked_target_frac": masked_target_frac,
            "gradient_checkpointing": grad_ckpt,
            "save_steps": save_steps,
            "eval_strategy": args.eval_strategy,
            "eval_steps": eval_steps,
            "resume_from_checkpoint": resume_from_checkpoint,
            "ignore_data_skip": bool(args.ignore_data_skip),
            "skip_rng_state_resume": bool(args.skip_rng_state_resume),
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
