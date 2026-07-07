#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval_token_delta_sae_causal import get_layer_module, per_example_nll
from remote_common import ensure_dir, load_yaml, read_jsonl


FIELDNAMES = [
    "model",
    "adapter_dir",
    "data_dir",
    "layer",
    "batch_size",
    "sample_mode",
    "max_examples",
    "warmup_steps",
    "bench_steps",
    "completed_warmup_steps",
    "completed_steps",
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
    "process_maxrss_gib",
    "error",
]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def maxrss_gib() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def write_rows(out_dir: Path, rows: list[dict[str, Any]], target_gib: float, target_fraction: float, tolerance_gib: float) -> None:
    rows = sorted(rows, key=lambda r: int(r.get("batch_size") or 0))
    csv_path = out_dir / "token_causal_vram_probe.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "token_causal_vram_probe.json", rows)

    ok_rows = [
        r
        for r in rows
        if r.get("status") == "ok"
        and int(r.get("completed_steps") or 0) > 0
        and r.get("peak_reserved_gib") is not None
    ]
    observed_total = next(
        (float(r["device_total_gib"]) for r in rows if r.get("device_total_gib") is not None),
        None,
    )
    effective_target = float(target_gib)
    if effective_target <= 0.0 and observed_total is not None:
        effective_target = observed_total * float(target_fraction)

    if ok_rows and effective_target > 0.0:
        within_target = [
            r for r in ok_rows if float(r["peak_reserved_gib"]) <= effective_target + float(tolerance_gib)
        ]
        if within_target:
            rec = max(within_target, key=lambda r: float(r["peak_reserved_gib"]))
            reason = "highest OK peak_reserved_gib within target+tolerance"
        else:
            rec = min(ok_rows, key=lambda r: abs(float(r["peak_reserved_gib"]) - effective_target))
            reason = "closest OK peak_reserved_gib to target"
    elif ok_rows:
        rec = max(ok_rows, key=lambda r: float(r["peak_reserved_gib"]))
        reason = "highest OK peak_reserved_gib; no target available"
    else:
        rec = None
        reason = "no OK batch candidate completed"

    recommendation = {
        "target_gib": None if effective_target <= 0.0 else round(effective_target, 3),
        "target_fraction": float(target_fraction),
        "tolerance_gib": float(tolerance_gib),
        "recommended_batch_size": None if rec is None else int(rec["batch_size"]),
        "recommended_peak_reserved_gib": None if rec is None else float(rec["peak_reserved_gib"]),
        "recommended_tokens_per_sec": None if rec is None else float(rec["tokens_per_sec"]),
        "rows_csv": str(csv_path),
        "reason": reason,
    }
    write_json(out_dir / "recommendation.json", recommendation)
    print(json.dumps({"out_dir": str(out_dir), "recommendation": recommendation, "rows": rows}, indent=2))


def run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    row_dir = ensure_dir(out_dir / "rows")
    rows: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()

    for batch_size in parse_int_list(args.batch_sizes):
        row_json = row_dir / f"batch_size_{batch_size}.json"
        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(args.config),
            "--adapter-dir",
            str(args.adapter_dir),
            "--data-dir",
            str(args.data_dir),
            "--out-dir",
            str(out_dir),
            "--batch-size",
            str(batch_size),
            "--row-json",
            str(row_json),
            "--layer",
            str(args.layer),
            "--max-examples",
            str(args.max_examples),
            "--warmup-steps",
            str(args.warmup_steps),
            "--bench-steps",
            str(args.bench_steps),
            "--sample-mode",
            args.sample_mode,
            "--attn-impl",
            args.attn_impl,
            "--seed",
            str(args.seed),
        ]
        print(f"[token_causal_vram] running batch_size={batch_size}", flush=True)
        result = subprocess.run(cmd, text=True)
        if row_json.exists():
            rows.append(json.loads(row_json.read_text(encoding="utf-8")))
        else:
            status = "launcher_error"
            if result.returncode < 0:
                status = f"signal_{abs(result.returncode)}"
            rows.append(
                {
                    "model": "",
                    "adapter_dir": str(args.adapter_dir),
                    "data_dir": str(args.data_dir),
                    "layer": int(args.layer),
                    "batch_size": int(batch_size),
                    "sample_mode": args.sample_mode,
                    "max_examples": int(args.max_examples),
                    "warmup_steps": int(args.warmup_steps),
                    "bench_steps": int(args.bench_steps),
                    "completed_warmup_steps": 0,
                    "completed_steps": 0,
                    "attn_impl": args.attn_impl,
                    "status": status,
                    "elapsed_sec": 0.0,
                    "examples_per_sec": 0.0,
                    "tokens_per_sec": 0.0,
                    "avg_tokens_per_example": 0.0,
                    "peak_allocated_gib": None,
                    "peak_reserved_gib": None,
                    "device_total_gib": None,
                    "device_free_after_gib": None,
                    "process_maxrss_gib": None,
                    "error": f"worker exited rc={result.returncode} without row output",
                }
            )

    write_rows(
        out_dir,
        rows,
        target_gib=float(args.target_gib),
        target_fraction=float(args.target_fraction),
        tolerance_gib=float(args.target_tolerance_gib),
    )


def load_probe_texts(
    data_dir: Path,
    tokenizer: Any,
    *,
    max_seq_len: int,
    max_examples: int,
    sample_mode: str,
    seed: int,
) -> list[str]:
    rows = list(read_jsonl(data_dir / "eval.jsonl"))
    if max_examples > 0 and len(rows) > max_examples:
        rows = rows[:max_examples]
    texts = [str(row["text"]) for row in rows]
    if sample_mode == "first":
        return texts
    if sample_mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(texts)).tolist()
        return [texts[i] for i in order]
    if sample_mode != "longest":
        raise ValueError(f"unsupported sample_mode={sample_mode}")

    lengths: list[int] = []
    for start in range(0, len(texts), 128):
        tok = tokenizer(
            texts[start : start + 128],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        lengths.extend(len(ids) for ids in tok["input_ids"])
    order = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)
    return [texts[i] for i in order]


def take_batch(texts: list[str], start: int, batch_size: int) -> list[str]:
    if not texts:
        raise ValueError("No probe texts loaded")
    out: list[str] = []
    for offset in range(batch_size):
        out.append(texts[(start + offset) % len(texts)])
    return out


def run_worker(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    row: dict[str, Any] = {
        "model": str(cfg["model_name_or_path"]),
        "adapter_dir": str(args.adapter_dir),
        "data_dir": str(args.data_dir),
        "layer": int(args.layer),
        "batch_size": int(args.batch_size),
        "sample_mode": args.sample_mode,
        "max_examples": int(args.max_examples),
        "warmup_steps": int(args.warmup_steps),
        "bench_steps": int(args.bench_steps),
        "completed_warmup_steps": 0,
        "completed_steps": 0,
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
        "process_maxrss_gib": None,
        "error": "",
    }
    torch = None
    start_time = time.time()
    total_tokens = 0

    try:
        import torch as torch_mod
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

        torch = torch_mod
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for token causal VRAM benchmarking")

        set_seed(args.seed)
        device = torch.device("cuda:0")
        model_name = cfg["model_name_or_path"]
        max_seq_len = int(cfg["training"]["max_seq_len"])

        tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        texts = load_probe_texts(
            args.data_dir,
            tokenizer,
            max_seq_len=max_seq_len,
            max_examples=int(args.max_examples),
            sample_mode=args.sample_mode,
            seed=int(args.seed),
        )

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=bool(cfg["quantization"]["load_in_4bit"]),
            bnb_4bit_quant_type=str(cfg["quantization"]["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=getattr(torch, str(cfg["quantization"]["bnb_4bit_compute_dtype"])),
            bnb_4bit_use_double_quant=bool(cfg["quantization"]["bnb_4bit_use_double_quant"]),
        )
        model_kwargs: dict[str, Any] = {
            "quantization_config": quant_cfg,
            "torch_dtype": torch.bfloat16 if bool(cfg["training"].get("bf16", True)) else torch.float16,
            "device_map": {"": 0},
        }
        if args.attn_impl and args.attn_impl != "auto":
            model_kwargs["attn_implementation"] = args.attn_impl
        backbone = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = PeftModel.from_pretrained(backbone, args.adapter_dir)
        model.config.use_cache = False
        model.eval()

        layer_module = get_layer_module(model, int(args.layer))
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        if hidden_size <= 0:
            hidden_size = int(backbone.config.hidden_size)

        def run_step(batch_texts: list[str]) -> int:
            tok = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            attn = tok["attention_mask"]
            patch_mats = [
                np.zeros((int(attn[i].sum().item()), hidden_size), dtype=np.float32)
                for i in range(len(batch_texts))
            ]

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                if isinstance(output, tuple):
                    hs = output[0]
                    rest = output[1:]
                else:
                    hs = output
                    rest = None
                patch = torch.zeros_like(hs)
                for bi, mat in enumerate(patch_mats):
                    valid = int(attn[bi].sum().item())
                    take = min(valid, int(mat.shape[0]))
                    if take > 0:
                        patch[bi, :take, :] = torch.from_numpy(mat[:take]).to(device=device, dtype=hs.dtype)
                hs = hs + patch
                if rest is None:
                    return hs
                return (hs, *rest)

            handle = layer_module.register_forward_hook(hook)
            try:
                out = model(**tok, return_dict=True)
            finally:
                handle.remove()
            nll = per_example_nll(out.logits.float(), tok["input_ids"], tok["attention_mask"])
            token_count = int(tok["attention_mask"].sum().item())
            _ = float(nll.mean().detach().cpu().item())
            del out, nll, tok, patch_mats
            return token_count

        with torch.no_grad():
            for step in range(int(args.warmup_steps)):
                batch_texts = take_batch(texts, step * int(args.batch_size), int(args.batch_size))
                _ = run_step(batch_texts)
                row["completed_warmup_steps"] = step + 1
                torch.cuda.synchronize(device)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            bench_start = int(args.warmup_steps) * int(args.batch_size)
            for step in range(int(args.bench_steps)):
                batch_texts = take_batch(
                    texts,
                    bench_start + step * int(args.batch_size),
                    int(args.batch_size),
                )
                total_tokens += run_step(batch_texts)
                row["completed_steps"] = step + 1
                torch.cuda.synchronize(device)

    except Exception as exc:  # noqa: BLE001 - benchmark rows should survive failures.
        message = str(exc).splitlines()[0]
        row["error"] = message
        if torch is not None:
            oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
            if isinstance(exc, oom_type) or "out of memory" in message.lower():
                row["status"] = "cuda_oom"
            else:
                row["status"] = "runtime_error"
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            row["status"] = "runtime_error"

    elapsed = time.time() - start_time
    row["elapsed_sec"] = round(elapsed, 3)
    completed = int(row["completed_steps"])
    if completed > 0 and elapsed > 0.0:
        examples = completed * int(args.batch_size)
        row["examples_per_sec"] = round(examples / elapsed, 4)
        row["tokens_per_sec"] = round(total_tokens / elapsed, 2)
        row["avg_tokens_per_example"] = round(total_tokens / examples, 2)
    if torch is not None and torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
            row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / (1024**3), 3)
            row["device_total_gib"] = round(total_bytes / (1024**3), 3)
            row["device_free_after_gib"] = round(free_bytes / (1024**3), 3)
        except Exception:
            pass
    row["process_maxrss_gib"] = round(maxrss_gib(), 3)

    if args.row_json is not None:
        write_json(args.row_json, row)
    print(json.dumps(row, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-sizes", default="8,12,16,20,24")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--row-json", type=Path)
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--max-examples", type=int, default=4096)
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--bench-steps", type=int, default=2)
    ap.add_argument("--sample-mode", choices=["longest", "first", "random"], default="longest")
    ap.add_argument("--attn-impl", default="auto")
    ap.add_argument("--target-gib", type=float, default=0.0)
    ap.add_argument("--target-fraction", type=float, default=0.87)
    ap.add_argument("--target-tolerance-gib", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.batch_size > 0:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
