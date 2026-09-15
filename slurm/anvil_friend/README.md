# Anvil runners used for the train-matched factorial and the LANL replication (Sept 2026)

Run on a collaborator's Anvil allocation (`x-bbhusal1`; accounts cis260991-gpu / tra250034-gpu). Paths assume `$SCRATCH/p1`, `$SCRATCH/p1_8b`, `$SCRATCH/lanl`, `HF_HUB_CACHE=$SCRATCH/hf_cache` (offline), conda env `$SCRATCH/conda_envs/cert-qlora` (transformers 5.16.1).

- `run_mode.sh` — 3B factorial condition: build serialization (`--profile-mode`), QLoRA, delta extraction, SAE, quick ROC (A100-40GB, `gpu` partition).
- `sae_only.sh` — SAE re-run with `--max-rows 2000000` (the `--max-rows 0` default requested a 356 GiB allocation; root cause, not a bigger node).
- `run_8b.sh` / `smoke_8b.sh` — Qwen3-8B condition on 4xH100 DDP (`ai` partition; `--micro-batch-size 2 --grad-accum 2`, checkpoint auto-resume); smoke test on `gpu-debug` first.
- `fold_eval.sbatch` — fold-aligned re-score (`scripts/eval_fold_aligned_detector_metrics.py`) + `scripts/factorial/fold_boot.py`.
- `lanl_run.sh` / `lanl_extract.sh` — LANL condition training and delta extraction (`--chunk-examples 512` to keep host RAM under 180 GB on 1.5k-token windows).
- `sitecustomize.py` — scoped no-op for transformers' `check_torch_load_is_safe` (CVE-2025-32434 guard) so checkpoint resume works on the cluster's torch; put on `PYTHONPATH` only for resume jobs.

Repo-side changes that these runs needed are in `scripts/train_qlora.py` (warmup_ratio -> warmup_steps when the installed transformers no longer accepts warmup_ratio) and `scripts/extract_adapter_deltas.py` (`ex.get("n_sessions_total", ...)` for serializations without that field). See `results/train_matched_factorial/README.md` and `results/lanl_auth_replication/README.md` for the outputs and job ids.

Note: the 3B runs used `configs/qwen3b_qlora_session.yaml` with `micro_batch_size: 4` / `gradient_accumulation_steps: 4` on A100-40GB (same effective batch of 16 as the repo default `1 x 16`); the repo config is left at the conservative default.
