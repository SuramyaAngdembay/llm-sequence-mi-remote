# Qwen3-8B Targeted Scale-Up

This supersedes the earlier `Qwen/Qwen2.5-7B` target for the next scale-up run.

## Rationale

The `Qwen 3B` token branch beat the matched local session-AE day-level baseline
and stayed positive under bootstrap. The next run should therefore keep the same
mechanistic protocol and upgrade only the base sequence model.

## Target

- base model: `Qwen/Qwen3-8B`
- training: 4-bit NF4 QLoRA with bf16 compute and LoRA adapters
- Anvil partition: `ai`
- training hardware: one 4x H100 80GB node
- extraction/eval hardware: one H100 80GB
- Qwen3 environment: `/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3`

QLoRA does not require a separate pre-quantized checkpoint. The normal Hugging
Face model weights are loaded with `BitsAndBytesConfig(load_in_4bit=True)`.

## First Targeted Band

`Qwen3-8B` has a deeper 36-layer stack, so the first token-delta extraction band is:

- layer `18`
- layer `26`
- layer `34`

First SAE regimes remain:

- `latent_mult=2, k=8`
- `latent_mult=4, k=4`

## Launch

After the Qwen3-capable env and HF cache are validated:

```bash
bash scripts/submit_qwen3_8b_targeted_pipeline_anvil.sh
```

The original conservative launcher kept the effective batch at `32`:

- `NPROC=4`
- `MICRO_BS=2`
- `GRAD_ACCUM=4`

## VRAM Retune

The first live H100 run, Slurm `18597248`, used only about `20 GB` per `80 GB`
H100 with `MICRO_BS=2`, `GRAD_ACCUM=4`, and gradient checkpointing enabled.
It was healthy but underutilized, and the full validation pass every `1000`
steps cost about `36.5` minutes. It hit the `12:00:00` Slurm limit at
`checkpoint-10000`, which is `0.2557` epochs out of a planned `39101` optimizer
steps. This was a wall-clock limit, not a model failure.

Retune target:

- aim for roughly `60-70 GB` per H100, leaving some headroom under the `80 GB`
  H100 limit
- use `MICRO_BS=12`, `GRAD_ACCUM=1`
- keep gradient checkpointing enabled with `GC_MODE=on`
- disable intermediate eval with `EVAL_STRATEGY=no`
- keep checkpoint saves every `1000` steps
- use a `48:00:00` wall clock for the 4-H100 training job, because the
  conservative run reached only about one quarter epoch in `12:00:00`

The high-VRAM retune intentionally changes the effective batch from `32` to `48`.
If it OOMs, fall back first to `MICRO_BS=10`, `GRAD_ACCUM=1`, `GC_MODE=on`;
if that still OOMs, use `MICRO_BS=8`, `GRAD_ACCUM=1`, `GC_MODE=on`.

VRAM depends on all of these:

- model family and architecture (`Qwen3-8B` is not interchangeable with `Qwen2.5-7B`)
- sequence length and padding distribution from the session dataset
- per-GPU microbatch size
- gradient checkpointing mode
- attention implementation
- whether the pass is training, eval, checkpoint save, or adapter extraction

To make the estimate explicit, run the one-H100 stress benchmark:

```bash
sbatch slurm/benchmark_qwen3_8b_vram.template.sbatch
```

The benchmark uses the exact Qwen3-8B QLoRA stack and synthetic full-length
`seq_len=2048` batches, so it is closer to a worst-case per-GPU estimate than a
small random sample of session rows. Results are written to:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_vram_benchmark/qwen3_8b_qlora_vram_benchmark.csv`
- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_vram_benchmark/qwen3_8b_qlora_vram_benchmark.json`

Submitted benchmark:

- Slurm `18616324`, `qwen3_vram_bench`
- dependency `afterany:18597248`
- resources: `1x H100`, `16` CPU cores, `160G` RAM, `01:30:00`
- benchmark settings: `MICRO_BATCHES=4,8,12,16`, `SEQ_LEN=2048`, `GC_MODE=off`
- high-VRAM retune job `18615954` was updated to wait on `afterok:18616324`

Benchmark outcome:

- Slurm `18616324`, `GC_MODE=off`, `MICRO_BS=4`: OOM at `78.381 GB`
  allocated / `78.494 GB` reserved
- Slurm `18630407`, `GC_MODE=off`: `MICRO_BS=1` ok at `42.970 GB`,
  `MICRO_BS=2` ok but too close at `77.413 GB`, `MICRO_BS=3` OOM
- Slurm `18630408`, `GC_MODE=on`: `MICRO_BS=4` ok at `27.063 GB`,
  `MICRO_BS=8` ok at `45.598 GB`, `MICRO_BS=10` ok at `54.866 GB`,
  `MICRO_BS=12` ok at `64.134 GB`

Chosen training setting after the benchmarks:

- `MICRO_BS=12`
- `GRAD_ACCUM=1`
- `GC_MODE=on`
- effective batch `48` across `4x H100`
- expected peak VRAM about `64-66 GB/H100` under synthetic full-length
  `seq_len=2048` stress

Optional 70k-MiB VRAM probe:

- launcher: `scripts/submit_qwen3_8b_vram_70k_probe_anvil.sh`
- default probe: `MICRO_BATCHES=12,13`, `GC_MODE=on`, `SEQ_LEN=2048`
- output dir:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/qwen3_8b_vram_benchmark_70k_probe`
- rationale: `MICRO_BS=12` measured `64.134 GiB` allocated / `65.490 GiB`
  reserved; linear extrapolation puts `MICRO_BS=13` near the requested
  `70k MiB` region while still leaving visible H100 headroom
- do not make `70k MiB` the default training target unless the probe shows
  stable headroom; keep the live fresh training run at `MICRO_BS=12`

For retunes after an already-running job, use `RESUME_FROM_CHECKPOINT=latest`
so the Slurm script resolves the newest `checkpoint-*` under `OUTPUT_DIR` at
job start.

Submitted high-VRAM retune on 2026-06-26:

- left conservative training job `18597248` running on `h014`
- canceled stale conservative downstream jobs `18597250` and `18597251`
- submitted high-VRAM training job `18615954`; initially `afterany:18597248`,
  then updated to `afterok:18616324` so the VRAM benchmark runs first
- submitted high-VRAM token extraction `18615955`, dependency `afterok:18615954`
- submitted high-VRAM token SAE `18615957`, dependency `afterok:18615955`
- after conservative job `18597248` timed out, pending retune job `18615954`
  was updated in Slurm to `TimeLimit=2-00:00:00`
- `slurm/train_qlora_ddp.template.sbatch` now defaults to `#SBATCH --time=2-00:00:00`
  for future QLoRA DDP submissions

High-VRAM submission settings:

- `MICRO_BS=12`
- `GRAD_ACCUM=1`
- `GC_MODE=on`
- `EVAL_STRATEGY=no`
- `SAVE_STEPS=1000`
- `RESUME_FROM_CHECKPOINT=latest`
- `IGNORE_DATA_SKIP=1`

This is the first attempt to push VRAM utilization toward the requested
`~70 GB/H100` range. Verify actual memory with:

```bash
srun --jobid=18615954 --overlap --nodes=1 --ntasks=1 \
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

Corrected benchmark-backed chain submitted on 2026-06-27:

- train: Slurm `18631661`, `qwen_qlora_ddp`, `4x H100`, `48:00:00`,
  failed before training at resume RNG restore
- token extraction: Slurm `18631662`, dependency `afterok:18631661`
- token SAE: Slurm `18631663`, dependency `afterok:18631662`
- training wrapper: `slurm/train_qlora_ddp.template.sbatch`
- training script: `scripts/train_qlora.py`
- pipeline launcher: `scripts/submit_qwen3_8b_targeted_pipeline_anvil.sh`

Resume failure and fix:

- job `18631661` started on `h013` at `2026-06-27T08:28:47` and failed at
  `2026-06-27T08:32:21`
- it loaded the intended settings (`MICRO_BS=12`, `GRAD_ACCUM=1`, `GC_MODE=on`)
  and resolved `checkpoint-10000`
- failure cause: `transformers.Trainer._load_rng_state` called
  `torch.load(..., weights_only=True)` on `rng_state_*.pth`, which rejected the
  NumPy RNG pickle globals
- `scripts/train_qlora.py` now supports `--skip-rng-state-resume`, which keeps
  checkpoint resume for model/optimizer/scheduler while skipping only RNG-state
  restoration
- `slurm/train_qlora_ddp.template.sbatch` passes this when
  `SKIP_RNG_STATE_RESUME=1`
- `scripts/submit_qwen3_8b_targeted_pipeline_anvil.sh` defaults
  `SKIP_RNG_STATE_RESUME=1` for this Qwen3 resume path

Resubmitted after the RNG-resume fix, then canceled before start:

- train: Slurm `18649348`, `qwen_qlora_ddp`, `4x H100`, `48:00:00`,
  canceled while still pending
- token extraction: Slurm `18649349`, dependency `afterok:18649348`
- token SAE: Slurm `18649350`, dependency `afterok:18649349`
- reason: this would still have been a hybrid continuation from
  `checkpoint-10000` while changing effective batch from `32` to `48`

Fresh benchmark-backed chain submitted on 2026-06-27:

- train: Slurm `18649521`, `qwen_qlora_ddp`, `4x H100`, `48:00:00`,
  queued with `Reason=Priority`
- token extraction: Slurm `18649522`, dependency `afterok:18649521`
- token SAE: Slurm `18649523`, dependency `afterok:18649522`
- checkpoint root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh`
- token cache root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh`
- frontier root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh`
- resume disabled: `RESUME_FROM_CHECKPOINT=`, `IGNORE_DATA_SKIP=0`,
  `SKIP_RNG_STATE_RESUME=0`

The earlier unsafe `MICRO_BS=16`, `GC_MODE=off` train chain
`18615954 -> 18615955 -> 18615957` was canceled after the benchmark showed
OOM at `MICRO_BS=4`, `GC_MODE=off`.

## Original Conservative Pipeline

Submitted on 2026-06-26 from Anvil `login02`:

- training: Slurm `18597248`, `qwen_qlora_ddp`, partition `ai`, `4x H100`
- token extraction: Slurm `18597250`, canceled after the VRAM retune was queued
- token SAE frontier: Slurm `18597251`, canceled after the VRAM retune was queued

Runtime inputs:

- conda env: `/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3`
- config: `configs/qwen3_8b_qlora_session_targeted.yaml`
- checkpoint root: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp`
- token cache root: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted`
- frontier root: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b`

Qwen3 cache and environment validation:

- `Qwen/Qwen3-8B` is cached under project `HF_HOME`
- offline `AutoConfig` loads as `model_type=qwen3`
- offline tokenizer loads with the fast tokenizer
- `transformers=4.51.3`, `tokenizers=0.21.4`

## Success Criterion

The run only advances the branch if it improves mechanistic repair evidence:

- beat the current control-fixed token best effect, `0.001446`
- keep positive bootstrap intervals
- widen the margin over the matched local session-AE day-level baseline, `0.001133`
- ideally stay positive across more than one context/top-set row
