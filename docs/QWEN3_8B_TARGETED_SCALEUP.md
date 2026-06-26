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

The default launcher keeps the effective batch at `32`:

- `NPROC=4`
- `MICRO_BS=2`
- `GRAD_ACCUM=4`

## Submitted Anvil Pipeline

Submitted on 2026-06-26 from Anvil `login02`:

- training: Slurm `18597248`, `qwen_qlora_ddp`, partition `ai`, `4x H100`, pending on priority
- token extraction: Slurm `18597250`, `token_delta_extract`, dependency `afterok:18597248`
- token SAE frontier: Slurm `18597251`, `token_delta_sae`, dependency `afterok:18597250`

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
