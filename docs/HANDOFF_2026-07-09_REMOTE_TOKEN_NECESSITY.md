# Remote Token Necessity Plan

Status: ready for Anvil implementation and first headline runs.

## Why Necessity Is The Next Gap

The remote branch now has:

- detector success on `r6.2`
- detector success on `r4.2`
- positive token-level sufficiency on `r6.2`
- positive token-level sufficiency on native `r4.2`
- active-control audits for the current `8B` headline branches on both datasets

What it does **not** yet have is a direct token-level negative intervention
showing that ablating the selected remote features weakens the anomaly signal
more than ablating matched controls.

That is the missing remote necessity claim.

## Immediate Run Priority

Run these first:

1. `r6.2` `Qwen3-8B` necessity on the audited headline branch
2. `r4.2` `Qwen3-8B` necessity on the audited native headline branch

Do **not** prioritize these yet:

- `r6.2` `Qwen3-3B` necessity
  - first give `3B` the same `control5_active` audit treatment if we want a
    cleaner scale-comparison appendix result
- `r4.2` `Qwen3-3B` necessity
  - there is no remote `3B` `r4.2` branch in the repo/results at present

## Necessity Definition

The necessity evaluator uses paired positive-vs-benign receiver ablation:

- positive eval receivers are matched to benign eval receivers by context mode
- selected sparse token features are ablated only where they are active
- ablation shrinks the selected sparse feature activations toward zero by
  `alpha`
- the same protocol is run for the top feature set and the control feature set

Headline metric:

- `top_minus_control_necessity`

where

- `top_necessity_advantage = top_benign_mean_best_delta - top_positive_mean_best_delta`
- `control_necessity_advantage = control_benign_mean_best_delta - control_positive_mean_best_delta`
- `top_minus_control_necessity = top_necessity_advantage - control_necessity_advantage`

Interpretation:

- more negative deltas on positive receivers mean ablation removes anomaly
  evidence
- a positive `top_minus_control_necessity` means the selected top features are
  more necessary than the matched control features

## New Scripts

Evaluator:

- `scripts/eval_token_delta_sae_necessity.py`

Bootstrap:

- `scripts/bootstrap_token_delta_sae_necessity.py`

Slurm templates:

- `slurm/eval_token_delta_sae_necessity.template.sbatch`
- `slurm/bootstrap_token_delta_sae_necessity_cpu.template.sbatch`

Headline wrappers:

- `scripts/submit_qwen3_8b_token_necessity_bundle_anvil.sh`
- `scripts/submit_qwen3_8b_r42_token_necessity_bundle_anvil.sh`

Probe wrappers:

- `scripts/submit_qwen3_8b_token_necessity_gpu_debug_probe_anvil.sh`
- `scripts/submit_qwen3_8b_r42_token_necessity_gpu_debug_probe_anvil.sh`

The probe wrappers submit short A100 jobs to Anvil `gpu-debug` with
`MAX_PAIRS=16` by default. They exercise the real necessity evaluator path and
record:

- Slurm RSS via `sacct`
- `/usr/bin/time -v` output in the probe output directory
- `nvidia-smi` polling at `gpu_poll_${SLURM_JOB_ID}.csv`

Hardware note:

- Anvil `debug` is CPU-only.
- Anvil `gpu-debug` is A100-only, 30-minute max walltime, and uses
  `cis230270-gpu`.
- There is no separate H100 `ai-debug` partition visible from Anvil, so H100
  VRAM confirmation still requires a short normal `ai` job.

## Headline Run 1: R6.2 8B

This is the audited `r6.2` headline branch:

- `layer=18`
- `latent_mult=4`
- `k=8`
- `top5`
- `control5_active`

Launch:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_token_necessity_bundle_anvil.sh
```

Recommended preflight probe:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_token_necessity_gpu_debug_probe_anvil.sh
```

Default output root:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b/l18_m04_k08_top5_control5_active_necessity/`

## Headline Run 2: R4.2 8B

This is the audited native `r4.2` headline branch:

- `layer=26`
- `latent_mult=2`
- `k=4`
- `top5`
- `control5_active`

Launch:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r42_token_necessity_bundle_anvil.sh
```

If `gpu` / A100 scheduling is easier on Anvil, use:

```bash
bash scripts/submit_qwen3_8b_r42_token_necessity_gpu_anvil.sh
```

Recommended preflight probe:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r42_token_necessity_gpu_debug_probe_anvil.sh
```

Default output root:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_r42/l26_m02_k04_top5_control5_active_necessity/`

## Expected Outputs

Per run:

- `token_delta_sae_necessity_summary.csv`
- `token_delta_sae_necessity_best_rows.csv`
- `token_delta_sae_necessity_selected_sets.csv`
- `TOKEN_DELTA_SAE_NECESSITY_REPORT.md`
- `token_delta_sae_necessity_summary.json`

Per bootstrap:

- `bootstrap/token_delta_sae_necessity_bootstrap_summary.csv`
- `bootstrap/TOKEN_DELTA_SAE_NECESSITY_BOOTSTRAP_REPORT.md`

Candidate-row CSVs are expected but do not need to be committed by default.

## Decision Rule

If the best necessity rows:

- stay positive on `top_minus_control_necessity`
- and keep bootstrap CIs above zero

then the remote `8B` branch can be described as having both:

- audited causal sufficiency
- audited token-level necessity

on the corresponding dataset.

If either necessity run fails while sufficiency stays positive, then the claim
should remain:

- strong sufficiency
- incomplete necessity

## Current Scope Boundary

The immediate paper-value path is:

- `8B r6.2` necessity
- `8B r4.2` necessity
- final detector and mechanistic tables

Not:

- broad new model sweeps
- `r4.2 3B` branch creation
- `31B` scale-up
