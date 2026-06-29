# Handoff 2026-06-29: Qwen3-8B Token Causal Eval

This handoff starts the first model-level causal patching run for the completed
`Qwen/Qwen3-8B` token frontier.

## What Is Already Done

Completed on Anvil:

- `Qwen3-8B` QLoRA training
- token-level delta extraction
- streamed token-SAE frontier over layers `18,26,34`

Repo result bundle:

- `results/qwen3_8b_token_frontier/HANDOFF.md`
- `results/qwen3_8b_token_frontier/token_extract/token_extract_summary.json`
- `results/qwen3_8b_token_frontier/token_sae/DELTA_SAE_FRONTIER_REPORT.md`

## Why This Must Run On Anvil

Do **not** try to run the `Qwen3-8B` causal patch jobs on Magnolia:

- the live adapter checkpoint is on Anvil project storage
- the token delta cache is about `956G`
- the streamed frontier artifacts are on Anvil project storage
- causal patching needs the H100-side runtime, not just the lightweight git summaries

Magnolia should only be used afterward for:

- reading committed markdown/CSV summaries
- strict comparison scripts against the local session-AE baseline
- pulling back small result files with `git` or `rsync`

## Runtime Paths

These are the successful `Qwen3-8B` paths the causal eval should use:

- adapter:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter`
- token cache:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2`
- SAE frontier:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream`

The eval bundle script now defaults to those exact roots.

## Recommended First Causal Configs

From the frontier handoff, start with:

1. `layer=18`, `latent_mult=4`, `k=4`
2. `layer=18`, `latent_mult=2`, `k=4`
3. `layer=18`, `latent_mult=4`, `k=8`

These are the highest-value first checks. Do not interpret frontier proxy
metrics as a substitute for model-level causal patching.

## Commands On Anvil

Update the repo first:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
```

Submit the full recommended causal suite:

```bash
bash scripts/submit_qwen3_8b_recommended_causal_suite_anvil.sh
```

If you only want one config first:

```bash
LAYER=18 LATENT_MULT=4 TOPK=4 \
bash scripts/submit_qwen3_8b_token_eval_bundle_anvil.sh
```

## What Each Submission Does

Each `submit_qwen3_8b_token_eval_bundle_anvil.sh` run launches:

1. GPU token causal patch eval on the chosen SAE config
2. dependent CPU bootstrap over the resulting best-row CSV

## Expected Output Files

Per config under:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh/<tag>/`

expect:

- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_best_rows.csv`
- `token_delta_sae_causal_selected_sets.csv`
- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `bootstrap/token_delta_sae_bootstrap_summary.csv`
- `bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`

## What To Commit Back

Commit lightweight files back into git under a new results directory, for example:

- `results/qwen3_8b_token_causal/<tag>/TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `results/qwen3_8b_token_causal/<tag>/token_delta_sae_causal_summary.csv`
- `results/qwen3_8b_token_causal/<tag>/token_delta_sae_causal_selected_sets.csv`
- `results/qwen3_8b_token_causal/<tag>/bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`
- `results/qwen3_8b_token_causal/<tag>/bootstrap/token_delta_sae_bootstrap_summary.csv`

Large caches and model artifacts should stay Anvil-only.

## Success Criterion

The `Qwen3-8B` branch only counts as an improvement if its token-level causal
patching:

- beats the current best `Qwen 3B` remote token result (`0.001446`)
- keeps positive bootstrap intervals
- stays ahead of the matched local session-AE day-level baseline (`0.001133`)

Until those causal outputs exist, the `Qwen3-8B` branch should be treated as a
promising scale-up with a successful frontier, not as a mechanistic win.
