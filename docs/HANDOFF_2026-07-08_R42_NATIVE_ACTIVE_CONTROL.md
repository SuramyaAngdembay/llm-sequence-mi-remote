# R4.2 Native Active-Control Audit

Status: ready to launch on Anvil.

## Why This Run Exists

The native `r4.2` remote token search already found a positive full uncapped
config:

- `layer=26`
- `latent_mult=2`
- `k=4`
- best row: `team / top5`
- estimate `0.001307`
- bootstrap CI `[0.000960, 0.001663]`

See:

- `results/qwen3_8b_r42_token_causal/native_search_v3_bs24/RESULTS.md`

That is enough for the main native-rediscovery claim, but the cleanest next
robustness check is the same one used on the positive `r6.2` branch:

- rerun the winning config with `control5_active`
- keep the full uncapped streamed evaluator
- keep the successful `BATCH_SIZE=24`, `PATCH_CHUNK_SIZE=24`, `CAUSAL_MEM=480G`
  settings

The goal is not to search again. The goal is to test whether the positive
`r4.2` native result survives a stronger active control.

## Launch Command

On Anvil:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r42_native_active_control_anvil.sh
```

## Fixed Run Specification

This wrapper hard-codes the current native `r4.2` winner:

- `LAYER=26`
- `LATENT_MULT=2`
- `TOPK=4`
- `TOP_SETS=top5`
- `CONTROL_SET=control5_active`
- `ACTIVE_CONTROL_MIN_FRAC=0.002`
- `CONTEXT_MODES=team,role,dept,dept_role`
- `BATCH_SIZE=24`
- `PATCH_CHUNK_SIZE=24`
- `CAUSAL_MEM=480G`
- `TOKEN_DELTA_DTYPE=float32`
- `MAX_RECEIVERS=0`
- `MAX_CANDIDATE_DONORS=16`
- `N_BOOTSTRAP=4000`

The wrapper writes to:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_v1/l26_m02_k04_top5_control5_active/`

## Expected Outputs

Per causal run:

- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_best_rows.csv`
- `token_delta_sae_causal_selected_sets.csv`
- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`

Per bootstrap:

- `bootstrap/token_delta_sae_bootstrap_summary.csv`
- `bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`

Candidate-row CSVs do not need to be committed by default.

## Decision Rule

If the best `control5_active` row stays:

- positive on `top_minus_control_advantage`
- with a bootstrap CI above zero

then the native `r4.2` remote mechanism should be treated as a stronger,
audited result rather than just a first positive search hit.

If it collapses under `control5_active`, then the native `r4.2` result remains
interesting but should be reported more cautiously than the `r6.2` audited
headline.

## Current Comparison Context

Matched day-level comparator on the same `r4.2` receiver-days:

- best local adaptive session-AE: about `0.000986`
- best local residual baseline: about `0.001380`
- current native remote `r4.2` winner: `0.001307`

So this active-control rerun is a robustness check on a result that is already:

- better than the local adaptive comparator
- close to, but not above, the strongest local residual comparator
