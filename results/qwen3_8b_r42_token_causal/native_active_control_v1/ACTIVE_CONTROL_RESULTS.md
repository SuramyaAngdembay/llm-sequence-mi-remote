# Qwen3-8B R4.2 Native Active-Control Results

Status: completed on Anvil, 2026-07-09.

This bundle records the tightened active-control rerun for the strongest
`r4.2` native remote token-causal config found in the native search:

- `layer=26`
- `latent_mult=2`
- `k=4`
- target set: `top5`
- control set: `control5_active`
- context modes: `team,role,dept,dept_role`

The run used the full uncapped streamed evaluator with all `1309` positive
eval receiver-days.

## Jobs

| job | id | partition | state | elapsed | max RSS |
|---|---:|---|---|---:|---:|
| causal | `18994506` | `gpu` / A100 | completed | `17:28:09` | `160722336K` |
| bootstrap | `18994507` | `shared` | completed | `00:00:23` | `187620K` |

Both jobs completed with exit code `0:0`.

The A100 run used:

- `BATCH_SIZE=12`
- `PATCH_CHUNK_SIZE=12`
- `CAUSAL_MEM=240G`
- `TOKEN_DELTA_DTYPE=float32`
- `MAX_RECEIVERS=0`
- `MAX_CANDIDATE_DONORS=16`
- `N_BOOTSTRAP=4000`

The causal evaluator wrote the full expected `1,333,016` candidate rows.

## Bootstrap Results

| context | estimate | 95% CI | read |
|---|---:|---:|---|
| `dept` | `0.001028` | `[0.000780, 0.001270]` | positive |
| `team` | `0.001016` | `[0.000752, 0.001277]` | positive |
| `role` | `0.000748` | `[0.000476, 0.001021]` | positive |
| `dept_role` | `0.000678` | `[0.000410, 0.000951]` | positive |

## Interpretation

The native `r4.2` remote token-causal result survives the stronger
`control5_active` audit. The effect is much smaller than the audited `r6.2`
remote token result, but it is consistently positive across all four context
modes with bootstrap confidence intervals above zero.

This updates the `r4.2` read:

- direct transfer of the `r6.2` token mechanism to `r4.2`: failed
- native `r4.2` remote token mechanism: positive after active-control audit
- local `r4.2` session mechanism: positive, similar scale

The active-control result is slightly below the earlier native-search result
because the control is stronger:

- earlier native `l26_m02_k04 / team / top5`: `0.001307`
- active-control `l26_m02_k04 / team / top5`: `0.001016`

## Committed Artifacts

Committed under:

`results/qwen3_8b_r42_token_causal/native_active_control_v1/l26_m02_k04_top5_control5_active_gpu_bs12/`

Included:

- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `token_delta_sae_causal_best_rows.csv`
- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_summary.json`
- `token_delta_sae_causal_selected_sets.csv`
- `bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`
- `bootstrap/token_delta_sae_bootstrap_summary.csv`

Not committed:

- `token_delta_sae_causal_candidate_rows.csv`

The candidate-row CSV is about `239 MB`, so it remains on Anvil for optional
deep audit.

## Anvil Paths For Rsync

Full Anvil output directory:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/`

Candidate-row CSV:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/token_delta_sae_causal_candidate_rows.csv`

Suggested Magnolia audit target:

`srangdembay@magnolia.usm.edu:/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/anvil_token_causal/r42_native_active_control_l26_m02_k04/`

To rsync the compact committed-style artifacts:

```bash
rsync -avP \
  /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/ \
  --exclude token_delta_sae_causal_candidate_rows.csv \
  srangdembay@magnolia.usm.edu:/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/anvil_token_causal/r42_native_active_control_l26_m02_k04/
```

To rsync only the candidate-row CSV for deep audit:

```bash
rsync -avP \
  /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/token_delta_sae_causal_candidate_rows.csv \
  srangdembay@magnolia.usm.edu:/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/anvil_token_causal/r42_native_active_control_l26_m02_k04/
```
