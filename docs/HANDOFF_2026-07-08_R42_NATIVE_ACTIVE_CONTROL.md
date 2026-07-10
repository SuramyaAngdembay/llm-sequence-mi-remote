# R4.2 Native Active-Control Audit

Status: completed on Anvil, 2026-07-09.

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

That was enough for the main native-rediscovery claim, but the cleanest next
robustness check was the same one used on the positive `r6.2` branch:

- rerun the winning config with `control5_active`
- keep the full uncapped streamed evaluator
- use the hardware-specific full-run-safe batch settings

The goal was not to search again. The goal was to test whether the positive
`r4.2` native result survives a stronger active control.

## Completed Active-Control Result

The A100/GPU launcher completed successfully:

- causal job: `18994506`
- bootstrap job: `18994507`
- causal partition/account: `gpu` / `cis230270-gpu`
- causal node: `g002`
- causal elapsed: `17:28:09`
- causal max RSS: `160722336K`
- bootstrap elapsed: `00:00:23`
- candidate rows written: `1,333,016`

Bootstrap active-control estimates:

| context | estimate | 95% CI |
|---|---:|---:|
| `dept` | `0.001028` | `[0.000780, 0.001270]` |
| `team` | `0.001016` | `[0.000752, 0.001277]` |
| `role` | `0.000748` | `[0.000476, 0.001021]` |
| `dept_role` | `0.000678` | `[0.000410, 0.000951]` |

Conclusion: the native `r4.2` token mechanism survives the stronger
`control5_active` audit. The effect is smaller than the original native-search
estimate, but every context mode remains positive with a bootstrap CI above
zero.

Committed compact results are under:

`results/qwen3_8b_r42_token_causal/native_active_control_v1/l26_m02_k04_top5_control5_active_gpu_bs12/`

The full Anvil output remains at:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/`

## Launch Command

Original Anvil launch command:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r42_native_active_control_anvil.sh
```

### A100/GPU Partition Variant

The completed run used the A100 `gpu` partition variant:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r42_native_active_control_gpu_anvil.sh
```

This launcher keeps the experimental specification unchanged, but uses the
committed A100 causal-VRAM probe and full-run result:

- `partition=gpu`
- `account=cis230270-gpu`
- `BATCH_SIZE=12`
- `PATCH_CHUNK_SIZE=12`
- `CAUSAL_MEM=240G`

The short A100 probe found `BATCH_SIZE=16` as its highest clean point, but the
full active-control evaluator job `18971394` OOMed after 1h27m when
`logits.float()` plus cross-entropy requested a late 9.44 GiB allocation.
`BATCH_SIZE=12` is therefore the full-run default; `16`, `20`, and `24` are not
safe for this workload on a 40GB A100.

## Fixed Experimental Specification

Both launchers hard-code the current native `r4.2` winner:

- `LAYER=26`
- `LATENT_MULT=2`
- `TOPK=4`
- `TOP_SETS=top5`
- `CONTROL_SET=control5_active`
- `ACTIVE_CONTROL_MIN_FRAC=0.002`
- `CONTEXT_MODES=team,role,dept,dept_role`
- `TOKEN_DELTA_DTYPE=float32`
- `MAX_RECEIVERS=0`
- `MAX_CANDIDATE_DONORS=16`
- `N_BOOTSTRAP=4000`

Hardware-specific resource settings:

| launcher | partition | batch | patch chunk | memory | output root |
|---|---|---:|---:|---:|---|
| `submit_qwen3_8b_r42_native_active_control_anvil.sh` | `ai` / H100 | `24` | `24` | `480G` | `token_delta_sae_causal_qwen3_8b_r42_native_active_control_v1` |
| `submit_qwen3_8b_r42_native_active_control_gpu_anvil.sh` | `gpu` / A100 | `12` | `12` | `240G` | `token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1` |

The completed run used the A100/GPU output directory:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/`

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

## Decision Rule Result

The best `control5_active` rows stayed:

- positive on `top_minus_control_advantage`
- with a bootstrap CI above zero

So the native `r4.2` remote mechanism should be treated as a stronger,
audited result rather than just a first positive search hit.

## Current Comparison Context

Matched day-level comparator on the same `r4.2` receiver-days:

- best local adaptive session-AE: about `0.000986`
- best local residual baseline: about `0.001380`
- current native remote `r4.2` winner: `0.001307`

So this active-control rerun is a robustness check on a result that is already:

- better than the local adaptive comparator
- close to, but not above, the strongest local residual comparator

## Rsync Paths

Full compact-or-deep-audit Anvil output directory:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/`

Large candidate-row CSV for optional Magnolia audit:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/token_delta_sae_causal_candidate_rows.csv`

Suggested Magnolia target:

`srangdembay@magnolia.usm.edu:/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/anvil_token_causal/r42_native_active_control_l26_m02_k04/`

Compact artifact rsync:

```bash
rsync -avP \
  /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/ \
  --exclude token_delta_sae_causal_candidate_rows.csv \
  srangdembay@magnolia.usm.edu:/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/anvil_token_causal/r42_native_active_control_l26_m02_k04/
```

Candidate-row-only rsync:

```bash
rsync -avP \
  /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_gpu_v1/l26_m02_k04_top5_control5_active_gpu_bs12/token_delta_sae_causal_candidate_rows.csv \
  srangdembay@magnolia.usm.edu:/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/anvil_token_causal/r42_native_active_control_l26_m02_k04/
```
