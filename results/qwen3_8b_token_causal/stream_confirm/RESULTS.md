# Qwen3-8B R6.2 Streamed Evaluator Confirmation

Status: complete on Anvil, July 5, 2026.

This run reran the known-positive `r6.2` config with the current streamed token
causal evaluator used for the final `r4.2` uncapped suite.

## Run Scope

- config: `l18_m04_k08_stream_confirm_v1`
- layer: `18`
- latent multiplier: `4`
- SAE top-k: `8`
- context modes: `team,role,project_role,dept_role`
- top sets: `top1,top3,top5`
- control set: `control3`
- receiver cap: uncapped
- token delta dtype: `float32`
- output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_stream_confirm/l18_m04_k08_stream_confirm_v1`

## Headline

The streamed evaluator reproduces a clearly positive `r6.2` token-causal result.
This supports the interpretation that the negative `r4.2` run is a transfer
failure rather than an artifact of the streamed OOM rewrite.

Best bootstrap rows:

| Context | Target | Receivers | Estimate | 95% CI |
| --- | --- | ---: | ---: | --- |
| `dept_role` | `top5` | 70 | 0.018699 | [0.012729, 0.025157] |
| `role` | `top5` | 70 | 0.018603 | [0.013329, 0.024236] |
| `project_role` | `top5` | 70 | 0.013251 | [0.009462, 0.017327] |
| `team` | `top5` | 70 | 0.008216 | [0.005513, 0.010974] |

## Slurm Completion

- causal job: `18866598`, `COMPLETED`, elapsed `01:26:08`, `MaxRSS=39173076K`
- bootstrap job: `18866599`, `COMPLETED`, elapsed `00:00:20`, `MaxRSS=108152K`

## Committed Artifacts

- `l18_m04_k08_stream_confirm_v1/TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `l18_m04_k08_stream_confirm_v1/token_delta_sae_causal_summary.csv`
- `l18_m04_k08_stream_confirm_v1/token_delta_sae_causal_summary.json`
- `l18_m04_k08_stream_confirm_v1/token_delta_sae_causal_selected_sets.csv`
- `l18_m04_k08_stream_confirm_v1/token_delta_sae_causal_best_rows.csv`
- `l18_m04_k08_stream_confirm_v1/bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`
- `l18_m04_k08_stream_confirm_v1/bootstrap/token_delta_sae_bootstrap_summary.csv`

## Large File Left On Anvil

The candidate-row CSV was not committed. It remains available at:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_stream_confirm/l18_m04_k08_stream_confirm_v1/token_delta_sae_causal_candidate_rows.csv`
  - size: `23981738` bytes
