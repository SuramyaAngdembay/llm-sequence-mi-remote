# Qwen3-8B Token Necessity Results

Status: completed on Anvil, 2026-07-11.

These runs are the token-level necessity/ablation counterpart to the token causal sufficiency runs. Both used the chunked fp32 loss path:

- `LOSS_BATCH_SIZE=4`
- `CONTROL_SET=control5_active`
- `TOP_SETS=top5`
- `N_BOOTSTRAP=4000`
- partition/account: `gpu` / `cis230270-gpu`

## r6.2

Committed result directory:

- `results/qwen3_8b_token_necessity/l18_m04_k08_top5_control5_active_necessity_gpu_bs24_loss4/`

Anvil output directory:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_gpu/l18_m04_k08_top5_control5_active_necessity_gpu_bs24_loss4/`

Jobs:

- evaluator: `19058062`, completed, elapsed `00:23:17`, MaxRSS `21441208K`
- bootstrap: `19058063`, completed, elapsed `00:00:13`
- peak GPU memory poll: `13937 MiB`

Settings:

- `LAYER=18`
- `LATENT_MULT=4`
- `TOPK=8`
- `BATCH_SIZE=24`
- `PATCH_CHUNK_SIZE=24`
- `LOSS_BATCH_SIZE=4`
- candidate rows: `4480`

Bootstrap estimates:

| context | estimate | CI low | CI high |
|---|---:|---:|---:|
| role | 0.065688 | 0.053048 | 0.078480 |
| project_role | 0.064882 | 0.052287 | 0.077275 |
| dept_role | 0.056855 | 0.046256 | 0.067404 |
| team | 0.046524 | 0.036531 | 0.055851 |

Interpretation: the r6.2 token branch has a strong positive necessity signal. Removing the selected top token-SAE features hurts anomalous receivers substantially more than the active-control feature set.

Raw candidate rows were not committed to avoid unnecessary repo growth. If Magnolia needs row-level audit:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_gpu/l18_m04_k08_top5_control5_active_necessity_gpu_bs24_loss4/token_delta_sae_necessity_candidate_rows.csv`

## r6.2 Same-User Recovery

Committed result directory:

- `results/qwen3_8b_token_necessity/same_user_recovery/l18_m04_k08_top5_control5_active_necessity_no_same_user/`

Anvil output directory:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_no_same_user/l18_m04_k08_top5_control5_active_necessity_no_same_user/`

Jobs:

- evaluator: `19173969`, completed, elapsed `00:12:04`, MaxRSS `26663160K`
- bootstrap: `19173971`, completed, elapsed `00:00:21`
- peak GPU memory poll: `25517 MiB`

Settings:

- `LAYER=18`
- `LATENT_MULT=4`
- `TOPK=8`
- `BATCH_SIZE=96`
- `PATCH_CHUNK_SIZE=96`
- `LOSS_BATCH_SIZE=4`
- same-user benign matches excluded: yes
- candidate rows: `4480`

Bootstrap estimates:

| context | estimate | CI low | CI high |
|---|---:|---:|---:|
| project_role | 0.065188 | 0.055145 | 0.075023 |
| role | 0.062167 | 0.050265 | 0.073364 |
| dept_role | 0.056603 | 0.044996 | 0.068990 |
| team | 0.052234 | 0.042054 | 0.061397 |

Interpretation: the r6.2 token branch remains strongly necessity-positive
after same-user benign matches are excluded. This should supersede the older
permissive-match r6.2 necessity row for final paper tables.

## r4.2

Committed result directory:

- `results/qwen3_8b_r42_token_necessity/l26_m02_k04_top5_control5_active_necessity_gpu_bs14_loss4/`

Anvil output directory:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_r42_gpu/l26_m02_k04_top5_control5_active_necessity_gpu_bs14_loss4/`

Jobs:

- evaluator: `19058064`, completed, elapsed `01:41:04`, MaxRSS `44993384K`
- bootstrap: `19058065`, completed, elapsed `00:00:21`
- peak GPU memory poll: `17617 MiB`

Settings:

- `LAYER=26`
- `LATENT_MULT=2`
- `TOPK=4`
- `BATCH_SIZE=14`
- `PATCH_CHUNK_SIZE=14`
- `LOSS_BATCH_SIZE=4`
- candidate rows: `83776`

Bootstrap estimates:

| context | estimate | CI low | CI high |
|---|---:|---:|---:|
| role | 0.002370 | 0.001092 | 0.003635 |
| dept_role | 0.001475 | 0.000246 | 0.002661 |
| team | 0.000681 | -0.000758 | 0.002028 |
| dept | 0.000263 | -0.001018 | 0.001452 |

Interpretation: r4.2 shows a small positive necessity signal for `role` and `dept_role`, but it is far weaker than r6.2. This is consistent with the earlier transfer picture: r6.2 has a strong remote token mechanism, while r4.2 has weaker or shifted remote token support even though the detector transfers.

Raw candidate rows were not committed to avoid unnecessary repo growth. If Magnolia needs row-level audit:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_r42_gpu/l26_m02_k04_top5_control5_active_necessity_gpu_bs14_loss4/token_delta_sae_necessity_candidate_rows.csv`

## Notes

- The earlier r4.2 full attempt `19033317` failed from a late fp32 cross-entropy allocation.
- Commit `79edb72` changed the shared token patch scorer to chunk fp32 loss computation.
- Commit `bbd6c5c` made the Qwen3-8B necessity launchers default to `LOSS_BATCH_SIZE=4`.
- The completed full runs above used that chunked-loss path and did not OOM.
