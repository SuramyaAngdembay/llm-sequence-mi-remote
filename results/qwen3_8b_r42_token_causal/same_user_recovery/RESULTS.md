# Qwen3-8B r4.2 Same-User-Excluded Token Recovery

This directory records the r4.2 native token-mechanism recovery reruns after the
same-user donor/match validity audit.

## Causal Run

- GPU job: `19271697`
- CPU bootstrap job: `19271698`
- Status: completed
- Runtime: `15:53:04`
- Dataset: r4.2
- Model branch: Qwen3-8B QLoRA r4.2 adapter
- Token-SAE config: layer `26`, latent multiplier `2`, top-k `4`
- Intervention: token-level causal patching
- Feature target/control: `top5` vs `control5_active`
- Same-user donors excluded: yes
- Source output:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_no_same_user/l26_m02_k04_top5_control5_active_no_same_user`

| context_mode | n_receivers | n_complete_receivers | estimate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| team | 1301 | 1052 | 0.001418 | [0.001139, 0.001690] |
| role | 1119 | 1110 | 0.001112 | [0.000826, 0.001391] |
| dept_role | 1039 | 1034 | 0.001067 | [0.000824, 0.001305] |
| dept | 1199 | 1157 | 0.000982 | [0.000751, 0.001215] |

The estimate is the paired complete receiver-level top-vs-control contrast,
matching the bootstrap estimand. `n_receivers` is the available positive
receiver count for the context; `n_complete_receivers` is the count with all
top/control and benign/anomalous donor values present.

Runtime:

- peak GPU memory: `34739 MiB`
- active-average GPU memory: about `14.2 GiB`
- CPU MaxRSS: about `163.0 GiB`

Large untracked audit CSV:

- `token_delta_sae_causal_candidate_rows.csv`
- rows: `1086704`
- size: about `185.7 MiB`
- rsync source:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_active_control_no_same_user/l26_m02_k04_top5_control5_active_no_same_user/token_delta_sae_causal_candidate_rows.csv`

The candidate-row CSV is intentionally not committed because it exceeds normal
GitHub file-size limits.

## Necessity Run

The matching same-user-excluded r4.2 necessity result is tracked under:

`results/qwen3_8b_r42_token_necessity/same_user_recovery/l26_m02_k04_top5_control5_active_necessity_no_same_user/`

Top rows:

- `dept_role / top5`: estimate `0.00292177`, CI
  `[0.00145958, 0.00437872]`
- `role / top5`: estimate `0.00207545`, CI
  `[0.000678829, 0.00341792]`

## Interpretation

The r4.2 native token mechanism is now positive under same-user exclusion.

This is distinct from the earlier direct-transfer test from the r6.2 layer-18
branch, which stayed negative on r4.2. The updated interpretation is:

- direct r6.2 token-circuit transfer to r4.2 failed
- r4.2 still has a native Qwen3-8B token mechanism
- the native r4.2 mechanism is smaller than r6.2 but passes the stricter
  same-user control
- r4.2 necessity is stronger than r4.2 causal, especially for `dept_role` and
  `role`
