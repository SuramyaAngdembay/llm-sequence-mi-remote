# Qwen3-8B r4.2 Same-User-Excluded Necessity Recovery

This directory records the r4.2 token-necessity recovery run after the
same-user match validity audit.

## Run

- GPU job: `19222733`
- CPU bootstrap job: `19222735`
- Status: completed
- Runtime: `02:12:55`
- Dataset: r4.2
- Model branch: Qwen3-8B QLoRA r4.2 adapter
- Token-SAE config: layer `26`, latent multiplier `2`, top-k `4`
- Intervention: token-level necessity ablation
- Feature target/control: `top5` vs `control5_active`
- Same-user matches excluded: yes
- Source output:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_r42_no_same_user/l26_m02_k04_top5_control5_active_necessity_no_same_user`

| context_mode | n_pairs | estimate | 95% CI |
| --- | ---: | ---: | ---: |
| dept_role | 1039 | 0.002922 | [0.001460, 0.004379] |
| role | 1119 | 0.002075 | [0.000679, 0.003418] |
| dept | 1199 | 0.001155 | [-0.000242, 0.002536] |
| team | 1301 | 0.000662 | [-0.000880, 0.002236] |

Runtime:

- peak GPU memory: `39717 MiB`
- active-average GPU memory: about `23.7 GiB`
- CPU MaxRSS: about `49.6 GiB`

Optional untracked audit CSV:

- `token_delta_sae_necessity_candidate_rows.csv`
- rows: `74528`
- size: about `13.0 MiB`
- rsync source:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_r42_no_same_user/l26_m02_k04_top5_control5_active_necessity_no_same_user/token_delta_sae_necessity_candidate_rows.csv`

## Interpretation

The r4.2 native token features show a positive same-user-excluded necessity
effect for `dept_role` and `role`. The `dept` and `team` rows are directionally
positive but have confidence intervals crossing zero.
