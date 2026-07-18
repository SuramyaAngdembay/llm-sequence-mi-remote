# Qwen3-8B r6.2 Same-User-Excluded Necessity Recovery

This directory records the r6.2 token-necessity recovery run after the
same-user match validity audit.

## Run

- GPU job: `19173969`
- CPU bootstrap job: `19173971`
- Status: completed
- Runtime: `00:12:04`
- Dataset: r6.2
- Model branch: Qwen3-8B QLoRA fresh `mb12` adapter
- Token-SAE config: layer `18`, latent multiplier `4`, top-k `8`
- Intervention: token-level necessity ablation
- Feature target/control: `top5` vs `control5_active`
- Same-user benign matches excluded: yes
- Source output:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_no_same_user/l18_m04_k08_top5_control5_active_necessity_no_same_user`

| context_mode | n_pairs | estimate | 95% CI |
| --- | ---: | ---: | ---: |
| project_role | 70 | 0.065188 | [0.055145, 0.075023] |
| role | 70 | 0.062167 | [0.050265, 0.073364] |
| dept_role | 70 | 0.056603 | [0.044996, 0.068990] |
| team | 70 | 0.052234 | [0.042054, 0.061397] |

Runtime:

- peak GPU memory: `25517 MiB`
- active-average GPU memory: about `16.5 GiB`
- CPU MaxRSS: about `26.7 GiB`

The full candidate-row CSV is committed in this directory because it is small
enough for normal GitHub review.

## Interpretation

The r6.2 token branch remains strongly necessity-positive after same-user benign
matches are removed. This closes the same-user leakage concern for the r6.2
necessity branch and supports using the same-user-excluded row as the final
paper-safe necessity result.
