# Qwen3-8B r6.2 Same-User-Excluded Causal Recovery

This directory records the r6.2 token-causal recovery rerun after the July 2026
validity audit found that same-user donor matches were possible in the earlier
causal-patching pipeline.

## Run

- GPU causal job: `19173967`
- CPU bootstrap job: `19173968`
- Output directory:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh_no_same_user/l18_m04_k08_top5_control5_active_no_same_user`
- Tracked result directory:
  `results/qwen3_8b_token_causal/same_user_recovery/l18_m04_k08_top5_control5_active_no_same_user/`
- Dataset: r6.2
- Model branch: Qwen3-8B QLoRA, fresh `mb12` adapter
- Token-SAE config: layer `18`, latent multiplier `4`, top-k `8`
- Intervention: token-level causal patching
- Feature target/control: `top5` vs `control5_active`
- Donors: benign donors and anomalous controls matched by context, with
  same-user donors excluded
- Receiver set: all 70 positive eval receivers for this branch

## Bootstrap Summary

Bootstrap confidence intervals use 4000 receiver-level bootstrap draws.

| context_mode | estimate | 95% CI | top repair advantage | control repair advantage |
| --- | ---: | ---: | ---: | ---: |
| role | 0.006848 | [0.003362, 0.010790] | 0.008625 | 0.001777 |
| dept_role | 0.006818 | [0.003541, 0.010321] | 0.008830 | 0.002012 |
| project_role | 0.004201 | [0.001565, 0.006979] | 0.006116 | 0.001916 |
| team | n/a | n/a | n/a | n/a |

The `team` row has no finite estimate because the same-user-excluded donor
pool did not yield a valid anomalous-control comparison for that context in
this branch.

## Runtime

- Wall time: `01:01:57`
- CPU MaxRSS: about `44.4 GiB`
- Peak GPU memory: `29477 MiB`
- Active-average GPU memory: about `18.3 GiB`

## Interpretation

This rerun reduces the apparent causal-patching effect size relative to the
earlier r6.2 active-control result, but it remains clearly positive for the
main organization-context rows after same-user donors are removed.

The result supports the paper's r6.2 mechanistic claim in a more conservative
form:

- the earlier headline effect was partly inflated by permissive donor matching
- the effect does not disappear under same-user exclusion
- final r6.2 causal tables should use this same-user-excluded result, not the
  older permissive-donor row

