# Final Table Row Policy

This file is the shortest paper-safe row policy for the current CERT paper.

## Use

Detector table:

- `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/FOLD_ALIGNED_DETECTOR_REPORT.md`
- `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/FOLD_ALIGNED_DETECTOR_REPORT.md`

Remote mechanistic table:

- `results/qwen3_8b_token_causal/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_token_necessity/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_r42_token_causal/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_r42_token_necessity/same_user_recovery/RESULTS.md`

Remote-vs-local mechanistic comparison:

- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m04_k08_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l26_m02_k04_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

## Do Not Use

Detector headline rows:

- any `results/*/detector_metrics/DETECTOR_METRICS.md`

Historical remote mechanistic headline rows:

- permissive-donor / permissive-match rows superseded by `same_user_recovery/`

Historical compare reports:

- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_*`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_*`
- `results/qwen3b_pilot/strict_compare_remote70_daylevel_controlfix/*`

unless they were explicitly regenerated after commit `ee6a75a` and point to the
same-user-excluded remote summary rows.

## Claim Discipline

Safe claim:

- benign-trained session LLMs recover benchmark-specific causal structure on
  CERT insider-threat benchmarks
- direct token-mechanism transfer can fail
- native rediscovery can succeed

Unsafe claim:

- strong detector superiority on CERT
- universal transferable sparse insider-threat mechanism
