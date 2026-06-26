# Control-Fixed Token Causal Evaluation Handoff

Date: 2026-06-26

This note evaluates the Anvil `m04/k04_controlfix` run and leaves the Magnolia-side
strict comparison to the Magnolia agent.

## Completed Anvil Runs

- Causal eval: Slurm `18590322`, `tok_causal_m04k04_controlfix_b16`, completed on `ai/h019`
- Bootstrap: Slurm `18590324`, `tok_boot_m04k04_controlfix_b16`, completed on `shared/a008`
- Duplicate A100 attempt was canceled after the H100 run finished.

Tracked result files:

- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/token_delta_sae_causal_summary.csv`
- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/token_delta_sae_causal_selected_sets.csv`
- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/token_delta_sae_causal_best_rows.csv`
- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`
- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/bootstrap/token_delta_sae_bootstrap_summary.csv`

## Anvil Result Read

The control-fix did what it was supposed to do:

- `control3` no longer collapses to one feature.
- `control3` features are `[101, 173, 230]`.
- `control3` mean row gap is `0.0`.

Top causal rows by `top_minus_control_advantage`:

| rank | context | target | advantage |
|---:|:---|:---|---:|
| 1 | `team` | `top3` | `0.001446` |
| 2 | `project_role` | `top5` | `0.000749` |
| 3 | `team` | `top5` | `0.000494` |
| 4 | `dept_role` | `top3` | `0.000462` |
| 5 | `dept_role` | `top5` | `0.000397` |

Bootstrap rows with positive 95% CIs:

| context | target | estimate | 95% CI |
|:---|:---|---:|:---|
| `team` | `top3` | `0.001446` | `[0.000623, 0.002328]` |
| `project_role` | `top5` | `0.000749` | `[0.000306, 0.001207]` |
| `dept_role` | `top3` | `0.000462` | `[0.000053, 0.000888]` |

This means the control-fixed `m04/k04` run agrees with the earlier `m02/k08`
bootstrap result: the token-level branch remains positive after fixing control-set
quality.

## Current Branch Decision

The existing matched receiver-day comparison already put the token branch ahead of
the local session-AE branch:

- best local adaptive session-AE day-level advantage: `0.001133`
- best local residual day-level advantage: `0.000654`
- best token `m02/k08` advantage: `0.001405`

The new control-fixed `m04/k04` best row is `0.001446`, with a positive bootstrap
CI. That is slightly above the previous `m02/k08` matched token number, but it
still needs the Magnolia-side strict comparison to be regenerated on Magnolia
against the local session-AE artifacts.

Recommended interpretation until that Magnolia comparison is refreshed:

- Keep the token QLoRA branch as the main challenger.
- Do not escalate graph-first based on current evidence.
- Treat effect sizes as small, so the next useful work is direct replication,
  baseline robustness, and matched comparison refresh rather than a broad new sweep.

## Magnolia Agent Task

After pulling this GitHub commit on Magnolia, run the strict comparison there using
the control-fixed token summary:

```bash
python3 scripts/compare_remote_token_vs_local_session.py \
  --session-feature-path /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_lcdal_session_features_clean/sessionr6.2_features.parquet \
  --local-run-root /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_session_lcdal_autoencoder_mech_clean/plain \
  --local-adaptive-best-rows /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_session_lcdal_feature_bundle_repair_adaptive_remote70/adaptive_feature_bundle_repair_best_rows.csv \
  --local-residual-best-rows /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_session_lcdal_feature_bundle_repair_residual_remote70/residual_feature_bundle_repair_best_rows.csv \
  --remote-summary-csv results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/token_delta_sae_causal_summary.csv \
  --out-dir results/qwen3b_pilot/strict_compare_remote70_daylevel_controlfix
```

The desired Magnolia-side output is a refreshed
`results/qwen3b_pilot/strict_compare_remote70_daylevel_controlfix/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`
so both agents can compare the control-fixed token run against the session-AE
branch on the same receiver-day unit.
