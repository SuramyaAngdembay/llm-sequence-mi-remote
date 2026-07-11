# Optional Qwen2.5-3B Appendix Path

Status: optional only. Run this after the main `8B` closeout if we still want a
lighter model-scale appendix row for the remote mechanistic table.

## Why This Is Optional

The main paper claim is already carried by the `Qwen3-8B` branch:

- `r6.2`: detector, sufficiency, active-control, and necessity are all strong
- `r4.2`: detector, native sufficiency, active-control, and weaker/partial
  necessity are all present

The `Qwen2.5-3B` branch is useful only if we want to show the `3B -> 8B`
mechanistic scale step more explicitly.

## Fixed Config

Use only the earlier known-positive `r6.2` config:

- `layer=18`
- `latent_mult=2`
- `k=8`
- `top5`
- `control5_active`

Do **not** reopen a broader `3B` sweep.

## Launch

After the detector-metrics rows are queued:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3b_mech_appendix_suite_anvil.sh
```

This submits:

1. `r6.2` `Qwen2.5-3B` active-control causal audit
2. `r6.2` `Qwen2.5-3B` necessity audit

## Component Scripts

- `scripts/submit_qwen3b_active_control_anvil.sh`
- `scripts/submit_qwen3b_token_necessity_bundle_anvil.sh`
- `scripts/submit_qwen3b_mech_appendix_suite_anvil.sh`

## Expected Outputs

Active-control:

- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_best_rows.csv`
- `token_delta_sae_causal_selected_sets.csv`
- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `bootstrap/token_delta_sae_bootstrap_summary.csv`
- `bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`

Necessity:

- `token_delta_sae_necessity_summary.csv`
- `token_delta_sae_necessity_best_rows.csv`
- `token_delta_sae_necessity_selected_sets.csv`
- `TOKEN_DELTA_SAE_NECESSITY_REPORT.md`
- `bootstrap/token_delta_sae_necessity_bootstrap_summary.csv`
- `bootstrap/TOKEN_DELTA_SAE_NECESSITY_BOOTSTRAP_REPORT.md`

## Interpretation

If this appendix run stays positive on:

- `top_minus_control` for active-control sufficiency
- `top_minus_control_necessity` for necessity

then the scale story becomes:

- `3B` already showed the mechanism weakly
- `8B` makes the mechanism much cleaner and stronger

If this appendix run is weak or unstable, the correct writeup is still:

- `8B` headline branch carries the paper
- `3B` remains historical context, not core evidence
