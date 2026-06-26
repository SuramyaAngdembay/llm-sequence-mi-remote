# Handoff 2026-06-25: Token-Level Delta-SAE Causal Eval

This handoff completes the token-level escalation requested after the mean-pooled Phase 5 causal eval did not demonstrate patchability.

## What Completed

- `18554886 tok_extract_l18`: completed, layer-18 token deltas.
- `18554888 tok_sae_l18`: Slurm `OUT_OF_MEMORY`, but after saving the required SAE models for `m02/k08` and `m04/k04`.
- `18576713 tok_causal_m02k08_r2`: completed, exit `0:0`.
- `18576714 tok_causal_m04k04_r2`: completed, exit `0:0`.

The old dependency-blocked causal jobs were replaced because the SAE parent job ended nonzero after useful artifacts had already been written.

## Main Result

Token-level causal patching gives positive top-vs-control repair advantages:

- `layer=18, latent_mult=2, k=8`: best `+0.001405`, `team/top1`
- `layer=18, latent_mult=4, k=4`: best `+0.001335`, `team/top3`

This was the first sign that token-level intervention was better than mean-pooled patching.

## Superseding Update

This handoff is now partially superseded by:

- `docs/HANDOFF_2026-06-25_STRICT_COMPARISON.md`
- `results/qwen3b_pilot/strict_compare_remote70_daylevel/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

The updated state is:

- `m02/k08` bootstrap is now complete and the best `team` targets remain cleanly positive
- the strict receiver-day comparison against the local session branch is now done
- on the matched `(user_id, day_index)` unit, the remote token branch is currently ahead of the local session-AE branch

Matched-unit comparison:

- best local adaptive session-AE day-level advantage: `0.001133`
- best local residual day-level advantage: `0.000654`
- best remote token `m02/k08` advantage: `0.001405`

Bootstrap on the best remote config:

- `team/top1`: `0.001405`, CI `[0.000706, 0.002114]`
- `team/top5`: `0.001382`, CI `[0.000729, 0.002059]`
- `team/top3`: `0.000907`, CI `[0.000248, 0.001541]`

So this branch is no longer just “promising but unproven.” It is now the strongest current challenger.

## Final Control-Fix Update

The `m04/k04_controlfix` rerun and bootstrap are now complete:

- `control3` stayed at three features: `[101, 173, 230]`
- best row: `team/top3`, estimate `0.001446`, 95% CI `[0.000623, 0.002328]`
- next strongest row: `project_role/top5`, estimate `0.000749`, 95% CI `[0.000306, 0.001207]`

See:

- `results/qwen3b_pilot/CONTROLFIX_EVALUATION_HANDOFF.md`
- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `results/qwen3b_pilot/token_causal/l18_m04_k04_controlfix/bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`

## Repo Results

Start here:

- `results/qwen3b_pilot/TOKEN_PHASE5_FINDINGS.md`
- `results/qwen3b_pilot/token_causal/l18_m02_k08/TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `results/qwen3b_pilot/token_causal/l18_m04_k04/TOKEN_DELTA_SAE_CAUSAL_REPORT.md`

## Anvil Artifacts

Large files are intentionally Anvil-only:

- token deltas: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18/`
- SAE models: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3b/layer_18/`
- full causal candidate/best rows: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/`

## Code Changes Needed To Reproduce

- `extract_adapter_deltas.py` accepts `--layers`, so layer 18 can be extracted without materializing layers 12/24.
- token SAE loading samples while reading chunks, instead of loading the full token layer into RAM before applying `MAX_ROWS`.
- token causal eval first builds candidate receiver/donor pairs, then loads only token rows for those examples.
- token Slurm defaults now point at the layer-18 project token cache and expose `LAYERS`, `LATENT_MULTIPLIERS`, and `TOPK_VALUES`.

## Next Agent Tasks

1. Pull the latest GitHub commit on Magnolia.
2. Regenerate the strict receiver-day comparison for `l18_m04_k04_controlfix` on Magnolia, where the local session-AE artifacts live.
3. If the refreshed comparison agrees with the existing matched comparison, keep the token QLoRA branch ahead of graph-first escalation.
