# Token-Level Phase 5 Findings (2026-06-25)

This is the token-level follow-up requested after the mean-pooled Phase 5 causal patching result was null/slightly negative.

## Status

Completed on Anvil AI/H100:

- token-level adapter-delta extraction for layer 18
- token-level delta-SAE training for the two requested causal configs
- token-level causal patch evaluation for:
  - `layer=18, latent_mult=2, k=8`
  - `layer=18, latent_mult=4, k=4`

The original SAE frontier job ended with a Slurm memory OOM after writing the needed models. This did not invalidate the two requested causal evals; both replacement causal jobs loaded the saved SAE models and completed with exit code `0:0`.

## Extraction

- split: `eval`
- examples: `142,072`
- positive receivers: `70`
- layer: `18`
- unit: token
- chunks: `139`
- token rows: `41,676,708`
- cache size on Anvil: about `236G`

Repo summary files:

- `results/qwen3b_pilot/token_extract/token_extract_summary.json`
- `results/qwen3b_pilot/token_extract/token_chunk_manifest.csv`

Large cache is intentionally not in git:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18/`

## SAE Frontier

The token SAE job wrote these model artifacts before Slurm killed the later frontier point:

- `layer_18/m02_k08/delta_sae_model.pt`
- `layer_18/m04_k04/delta_sae_model.pt`

It also wrote `m02_k04`. It did not complete the full requested sweep summary because the job died while/after the later `m04_k08` point.

Slurm memory:

- failed SAE job: `MaxRSS=251657144K`, requested `240G`, state `OUT_OF_MEMORY`
- causal `m02/k08`: `MaxRSS=63795912K`
- causal `m04/k04`: `MaxRSS=99571344K`

This is why the causal evals fit: the causal script now loads only receiver/donor-relevant token rows, not the full 41.7M-row token cache.

## Causal Eval Results

Both token-level causal evals used:

- examples: `142,072`
- positive receivers: `70`
- loaded token rows: `1,112,836`
- candidate rows: `142,160`
- context modes: `team`, `role`, `project_role`, `dept_role`
- top sets: `top1`, `top3`, `top5`
- control set: `control3`
- alphas: `0.25`, `0.5`, `0.75`, `1.0`

### Best Rows By Config

`l18_m02_k08`:

| context | target | top_repair_advantage | control_repair_advantage | top_minus_control |
|---|---:|---:|---:|---:|
| team | top1 | 0.000406 | -0.000999 | 0.001405 |
| team | top5 | 0.000383 | -0.000999 | 0.001382 |
| team | top3 | -0.000092 | -0.000999 | 0.000907 |
| role | top3 | 0.000247 | 0.000017 | 0.000230 |
| dept_role | top3 | 0.000240 | 0.000073 | 0.000167 |

Selected features:

- top1: `[420]`
- top3: `[420, 2034, 3318]`
- top5: `[420, 2034, 3318, 1406, 1344]`
- control3: `[920, 225, 2018]`

`l18_m04_k04`:

| context | target | top_repair_advantage | control_repair_advantage | top_minus_control |
|---|---:|---:|---:|---:|
| team | top3 | 0.000545 | -0.000790 | 0.001335 |
| project_role | top5 | 0.000342 | -0.000022 | 0.000364 |
| dept_role | top5 | 0.000309 | 0.000014 | 0.000295 |
| role | top3 | 0.000210 | 0.000000 | 0.000209 |
| dept_role | top3 | 0.000180 | 0.000014 | 0.000166 |

Selected features:

- top1: `[4886]`
- top3: `[4886, 1183, 3735]`
- top5: `[4886, 1183, 3735, 737, 5281]`
- control3: `[8043]`

Important caveat: for `m04/k04`, `control3` only contains one feature after filtering, so that control comparison is weaker than the name implies.

## Interpretation

Token-level patching shows small positive top-vs-control repair advantages, strongest in the `team` context:

- `m02/k08`: best `top_minus_control_advantage = +0.001405`
- `m04/k04`: best `top_minus_control_advantage = +0.001335`

This is better than the mean-pooled Phase 5 result, where top-vs-control was mostly negative. However, the absolute effect size is still small. Treat this as weak positive evidence that token-local delta-SAE features are more patchable than mean-pooled features, not as a finished win condition.

## Files

Committed small result files:

- `results/qwen3b_pilot/token_causal/l18_m02_k08/`
- `results/qwen3b_pilot/token_causal/l18_m04_k04/`
- `results/qwen3b_pilot/token_sae/layer_18/m02_k08/delta_sae_proxy_selectivity.csv`
- `results/qwen3b_pilot/token_sae/layer_18/m04_k04/delta_sae_proxy_selectivity.csv`

Large Anvil-only artifacts:

- token cache: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18/`
- token SAE models: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3b/layer_18/`
- full causal rows: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/`

## Recommended Next Evaluation

1. Compare these token-level causal repair numbers against the session-AE baseline under the same receiver/donor protocol.
2. Re-run `m04/k04` with a stronger control pool if control-set parity matters.
3. Add bootstrap confidence intervals over the 70 positive receivers before treating the small advantages as robust.
4. If more token SAE frontier is needed, run only the requested config rather than the full `2,4 x 4,8` sweep, or raise memory above `240G`.
