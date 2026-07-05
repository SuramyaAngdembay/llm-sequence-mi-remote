# Qwen3-8B R4.2 Token Causal Results: Streamed Uncapped V2

Status: complete on Anvil, July 5, 2026.

This directory contains the lightweight committed artifacts for the final
uncapped R4.2 token causal suite. The large candidate-row CSVs are intentionally
not committed; exact Anvil paths are listed below for rsync if Magnolia needs a
deeper audit.

## Run Scope

- model branch: Qwen3-8B QLoRA transfer on R4.2
- causal protocol: token-level delta-SAE patching
- layer: `18`
- configs: `l18_m04_k04`, `l18_m02_k04`, `l18_m04_k08`
- context modes: `team,role,dept,dept_role`
- top sets: `top1,top3,top5`
- control set: `control3`
- receiver cap: uncapped (`COMMON_MAX_RECEIVERS=0`)
- positive receivers: `1309`
- token rows: `8870917`
- candidate rows per config: `2666032`
- token delta dtype: `float32`
- patch chunk size: `8`
- output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on_stream_uncapped_v2`

## Headline

All three final-table-valid uncapped R4.2 configs are negative on the
top-minus-control causal metric. Under this run, the Qwen3-8B R4.2 transfer did
not reproduce the positive R6.2 token-causal effect.

Best bootstrap row per config, sorted by estimate:

| Config | Context | Target | Receivers | Estimate | 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| `l18_m04_k04` | `dept` | `top5` | 1309 | -0.000941 | [-0.001167, -0.000719] |
| `l18_m02_k04` | `dept` | `top5` | 1309 | -0.001047 | [-0.001488, -0.000635] |
| `l18_m04_k08` | `dept` | `top3` | 1309 | -0.001132 | [-0.001438, -0.000820] |

Interpretation for Magnolia audit:

- This is the uncapped streamed run, not the earlier receiver-capped probe.
- The memory rewrite was intended to preserve the same causal estimand while
  avoiding host-memory OOM.
- The negative result should be treated as the current final valid R4.2 result
  unless audit finds an implementation or data-alignment issue.

## Slurm Completion

| Job | Config | State | Elapsed | ReqMem | MaxRSS | Node |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| `18836656` | `l18_m04_k04` | `COMPLETED` | `19:26:21` | `480G` | `161448296K` | `h003` |
| `18836658` | `l18_m02_k04` | `COMPLETED` | `18:13:37` | `480G` | `161096088K` | `h019` |
| `18836660` | `l18_m04_k08` | `COMPLETED` | `19:17:14` | `480G` | `162296388K` | `h015` |
| `18836657` | `l18_m04_k04` bootstrap | `COMPLETED` | `00:00:20` | `16G` | `205376K` | `a198` |
| `18836659` | `l18_m02_k04` bootstrap | `COMPLETED` | `00:00:17` | `16G` | `160596K` | `a187` |
| `18836661` | `l18_m04_k08` bootstrap | `COMPLETED` | `00:00:23` | `16G` | `187896K` | `a196` |

## Committed Artifacts

Each config directory includes:

- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_summary.json`
- `token_delta_sae_causal_selected_sets.csv`
- `token_delta_sae_causal_best_rows.csv`
- `bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`
- `bootstrap/token_delta_sae_bootstrap_summary.csv`

## Large Files Left On Anvil

These files are available for rsync if Magnolia needs candidate-level audit
detail:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on_stream_uncapped_v2/l18_m02_k04/token_delta_sae_causal_candidate_rows.csv`
  - size: `433247692` bytes
- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on_stream_uncapped_v2/l18_m04_k04/token_delta_sae_causal_candidate_rows.csv`
  - size: `437253861` bytes
- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on_stream_uncapped_v2/l18_m04_k08/token_delta_sae_causal_candidate_rows.csv`
  - size: `440074234` bytes
