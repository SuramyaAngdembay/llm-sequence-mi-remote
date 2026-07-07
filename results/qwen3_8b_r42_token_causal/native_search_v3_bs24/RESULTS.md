# Qwen3-8B R4.2 Native Token-Causal Search

Status: completed on Anvil, 2026-07-07.

This is the first `r4.2`-native remote token-causal search after the direct
`r6.2` mechanism transfer failed on `r4.2`.

## Setup

- model branch: `Qwen3-8B` r4.2 QLoRA adapter
- token delta cache:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on`
- frontier root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on`
- causal output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_search_v3_bs24`
- evaluator: full uncapped streamed token causal evaluator
- context modes: `team,role,dept,dept_role`
- receivers: all `1309` positive eval receiver-days
- candidate rows per config: `2,666,032`
- final runtime setting: `BATCH_SIZE=24`, `PATCH_CHUNK_SIZE=24`

## Why Batch 24

A short token-causal VRAM probe showed:

- H100/AI:
  - `batch_size=32` cleared in the short forward-hook probe at about
    `63.5 GiB` reserved
  - full causal `batch_size=32` later OOMed in the score-loss step when
    cross-entropy needed another about `21.4 GiB`
  - `batch_size=24` completed all full causal runs
- A100/GPU:
  - `batch_size=16` was the largest safe probe setting
  - `batch_size=20` and `24` OOMed

The committed probe artifacts are under:

- `results/qwen3_8b_r42_token_causal/vram_probe/`

## Results

| config | best context | target | estimate | CI low | CI high | read |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `l26_m02_k04` | `team` | `top5` | `0.001307` | `0.000960` | `0.001663` | positive |
| `l34_m04_k04` | `dept_role` | `top5` | `0.000119` | `-0.000038` | `0.000280` | weak/null |
| `l26_m04_k04` | `dept` | `top3` | `0.000054` | `-0.000149` | `0.000269` | null |
| `l26_m04_k08` | `role` | `top5` | `-0.000261` | `-0.000473` | `-0.000046` | negative |

Best positive row:

- `l26_m02_k04 / team / top5`
- estimate `0.0013066`
- CI `[0.0009597, 0.0016633]`
- `n_receivers=1309`

Other positive `l26_m02_k04 / top5` rows:

- `dept`: estimate `0.001186`, CI `[0.000896, 0.001484]`
- `dept_role`: estimate `0.001114`, CI `[0.000793, 0.001440]`
- `role`: estimate `0.001056`, CI `[0.000744, 0.001393]`

## Comparison

Earlier direct-transfer `r4.2` token-causal configs were negative:

- `l18_m04_k08`: about `-0.001132`
- `l18_m04_k04`: about `-0.000941`
- `l18_m02_k04`: about `-0.001047`

Local `r4.2` session-AE comparator on the same receiver-days:

- best local adaptive day-level advantage: about `+0.000986`
- best local residual day-level advantage: about `+0.001380`

Interpretation:

- direct transfer of the `r6.2` remote token mechanism still failed
- `r4.2` does have a positive native remote token-causal config
- the positive native config is `l26_m02_k04`
- its best estimate is competitive with the local `r4.2` session-AE scale,
  but does not clearly exceed the strongest local residual comparator
- the effect is much smaller than the `r6.2` remote token-causal headline

## Audit Notes

The full candidate-row CSVs were not committed because they are about
`413-419 MB` per config. They remain on Anvil at:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_search_v3_bs24/*/token_delta_sae_causal_candidate_rows.csv`

Commit includes:

- causal reports
- selected feature sets
- summary CSV/JSON
- best-row CSVs
- bootstrap reports
- bootstrap summary CSVs

If Magnolia needs deeper audit of donor matching or active-token support, rsync
the candidate-row CSV for `l26_m02_k04` first.
