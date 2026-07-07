# Token-Causal VRAM Probe

Status: completed on Anvil, 2026-07-06/07.

This probe was created after the `r4.2` native token-causal evaluator looked
underutilized at the original `BATCH_SIZE=8`.

The probe measures the Qwen3-8B token-causal forward-hook scoring path on real
`r4.2` eval texts. It records CUDA peak allocated/reserved memory, rough
throughput, and process RSS. It is not a replacement for the full causal
evaluator, but it is useful for choosing safe scoring batch size.

## H100 / AI

Probe output:

- `ai/token_causal_vram_probe.csv`
- `ai/recommendation.json`

Key H100 rows:

| batch size | status | peak reserved GiB | tokens/sec |
| ---: | --- | ---: | ---: |
| `8` | ok | `20.447` | `162.55` |
| `16` | ok | `34.957` | `241.72` |
| `24` | ok | `49.332` | `453.05` |
| `28` | ok | `56.502` | `508.08` |
| `32` | ok in probe | `63.533` | `535.60` |

The full causal evaluator later OOMed at `BATCH_SIZE=32` during the score-loss
step, because the short probe did not capture the full evaluator's worst-case
cross-entropy allocation. The successful full rerun used:

- `BATCH_SIZE=24`
- `PATCH_CHUNK_SIZE=24`

## A100 / GPU

Probe output:

- `gpu/token_causal_vram_probe.csv`
- `gpu/recommendation.json`

Key A100 rows:

| batch size | status | peak reserved GiB | tokens/sec |
| ---: | --- | ---: | ---: |
| `4` | ok | `13.160` | `73.97` |
| `8` | ok | `20.428` | `146.31` |
| `12` | ok | `27.711` | `221.00` |
| `16` | ok | `34.938` | `281.06` |
| `20` | CUDA OOM | `32.363` | `0.00` |
| `24` | CUDA OOM | `37.617` | `0.00` |

Recommendation:

- H100 full causal scoring: use `BATCH_SIZE=24` unless the evaluator is
  changed to avoid the large loss-allocation spike.
- A100 full causal scoring: start at `BATCH_SIZE=16`.
