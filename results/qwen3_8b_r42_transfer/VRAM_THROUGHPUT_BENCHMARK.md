# Qwen3-8B R4.2 Throughput/VRAM Benchmark

This benchmark was run before full `r4.2` transfer training to choose a
microbatch that uses the H100 more effectively without OOM.

## Data Build

R4.2 session JSONL was built with the fast builder:

- input: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2`
- output: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42`
- builder: `build_session_jsonl_fast`
- examples: `330,295`
- train: `287,827`
- val: `27,026`
- eval: `15,442`

## Benchmark Setup

- model: `Qwen/Qwen3-8B`
- training stack: 4-bit NF4 QLoRA, bf16 compute, gradient checkpointing on
- hardware: one H100 80 GB
- sample mode: `longest`
- warmup steps: `2`
- benchmark steps: `8`
- target: about `70 GiB` reserved without OOM

## Results

First low-range probe:

| microbatch | status | peak reserved GiB | tokens/sec |
|---:|---|---:|---:|
| `12` | ok | `33.963` | `916.718` |
| `13` | ok | `36.074` | `922.218` |
| `14` | ok | `38.098` | `929.773` |

Second high-range probe:

| microbatch | status | peak reserved GiB | tokens/sec |
|---:|---|---:|---:|
| `20` | ok | `50.449` | `927.042` |
| `24` | ok | `58.674` | `924.666` |
| `28` | ok | `66.611` | `922.881` |
| `32` | ok | `65.961` | `955.737` |

No candidate OOMed.

## Training Choice

Use `MICRO_BS=28`, `GRAD_ACCUM=1`, `GC_MODE=on` for the first full `r4.2`
training run. It is the benchmark recommendation because it is closest to the
reserved-VRAM target while still leaving H100 headroom.

The `micro_bs=32` probe also cleared and had the highest short-run throughput,
but its measured reserved memory was lower because the benchmark batches after
warmup had shorter average dynamic-padding lengths. Start with `28` for the
full run and revisit `32` only after the first transfer pass is complete.
