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

The initial benchmark recommendation was `MICRO_BS=28`, `GRAD_ACCUM=1`,
`GC_MODE=on`, because it was closest to the reserved-VRAM target in the short
probe.

The `micro_bs=32` probe also cleared and had the highest short-run throughput,
but its measured reserved memory was lower because the benchmark batches after
warmup had shorter average dynamic-padding lengths.

## Full-Training Correction

The first full `r4.2` training attempt with `MICRO_BS=28` failed at step
`55/2570` with CUDA OOM:

- failed job: `18733129`
- node: `h009`
- PyTorch allocated on failing rank: `66.57 GiB`
- total process memory in use on failing rank: `68.89 GiB`
- free memory before failure: `10.28 GiB`
- attempted additional allocation: `18.88 GiB`

This means the benchmark measured the steady-state region but did not capture a
late full-training transient allocation. The corrected full-run setting is:

- `MICRO_BS=22`
- `GRAD_ACCUM=1`
- `GC_MODE=on`

`MICRO_BS=22` is intentionally below the `24` probe result to keep extra
headroom while preserving better utilization than the earlier `12` setting.
