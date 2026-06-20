# Anvil Discovery Checklist

> ✅ **Answered in [ANVIL_SYSTEM_SPECS.md](./ANVIL_SYSTEM_SPECS.md)** (filled by the Anvil-side agent, 2026-06-20).

The remote agent should answer these before we finalize launch scripts.

## Hardware

- exact GPU model
- GPUs per node
- GPU memory per device
- host RAM per node
- local scratch path and size
- parallel filesystem path and quota

## Scheduler

- Slurm partition name
- account/allocation string
- time limits
- interactive GPU policy
- job array policy
- node exclusivity policy

## Environment

- preferred Python setup:
  - module
  - conda
  - micromamba
  - apptainer
- CUDA version
- whether `bitsandbytes` is known to work
- whether internet egress is allowed on compute nodes

## Transfer

- is `scp` allowed directly?
- is `rsync` allowed directly?
- is `Globus` available?
- is there a staging filesystem for large uploads?
- are login nodes suitable for large unpack jobs?

## Storage

- quota for home
- quota for scratch
- purge policy on scratch
- best location for:
  - raw data shards
  - token caches
  - adapter checkpoints
  - delta activations

## Practical Constraints

- maximum recommended walltime per job
- whether long single jobs are discouraged
- whether checkpoint/resume is necessary

