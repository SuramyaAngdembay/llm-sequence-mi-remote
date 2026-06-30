# Handoff: `r4.2` Remote Transfer And First Remote Run

This note records the current state of the `r4.2` remote-transfer branch and
the minimum context needed for the Anvil side to continue without rediscovering
the setup.

## Transfer Status

The `r4.2` transfer package has already been prepared on Magnolia and pushed to
Anvil.

Dataset destination on Anvil:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2/`

Transferred files:

- `sessionr4.2_shard_000.csv.gz`
- `sessionr4.2_shard_001.csv.gz`
- `labels_daily.parquet`
- `sessionr4.2_user_map.csv`
- `manifest.json`
- `sha256sums.txt`

The reason for using the dedicated `r4.2` subdirectory is to avoid mixing its
`manifest.json` and `sha256sums.txt` with the existing `r6.2` package at:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/`

## First Verification Step On Anvil

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main

cd /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2/
sha256sum -c sha256sums.txt
```

If those checks pass, treat the transfer as ready.

## What This `r4.2` Branch Is For

This is the first transfer/generalization check for the current winning remote
mechanistic path.

The question is not:

- can we make a better detector by trying many new architectures?

The question is:

- does the current winning token-level remote causal path survive a second CERT
  version?

## Major Parameters To Keep Fixed

Keep these high-level choices aligned with the successful `r6.2` remote branch:

- use structured **session JSONL**, not flattened prose
- use the same **benign-only** QLoRA training objective
- keep the branch **token-level**, not mean-pooled
- keep the same **matched donor / receiver** evaluation logic
- keep the same **top-vs-control** causal protocol
- keep the same **control-pool fix** that avoided inert `control3` sets

## Model Direction

For `r4.2`, use the **Qwen3 token branch** as the remote method family.

Do not reopen:

- broad mean-pooled delta runs
- broad graph detours
- large exploratory SAE sweeps before checking the targeted path

The exact model size can be adjusted on Anvil if needed, but the intended path
is still:

- QLoRA
- token delta extraction
- token-SAE frontier
- token causal patching

## Suggested First Remote Sequence

1. Build `r4.2` session JSONL from the transferred shards.
2. Run the first `Qwen3` QLoRA training pass.
3. Extract **token-level** deltas on the selected layer band.
4. Run the **targeted** token-SAE frontier.
5. Run token-level causal patching on the best few configs.

## Baseline Comparison Target

The remote `r4.2` branch is meant to compare against the compact baseline set
already fixed in:

- [R42_REMOTE_TRANSFER_AND_BASELINES.md](./R42_REMOTE_TRANSFER_AND_BASELINES.md)

Priority local baselines for the `r4.2` table:

- local session AE
- LSTM AE
- GRU AE
- Deep SVDD
- Isolation Forest

`MCM` is optional for `r4.2`, not required for the first transfer pass.

## Interpretation Rule

The `r4.2` remote branch is only a win if it preserves the same kind of
mechanistic result we now have on `r6.2`:

- positive token-level causal sufficiency
- nontrivial control comparison
- competitive or better matched-unit comparison against the best local baseline

The goal is not only better likelihood fit. The goal is to test whether the
mechanistic conclusion transfers.
