# Coordination Protocol

This branch is now coordinated through three things:

1. the GitHub repo
2. the transfer package manifest
3. the Anvil-side transfer receipt

This avoids ad hoc terminal-only coordination.

## Source Of Truth

### Code / configs / docs

- GitHub repo:
  - `SuramyaAngdembay/llm-sequence-mi-remote`

### Dataset payload

- package directory:
  - `artifacts/transfer_package/`
- package manifest:
  - `manifest.json`
- package checksums:
  - `sha256sums.txt`

### Remote transfer confirmation

- generated on Anvil after verification:
  - `transfer_receipt.json`

## Role Split

### Source side

- build package
- initiate transfer
- do not mutate file names after checksums are generated

### Anvil side

- receive package in:
  - `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/`
- verify checksums
- write `transfer_receipt.json`
- create runtime directories for training outputs

## Standard Flow

1. Pull latest repo on both sides.
2. Source side builds `artifacts/transfer_package/`.
3. Source side transfers package with either:
   - `rsync`
   - `Globus`
4. Anvil side runs:
   - `bash scripts/anvil_verify_and_stage_transfer.sh`
5. Anvil side confirms:
   - checksums passed
   - receipt written
   - project directories created

## Coordination Rule

Do not start Phase 1 JSONL building on Anvil until:

- `sha256sum -c sha256sums.txt` passes
- `transfer_receipt.json` exists

## Minimal Human Message Protocol

When coordinating across chat/agents, use these exact states:

- `PACKAGE_READY`
- `TRANSFER_STARTED`
- `TRANSFER_COMPLETE_UNVERIFIED`
- `TRANSFER_VERIFIED`
- `READY_FOR_JSONL`

That keeps the state unambiguous.

