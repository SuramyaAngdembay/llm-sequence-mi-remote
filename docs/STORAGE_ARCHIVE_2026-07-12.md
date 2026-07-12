# Storage Archive - 2026-07-12

This records stale `cert-qlora-MI` artifacts moved out of Anvil project storage
before the CERT recovery reruns.

## Archive Location

Scratch archive root:

```text
/anvil/scratch/x-sangdembay/cert_qlora_archives/20260712_stale_pre_recovery
```

Files:

```text
qwen3b_stale.tar
qwen3b_stale.contents.txt
old_r62_8b_superseded.tar
old_r62_8b_superseded.contents.txt
archive_manifest_20260712.txt
```

Important: scratch is better than raw directory transfer because the scratch file
quota is also tight. These archives use a small number of files rather than
recreating hundreds of project files on scratch.

## Archived Paths

`qwen3b_stale.tar` contains:

```text
cert-qlora-MI/token_delta_cache/qwen3b_session_token_deltas_l18
cert-qlora-MI/delta_cache/qwen3b_session_mean_deltas
cert-qlora-MI/outputs/delta_sae_frontier_qwen3b
cert-qlora-MI/outputs/delta_sae_causal_qwen3b
cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b
cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3b
cert-qlora-MI/checkpoints/qwen3b_session_qlora_ddp
```

`old_r62_8b_superseded.tar` contains:

```text
cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh
cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp
```

These were removed from project storage after tar creation and tar-listing
validation succeeded.

## Deliberately Not Archived

Active/current recovery paths were left in project storage:

```text
cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2
cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on
cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh
cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on
cert-qlora-MI/detector_score_cache
cert-qlora-MI/outputs/session_jsonl
cert-qlora-MI/outputs/session_jsonl_r42
```

## Restore Commands

Restore from `/anvil/projects/x-cis230270/x-sangdembay`:

```bash
cd /anvil/projects/x-cis230270/x-sangdembay
tar -xf /anvil/scratch/x-sangdembay/cert_qlora_archives/20260712_stale_pre_recovery/qwen3b_stale.tar
tar -xf /anvil/scratch/x-sangdembay/cert_qlora_archives/20260712_stale_pre_recovery/old_r62_8b_superseded.tar
```

## Cleanup Manifest

The exact command output and quota snapshot are in:

```text
/anvil/scratch/x-sangdembay/cert_qlora_archives/20260712_stale_pre_recovery/archive_manifest_20260712.txt
```

Scratch should not be treated as a permanent archive. If these artifacts become
important again, restore them or move the tarballs to a permanent storage target.

