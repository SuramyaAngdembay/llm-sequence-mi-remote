# Script Ownership

Implemented entry points:

- `build_session_jsonl.py`
  - convert raw LC-DAL session CSV shards into ordered JSONL sequence examples
- `train_qlora.py`
  - fine-tune Qwen with QLoRA on `train.jsonl` / `val.jsonl`
- `extract_adapter_deltas.py`
  - run base and adapted forward passes and store adapter deltas
- `train_delta_sae_frontier.py`
  - train the runnable SAE frontier on extracted deltas (mean or token unit)
- `eval_delta_sae_causal.py`
  - first model-level causal patch/repair eval on mean-pooled adapter deltas
- `eval_token_delta_sae_causal.py`
  - token-local sparse patch/repair eval on token-level adapter deltas
- `prepare_transfer_package.py`
  - build the Anvil transfer package
- `rsync_to_anvil.sh`
  - send the package to Anvil
- `anvil_verify_and_stage_transfer.sh`
  - verify checksums and create runtime directories on Anvil
- `submit_qwen3b_pipeline_anvil.sh`
  - submit the first end-to-end Anvil dependency chain

Shared helpers:

- `remote_common.py`
- `sae_core.py`

Still missing for a full end-to-end MI branch:

- sparse grounding from delta-SAE features back to token/session factors
- heldout-user fold orchestration matching the local benchmark exactly
