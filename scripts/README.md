# Script Ownership

Planned entry points:

- `build_session_jsonl.py`
  - convert raw LC-DAL session CSV shards into ordered JSONL sequence examples
- `train_qlora.py`
  - fine-tune Qwen with QLoRA
- `extract_adapter_deltas.py`
  - run base and adapted forward passes and store hidden-state deltas
- `train_delta_sae_frontier.py`
  - train SAE frontier on adapter deltas
- `eval_delta_sae_causal.py`
  - top-vs-control ablation, grounding, patching

These files do not exist yet in this directory. This README is here to pin the expected structure before the remote implementation starts.

