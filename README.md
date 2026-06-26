# LLM Sequence MI Remote Branch

This directory is the handoff/control-plane for the next branch:

- `QLoRA` fine-tuning on structured `r6.2` session sequences
- adapter-delta extraction
- delta-SAE training
- causal evaluation against the current session AE baseline

This branch is intentionally separate from the local CPU `AE`/`SAE` pipeline. The goal is to test whether a sequence-native model can produce cleaner, more repair-capable circuit units than the current session LC-DAL autoencoder line.

## Current Recommendation

- Keep the completed `Qwen 3B` token-level branch as the current reference run
- The next scale-up is a **targeted `Qwen 7B` token-level branch**
- Use `4x H100 80GB` on one Anvil `ai` node for training
- Use `1x H100 80GB` for token extraction and causal eval follow-ons
- Use structured session sequences, not flattened prose
- Fine-tune with `QLoRA`
- Train SAEs on `adapter deltas`, not full hidden states

## Why This Exists

The current strongest local MI result is the base LC-DAL session branch:

- good detection
- strongest family-level mechanism around `usb_activity`
- family-level repair beats residual baseline

But sparse session SAE units still failed as repair-capable patch units. This remote branch is the next major escalation.

The current remote result has now moved beyond a pure pilot:

- token-level `Qwen 3B` patching beat the matched local session-AE day-level baseline
- the strongest remote token configs have positive bootstrap intervals
- the branch decision is now to scale the **same token-level intervention path**, not to switch representations again first

## Directory Layout

- [docs](./docs): specs, runbook, cluster checklist
- [configs](./configs): model and SAE configs
- [slurm](./slurm): batch templates for remote execution
- [scripts](./scripts): runnable entry points plus ownership notes
- [manifests](./manifests): artifact manifests and transfer targets

## Current Status

- GitHub repo is live and being used as the control plane for the Anvil branch.
- First runnable execution path exists:
  1. `build_session_jsonl.py`
  2. `train_qlora.py`
  3. `extract_adapter_deltas.py`
  4. `train_delta_sae_frontier.py`
- The `Qwen 3B` token-level branch is complete and currently ahead of the local matched baseline.
- The next branch to run is a **targeted `Qwen 7B` token-level scale-up**, not a broad new frontier or a graph-first pivot.

## Local Source References

Primary local source files:

- raw session extract:
  - `/homes/01/srangdembay/InsiderThreatDetection/r6.2/lcdal-r62-full/extract_stage/r6.2/ExtractedData/sessionr6.2.csv`
- prepared session benchmark stats:
  - `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_lcdal_session_features_clean/sessionr6.2_prepare_stats.json`
- current primary MI memo:
  - `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/reports/SESSION_PRIMARY_MI_DECISION_MEMO_2026-06-19.md`
