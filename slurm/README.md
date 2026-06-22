# Slurm Templates

These templates are now Anvil-oriented defaults for the first `Qwen 3B` pilot.

Current job sequence:

1. `build_jsonl.template.sbatch`
2. `train_qlora.template.sbatch`
3. `extract_deltas.template.sbatch`
4. `train_delta_sae.template.sbatch`

Token-level escalation sequence:

1. `extract_token_deltas.template.sbatch`
2. `train_token_delta_sae.template.sbatch`
3. `eval_token_delta_sae_causal.template.sbatch`

All paths are parameterized through environment variables so the Anvil-side agent
can override them without patching the files.
