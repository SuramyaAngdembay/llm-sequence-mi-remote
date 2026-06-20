# Slurm Templates

These templates are intentionally cluster-agnostic placeholders.

The remote agent should fill in:

- account
- partition
- qos
- module loads
- scratch paths
- container or conda activation

Expected jobs:

- data serialization
- QLoRA fine-tuning
- delta extraction
- delta-SAE sweep
- causal evaluation

