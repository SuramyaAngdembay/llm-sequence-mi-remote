# Environment Spec

## Recommended Hardware

### Primary target

- `1x H100 80GB`

This is the recommended first serious run target.

### Acceptable alternatives

- `1x A100 80GB`
- `1x A100 40GB` for smoke or reduced-context pilot
- `1x RTX 3090 24GB` only for small smoke tests

## Model Sizing Guidance

### First model

- `Qwen 3B`

Why:

- much lower iteration cost than `7B`
- enough capacity to test whether sequence-native delta features beat the current session AE branch
- better first answer to the research question than jumping immediately to a larger model

### Second model only if the first works

- `Qwen 7B`

Do not start here unless:

- the remote environment is already stable
- tokenization/cache logic is already proven
- delta extraction pipeline already works on `3B`

## Training Mode

- frozen base model
- `4-bit` quantized base
- `LoRA` adapters in `bf16` or `fp16`
- benign-only objective for the first anomaly-detection-compatible branch

## Software Stack

Expected stack:

- Python `3.10` or `3.11`
- `torch`
- `transformers`
- `peft`
- `accelerate`
- `bitsandbytes`
- `datasets`
- `pyarrow`
- `pandas`
- `scikit-learn`
- `safetensors`

If Anvil prefers containerized workflows, use:

- `Apptainer`/`Singularity` image

Otherwise:

- `conda` or `micromamba`

## Disk and RAM Expectations

Minimum working assumptions for the first serious run:

- local scratch: `>= 500GB`
- system RAM: `>= 128GB`
- better if `>= 256GB`

Why:

- token caches
- sequence shards
- adapter checkpointing
- delta activation dumps

## Runtime Expectations

For a `Qwen 3B` pilot on H100:

- QLoRA training:
  - roughly `6-12` hours for a small/medium pilot
- delta extraction:
  - roughly `2-6` hours depending on layers and sample count
- delta-SAE:
  - roughly `4-12` hours depending on sweep size

If the full first-pass design creeps beyond about `48` hours, reduce:

- context length
- number of hooked layers
- number of training examples

before scaling up model size.

