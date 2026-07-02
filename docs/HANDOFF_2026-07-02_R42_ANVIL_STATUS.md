# Handoff 2026-07-02: R4.2 Qwen3-8B Anvil Status

This note records the live Anvil state for the `r4.2` transfer/generalization
branch so the Magnolia agent can prepare the next causal-patching work.

## Current Slurm State

Checked on Anvil at approximately `2026-07-02 18:16 EDT`.

- `18755589 qwen_qlora_ddp`: completed successfully.
- `18755590 token_delta_extract`: completed successfully.
- `18755591 token_delta_sae`: running on `ai` node `h017`.

The running SAE job was at `1:48:37 / 24:00:00` elapsed at the check time.
No errors were present in `logs/token_delta_sae-18755591.err`.

## Completed R4.2 Training

The corrected Qwen3-8B QLoRA transfer run completed with:

- model: `Qwen/Qwen3-8B`
- dataset: `r4.2` session JSONL
- launcher: `scripts/submit_qwen3_8b_r42_targeted_pipeline_anvil.sh`
- training mode: 4-H100 DDP
- effective batch: `MICRO_BS=22`, `GRAD_ACCUM=1`, `NPROC=4`
- attention implementation: `sdpa`
- gradient checkpointing: on
- checkpoint root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on`
- adapter:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter`

The earlier `MICRO_BS=28` run failed from an activation-memory transient, so
`MICRO_BS=22` is the corrected full-run setting for this branch.

## Completed Token-Delta Extraction

The token-delta extraction completed cleanly:

- job: `18755590 token_delta_extract`
- state: `COMPLETED`
- exit code: `0:0`
- runtime: `00:50:45`
- ran: `2026-07-02 00:59:46` to `2026-07-02 01:50:31 EDT`
- split: `eval`
- extracted examples: `42,468`
- layers: `18,26,34`
- pool unit: `token`
- chunk size: `512` examples
- token cache:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on`
- cache size at completion: about `272G`
- file count: `253`

This extraction used a conservative `BATCH_SIZE=4`. It was not VRAM-maxed; it
was chosen for reliability and to avoid the large token-output/chunking issues
seen earlier.

## Running Token-SAE Frontier

The downstream token-SAE frontier job is running:

- job: `18755591 token_delta_sae`
- node: `h017`
- requested resources: `1` H100, `24` CPUs, `240G` RAM
- start time: `2026-07-02 16:28:13 EDT`
- output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on`

At the latest log check, the SAE job had completed all four layer-18 configs
and at least the first two layer-26 configs. It was working on:

- `layer=26`
- `latent_mult=4`
- `topk=4`

The output directory was about `2.1G` at that point.

Expected frontier configs from the launcher are:

- layers: `18,26,34`
- latent multipliers: `2,4`
- top-k values: `4,8`

So the complete frontier should contain 12 configs:

- `layer_18/m02_k04`
- `layer_18/m02_k08`
- `layer_18/m04_k04`
- `layer_18/m04_k08`
- `layer_26/m02_k04`
- `layer_26/m02_k08`
- `layer_26/m04_k04`
- `layer_26/m04_k08`
- `layer_34/m02_k04`
- `layer_34/m02_k08`
- `layer_34/m04_k04`
- `layer_34/m04_k08`

Each completed config should include:

- `delta_sae_model.pt`
- `delta_sae_proxy_selectivity.csv`
- `delta_sae_top_features.csv`

## What Magnolia Can Prepare Now

Do not launch R4.2 causal patching until `18755591` completes, because the
frontier root is still being populated. Magnolia can prepare the run plan and
scripts now using these Anvil roots:

- data:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42`
- adapter:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter`
- token cache:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on`
- frontier:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on`
- recommended causal output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on`

The existing Qwen3-8B token eval bundle is parameterized and can be reused with
R4.2 overrides. Example for the R6.2 headline config transferred to R4.2:

```bash
DATA_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42 \
ADAPTER_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter \
EXTRACT_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on \
FRONTIER_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on \
OUTPUT_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on/l18_m04_k08_top5_control5_active \
BOOTSTRAP_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on/l18_m04_k08_top5_control5_active/bootstrap \
LAYER=18 LATENT_MULT=4 TOPK=8 \
TOP_SETS=top5 CONTROL_SET=control5_active ACTIVE_CONTROL_MIN_FRAC=0.002 \
bash scripts/submit_qwen3_8b_token_eval_bundle_anvil.sh
```

Recommended first R4.2 causal patching configs:

1. `l18_m04_k08_top5_control5_active`
2. `l18_m04_k04_top5_control5_active`

Reason: those are the audited R6.2 Qwen3-8B token-level configs. `l18_m04_k08`
is the conservative headline config; `l18_m04_k04` is the strong upper-bound
config that was later tightened with active controls.

Recommended follow-up transfer-probe configs if the frontier completes cleanly:

1. `l26_m04_k08_top5_control5_active`
2. `l26_m04_k04_top5_control5_active`
3. `l34_m04_k08_top5_control5_active`
4. `l34_m04_k04_top5_control5_active`

These test whether the transferred signal remains concentrated near layer 18
or moves later in Qwen3-8B after training on `r4.2`.

## Commit/Artifact Policy

While the SAE job is running, only this lightweight handoff note is committed.
When `18755591` finishes, commit a compact result bundle rather than the bulky
SAE model files:

- frontier summary CSV/JSON if generated
- per-config `delta_sae_proxy_selectivity.csv`
- per-config `delta_sae_top_features.csv`
- a short R4.2 frontier report

Do not commit the large `.pt` SAE models or the 272G token cache.
