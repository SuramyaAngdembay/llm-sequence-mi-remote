# Qwen3-8B Token Frontier Handoff

This is the Anvil result bundle for the Qwen3-8B token-level targeted scale-up.
It supersedes the older `qwen3b_pilot` frontier for the next Magnolia-side eval.

## Completed Runs

- QLoRA training: Slurm `18649521`, completed, final epoch `1.0`
- token extraction retry: Slurm `18688748`, completed
- first SAE retry: Slurm `18688749`, failed with CPU RAM OOM after the old dense eval path
- streamed SAE retry: Slurm `18695458`, completed, exit `0:0`, elapsed `03:42:04`

The OOM fix is commit `4ea4d0b` (`Stream delta SAE eval to reduce CPU memory`).
The successful retry peaked at about `105474232K` RSS under a `360G` request.

## Git Bundle

Committed lightweight eval files:

- `token_extract/token_extract_summary.json`
- `token_extract/token_chunk_manifest.csv`
- `token_sae/delta_sae_frontier_summary.csv`
- `token_sae/delta_sae_frontier_summary.json`
- `token_sae/DELTA_SAE_FRONTIER_REPORT.md`
- per-config `delta_sae_top_features.csv`
- per-config `delta_sae_proxy_selectivity.csv`

Raw `.pt` SAE models and token chunks are intentionally not in git.

## Anvil Runtime Paths

- adapter:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_ddp_mb12_gc_on_fresh/adapter`
- token cache:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2`
- SAE frontier:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream`

The token cache is about `956G`. The SAE frontier directory is about `4.6G`.

## Frontier Grid

The completed grid is:

- layers: `18,26,34`
- latent multipliers: `2,4`
- top-k: `4,8`
- rows per config: `2,000,000`

Top configs by `top5_minus_control3_advantage_proxy`:

| layer | latent_mult | k | recon_mse | top10_row_gap_mean | top5-control3 proxy |
|---:|---:|---:|---:|---:|---:|
| 18 | 4 | 4 | 0.077840 | 0.093653 | 0.028509 |
| 18 | 2 | 4 | 0.074744 | 0.085286 | 0.014818 |
| 18 | 4 | 8 | 0.054468 | 0.086540 | 0.014153 |
| 34 | 4 | 4 | 0.094887 | 0.095607 | 0.012530 |
| 34 | 4 | 8 | 0.072461 | 0.097798 | 0.011513 |

## Recommended Magnolia Eval

For causal eval/strict comparison, start with:

1. `layer=18`, `latent_mult=4`, `k=4`
2. `layer=18`, `latent_mult=2`, `k=4`
3. `layer=18`, `latent_mult=4`, `k=8`

Do not interpret the frontier proxy metrics as causal patchability by themselves.
They are selection metrics for the next token-level causal patching run.

## Large Model Files

Model files for the recommended first configs:

- `layer_18/m04_k04/delta_sae_model.pt` - `536985100` bytes
- `layer_18/m02_k04/delta_sae_model.pt` - `268516876` bytes
- `layer_18/m04_k08/delta_sae_model.pt` - `536985100` bytes

All model files live under:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_mb12_gc_on_fresh_v3_stream`
