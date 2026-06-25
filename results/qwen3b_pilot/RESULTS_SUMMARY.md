# Qwen2.5-3B Session QLoRA -> Delta-SAE: Pilot Results (2026-06-22)

End-to-end pilot on Anvil (4x H100). All artifacts from the repo pipeline.

## Phase 2 - QLoRA training
- Qwen2.5-3B, 4-bit NF4, LoRA r=16, 1 epoch on 1,251,225 benign user-days (4-GPU DDP, eff batch 32).
- train_loss 0.143 (converged early ~0.13 then plateaued -> 1 epoch was more than needed for this low-entropy data).
- Adapter: checkpoints/qwen3b_session_qlora_ddp/adapter/ (114 MB). Full loss curve in the slurm log.

## Phase 3 - Adapter-delta extraction (mean-pooled, layers 12/18/24)
- Split eval = 142,072 examples (benign val + positive-user days), 70 positives.
- Adapter changed likelihood strongly: base_nll 2.70 -> adapted_nll 0.44.
- delta_nll (adapted - base): benign -2.262 vs malicious -2.208 -> malicious improved slightly LESS
  (delta ~0.054), the faint residual the SAE probes (consistent with benign-only training).
- 35 chunks/layer. Artifacts: extract_summary.json, example_scores.parquet/csv (on Anvil), chunk_manifest.csv.

## Phase 4 - Delta-SAE frontier (36 configs: layers x latent{2,4,8} x topk{4,8,16,32})
**Proxy selectivity only** (removed-reconstruction-energy inside the SAE), NOT model-level causal patching.

Headline (top5 - control3 selectivity advantage):
- Best: Layer 18, latent_mult 2 (d_latent 4096), k=8 -> 0.614 (recon_mse 0.045, L0=8).
- Runner-up: Layer 18, mult 4, k=4 -> 0.605.
- Layer 18 (~0.61) > Layer 24 (~0.53) > Layer 12 (~0.49).
- Sparser/smaller dictionaries (low k, low mult) -> more selective top units; denser (k=16/32) -> better
  recon, less selective. Control features ~0 selectivity (as designed).
- High dead-feature fraction for wide+sparse configs (d_latent 16384, k=4 -> ~89% dead), expected.

Full table: delta_sae_frontier_summary.csv / DELTA_SAE_FRONTIER_REPORT.md.

## Verdict for evaluation
The delta-SAE DOES find sparse units with proxy-selectivity for malicious (top-vs-control ~0.6 at layer 18)
- a promising signal, but NOT the decisive test. To judge vs the session-AE baseline on the real win
condition (sufficiency/patchability), run Phase 5 causal eval (eval_delta_sae_causal.py): top-vs-control
ablation, grounding, sparse patch/repair on the best configs (start: layer 18, k=8/mult=2 and k=4/mult=4).

## 2026-06-25 token-level Phase 5 update

The mean-pooled Phase 5 eval was followed by token-level layer-18 extraction, token SAE training, and
token-local causal patching.

Completed causal evals:

- `layer=18, latent_mult=2, k=8`: best top-minus-control repair advantage `+0.001405` (`team/top1`).
- `layer=18, latent_mult=4, k=4`: best top-minus-control repair advantage `+0.001335` (`team/top3`).

Token-level patching is weakly positive and better than the mean-pooled patch result, but effect sizes remain
small. Treat this as evidence to continue with session-AE baseline comparison and receiver-level uncertainty
estimation, not as a finished win condition.

Details: `TOKEN_PHASE5_FINDINGS.md` and `token_causal/`.
