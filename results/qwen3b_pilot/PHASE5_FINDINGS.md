# Phase 5 — Causal Patch Eval: Findings (2026-06-22)

Model-level causal patching of **mean-pooled** adapter deltas at hidden-state layer 18, on the two best
frontier SAE configs. Both runs COMPLETED (~33 min each, 70 positive receivers, full protocol:
4 context modes x 4 alphas x {top1/top3/top5 vs control3} x benign/anomalous donors).

## Headline: causal patchability NOT demonstrated (null / slightly negative)
Clean metric = `top_minus_control_advantage` (higher = top SAE features cause more benign-specific repair
than control features; summary is sorted descending).

- l18/m02/k08: top_minus_control_advantage = **-0.0006 to -0.0038** across context modes — ALL negative.
- l18/m04/k04: **+0.00025 to -0.0028** — only team/top5 is marginally positive (+0.00025); rest <= 0.

The proxy-selective top features show **no consistent causal advantage over control features**.

## Magnitudes are near the noise floor
All repair deltas are ~0.001-0.005 NLL = **~0.2-1% of base adapted_nll (~0.44)**. The mean-pooled,
uniformly-broadcast patch barely perturbs the model, so the test has little power either way.

## Interpretation
Phase-4 proxy selectivity (~0.61 at layer 18) did **NOT** translate into causal patchability under this
coarse intervention. Per SUCCESS_CRITERIA, "sparse features without patchability" is not a win. The central
question (does delta-SAE beat the session-AE on sufficiency/patchability) remains **open**, gated by the
bluntness of the mean-pooled patch.

## Caveats (why "not shown", not "refuted")
1. Coarse patch: mean-pooled delta broadcast uniformly over tokens; tiny effect sizes point at this.
2. "Best" aggregation (min delta per receiver across alphas/donors) is optimistic and noisy at ~0 effects.
3. Small N = 70 positives -> low power.

## Recommended next steps
1. **Token-level extraction + patching** (the real circuit-patch test): per-token deltas, patch at token
   positions. Decisive escalation. NOTE: token-level deltas are far larger -> use $SCRATCH + sharding.
2. **Head-to-head vs the session-AE baseline** under the same causal protocol (the win condition is relative).
3. Lower priority: 7B scale-up.

Per-config: summary.csv + report.md + selected_sets.csv in `causal/<config>/`. Full candidate/best rows on
Anvil at `outputs/delta_sae_causal_qwen3b/<config>/`.

## 2026-06-25 follow-up

The token-level escalation in step 1 is now complete for layer 18 on the two requested configs. Results are in
`TOKEN_PHASE5_FINDINGS.md` and `token_causal/`.

Short version: token-level patching gives small positive top-vs-control repair advantages
(`+0.001405` for `m02/k08`, `+0.001335` for `m04/k04`), strongest in `team` context. This is better than the
mean-pooled result, but still small enough that the next decisive step is session-AE baseline comparison plus
bootstrap confidence intervals over the 70 positive receivers.
