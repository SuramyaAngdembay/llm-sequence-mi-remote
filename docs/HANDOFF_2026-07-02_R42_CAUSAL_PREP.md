# Handoff: `r4.2` Causal-Patching Prep

This note records the `r4.2` token-level causal-eval setup so the Anvil side
can move directly from completed token-SAE frontier outputs into patching.

## Why This Exists

The `r4.2` remote branch is intended to test transfer of the current winning
mechanistic protocol, not to reopen broad architecture exploration.

So once the `r4.2` token-SAE frontier finishes, the next step should be:

1. targeted token causal patching
2. bootstrap uncertainty estimates
3. matched comparison against the local `r4.2` baseline table

## Major Parameters To Keep Fixed

Keep the same causal protocol family that worked on `r6.2`:

- token-level patching, not mean-pooled patching
- matched donor/receiver selection
- `team,role,project_role,dept_role` context modes
- top-vs-control comparison
- active-control threshold available from the start
- bootstrap on the best-row CSV output

## Default `r4.2` Runtime Roots

- session JSONL:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/session_jsonl_r42`
- adapter:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter`
- token cache:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on`
- frontier:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_frontier_qwen3_8b_r42_mb22_gc_on`
- causal outputs:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on`

## Prepared Launchers

Single-config causal bundle:

```bash
bash scripts/submit_qwen3_8b_r42_token_eval_bundle_anvil.sh
```

Recommended first causal suite:

```bash
bash scripts/submit_qwen3_8b_r42_recommended_causal_suite_anvil.sh
```

## Recommended First Configs

Before seeing the full `r4.2` frontier, use the same strong `r6.2` priors:

1. `layer=18, latent_mult=4, k=4`
2. `layer=18, latent_mult=2, k=4`
3. `layer=18, latent_mult=4, k=8`

These are defaults, not a claim that `r4.2` must peak at the same exact point.
If the finished frontier clearly points elsewhere, override the bundle inputs.

## Default Causal Settings

- `TOP_SETS=top1,top3,top5`
- `CONTROL_SET=control3`
- `ACTIVE_CONTROL_MIN_FRAC=0.002`
- `ALPHAS=0.25,0.5,0.75,1.0`
- `MAX_CANDIDATE_DONORS=16`
- `N_BOOTSTRAP=4000`

## Expected Outputs Per Config

- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_best_rows.csv`
- `token_delta_sae_causal_selected_sets.csv`
- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `bootstrap/token_delta_sae_bootstrap_summary.csv`
- `bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`

## Follow-Up Rule

If one `r4.2` config shows a suspiciously large gain with weak/inert controls,
repeat the `r6.2` audit pattern:

- inspect active receiver-token support
- rerun with an active control set if needed

Do not assume the first strongest raw row is automatically the clean headline.
