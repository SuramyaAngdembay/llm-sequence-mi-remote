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
- `team,role,dept,dept_role` context modes
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

- `CONTEXT_MODES=team,role,dept,dept_role`
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

## Anvil Submission Status

Submitted from Anvil at `2026-07-02 18:36 EDT` while the token-SAE frontier
job was still running.

Frontier dependency:

- `18755591 token_delta_sae`

Queued causal jobs:

- `18810245 token_delta_causal`: `l18_m04_k04`
- `18810247 token_delta_causal`: `l18_m02_k04`
- `18810249 token_delta_causal`: `l18_m04_k08`

All three causal jobs were verified with:

- `Dependency=afterok:18755591(unfulfilled)`

Queued bootstrap jobs:

- `18810246 tok_boot_cpu`: after `18810245`
- `18810248 tok_boot_cpu`: after `18810247`
- `18810250 tok_boot_cpu`: after `18810249`

The causal suite should not begin until the R4.2 token-SAE frontier exits
successfully.

## Anvil Resubmission Status

Checked on Anvil at `2026-07-03 19:07 EDT`.

The first queued causal suite started after the frontier completed, but all
three causal jobs failed quickly because the R4.2 structured JSONL context does
not include a `project` column. The failing mode was `project_role`.

Failed causal jobs:

- `18810245 token_delta_causal`: `l18_m04_k04`, failed with missing `project`
- `18810247 token_delta_causal`: `l18_m02_k04`, failed with missing `project`
- `18810249 token_delta_causal`: `l18_m04_k08`, failed with missing `project`

Their corresponding bootstrap jobs are now dependency-never-satisfied:

- `18810246`
- `18810248`
- `18810250`

The corrected suite was resubmitted with:

- `COMMON_CONTEXT_MODES=team,role,dept,dept_role`

Corrected causal jobs:

- `18832356 token_delta_causal`: `l18_m04_k04`
- `18832358 token_delta_causal`: `l18_m02_k04`
- `18832360 token_delta_causal`: `l18_m04_k08`

Corrected bootstrap jobs:

- `18832357 tok_boot_cpu`: after `18832356`
- `18832359 tok_boot_cpu`: after `18832358`
- `18832361 tok_boot_cpu`: after `18832360`

As of the resubmission check, the corrected causal jobs were pending on
priority, with no remaining frontier dependency.

## Anvil Memory Resubmission Status

Checked on Anvil at `2026-07-03 22:06 EDT`.

The corrected context-mode suite started, but all three causal jobs were killed
by Slurm CPU-memory OOM. This was not a GPU VRAM failure and not another
missing-column/config error.

Failed causal jobs:

- `18832356 token_delta_causal`: `l18_m04_k04`, `OUT_OF_MEMORY`
- `18832358 token_delta_causal`: `l18_m02_k04`, `OUT_OF_MEMORY`
- `18832360 token_delta_causal`: `l18_m04_k08`, `OUT_OF_MEMORY`

Slurm accounting:

- requested memory: `240G`
- observed `MaxRSS`: about `251,657,000K`
- failure point: about five minutes into each causal job

The R4.2 causal wrapper now overrides the generic Slurm template memory with:

- `CAUSAL_MEM=360G`

This keeps the higher memory request scoped to R4.2 causal jobs rather than
changing the shared token-causal template for every branch.

Resubmitted at `2026-07-03 22:09 EDT` with `CAUSAL_MEM=360G`.

Memory-resubmitted causal jobs:

- `18835178 token_delta_causal`: `l18_m04_k04`
- `18835180 token_delta_causal`: `l18_m02_k04`
- `18835182 token_delta_causal`: `l18_m04_k08`

Memory-resubmitted bootstrap jobs:

- `18835179 tok_boot_cpu`: after `18835178`
- `18835181 tok_boot_cpu`: after `18835180`
- `18835183 tok_boot_cpu`: after `18835182`

Verified Slurm submission detail on all three causal jobs:

- `SubmitLine=sbatch --parsable --mem=360G --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch`
- `ReqTRES=cpu=24,mem=360G,node=1,billing=1,gres/gpu=1`

As of submission, the three causal jobs were pending on priority.
