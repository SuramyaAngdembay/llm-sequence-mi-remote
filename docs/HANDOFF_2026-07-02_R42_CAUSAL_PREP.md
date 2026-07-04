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

## Full-Run OOM Diagnosis

Checked on Anvil at `2026-07-04 01:05 EDT`.

The `CAUSAL_MEM=360G` retry also failed with CPU-memory OOM:

- `18835178 token_delta_causal`: `l18_m04_k04`, `OUT_OF_MEMORY`
- `18835180 token_delta_causal`: `l18_m02_k04`, `OUT_OF_MEMORY`
- `18835182 token_delta_causal`: `l18_m04_k08`, `OUT_OF_MEMORY`

Slurm accounting for the 360G retry:

- requested memory: `360G`
- observed `MaxRSS`: about `377,486,000K`
- failure point: about five to six minutes into each causal job

This confirms the issue is not a small under-request. The current causal
evaluator materializes large dense arrays:

- full layer-18 token rows: `11,860,673`
- needed token rows for the full default R4.2 suite: `8,101,247`
- token delta matrix `x`: about `123.62 GiB`
- normalized copy `x_norm`: about `123.62 GiB`
- dense m02 SAE activation matrix: about `247.23 GiB`
- dense m04 SAE activation matrix: about `494.46 GiB`
- largest per-inner-loop patch list: about `82.11 GiB`

So the full default m04 run wants substantially more than `360G`, and likely
more than is worth requesting blindly.

Why this failed on `r4.2` even though the `r6.2` raw token cache was larger:

- the successful `r6.2` token-causal jobs had only `70` positive receivers
- those jobs loaded `1,112,836` receiver/donor-relevant token rows
- they produced `142,160` candidate rows
- the full default `r4.2` suite needs `8,101,247` receiver/donor-relevant token
  rows and `2,666,032` candidate rows

So the raw extraction size is not the bottleneck. The bottleneck is the number
of causal receiver/donor examples and their token rows after the matched-pair
selection step.

Sizing estimates by receiver cap:

| max receivers | total pairs | needed rows | x GiB | m02 dense GiB | m04 dense GiB | max patch list GiB | candidate rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4,081 | 599,444 | 9.15 | 18.29 | 36.59 | 2.06 | 65,296 |
| 64 | 8,171 | 1,109,244 | 16.93 | 33.85 | 67.70 | 4.17 | 130,736 |
| 128 | 16,344 | 1,829,910 | 27.92 | 55.84 | 111.69 | 7.92 | 261,504 |
| 256 | 32,592 | 3,110,190 | 47.46 | 94.92 | 189.83 | 16.92 | 521,472 |
| 512 | 65,163 | 4,772,681 | 72.83 | 145.65 | 291.30 | 32.11 | 1,042,608 |
| full | 166,627 | 8,101,247 | 123.62 | 247.23 | 494.46 | 82.11 | 2,666,032 |

Recommended path:

1. Run a bounded R4.2 causal probe with `COMMON_MAX_RECEIVERS=128` into a
   separate output root. This should fit under the current `360G` request and
   gives a fast signal without pretending it is the full headline result.
2. Patch the evaluator for the full run so it does not materialize dense
   `n_token_rows x d_latent` SAE activations or full `patch_list` arrays. The
   better full-run design is to cache only token-level top-k SAE activations and
   stream pair batches.
3. After the memory-efficient evaluator exists, rerun the full uncapped suite.

The R4.2 recommended launcher now exposes:

- `COMMON_MAX_RECEIVERS`
- `COMMON_MAX_CANDIDATE_DONORS`
- `COMMON_PATCH_CHUNK_SIZE`

These default to the previous full-run behavior, but allow capped probes
without editing the scripts again. `COMMON_PATCH_CHUNK_SIZE=0` means use the
model scoring batch size as the patch-construction batch size.

## Receiver-Capped Probe Submission

Submitted at `2026-07-04 01:16 EDT`.

Purpose: get an end-to-end R4.2 causal signal while the full evaluator is being
made memory efficient. This is a probe, not the final uncapped headline run.

Soundness note:

- The receiver cap changes the estimand from the full R4.2 causal estimate to a
  fixed receiver-sampled probe estimate. Treat it as a triage/sanity result, not
  the final headline number.
- The evaluator now samples capped positive receivers once per run seed and reuses
  that same receiver set across context modes and benign/anomalous donor pools.
  This keeps top/control and donor-label comparisons aligned inside the capped
  probe instead of comparing different receiver subsets.
- The full-strength result should still come from an uncapped rerun after the
  memory-efficient evaluator rewrite.

Submission settings:

- `COMMON_MAX_RECEIVERS=128`
- `COMMON_MAX_CANDIDATE_DONORS=16`
- `COMMON_CAUSAL_MEM=360G`
- output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on_recv128`

Probe causal jobs:

- `18836313 token_delta_causal`: `l18_m04_k04`
- `18836315 token_delta_causal`: `l18_m02_k04`
- `18836317 token_delta_causal`: `l18_m04_k08`

Probe bootstrap jobs:

- `18836314 tok_boot_cpu`: after `18836313`
- `18836316 tok_boot_cpu`: after `18836315`
- `18836318 tok_boot_cpu`: after `18836317`

Verified Slurm submission detail on all three probe causal jobs:

- `SubmitLine=sbatch --parsable --mem=360G --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch`
- `ReqTRES=cpu=24,mem=360G,node=1,billing=1,gres/gpu=1`

These receiver-capped probe jobs were cancelled on Anvil after the streamed
uncapped evaluator became available, to avoid spending H100 SUs on non-final
probe estimates.

## Follow-Up Evaluator Rewrite

Prepared locally on Magnolia after the OOM diagnosis.

Purpose: make the **full uncapped R4.2 causal estimate** feasible again without
changing the causal protocol.

What changed in `scripts/eval_token_delta_sae_causal.py`:

- stop materializing a full `x_norm` copy of the needed token rows
- stop materializing a full dense `sparse_all` latent matrix for all needed rows
- keep extracted token deltas in `float16` in host RAM, then cast per batch
- build sparse SAE activations only for the current pair batch's unique examples
- stop accumulating a full in-memory `candidate_rows` list; stream it directly to CSV
- keep only the compact per-receiver best-row table in memory
- Anvil follow-up: expose `PATCH_CHUNK_SIZE`/`COMMON_PATCH_CHUNK_SIZE` so patch
  construction can be reduced independently of model scoring batch size if host
  memory is still tight

What did **not** change:

- same matched donor / receiver logic
- same context modes
- same feature-set selection logic
- same token-level patching and scoring procedure
- same bootstrap downstream on `token_delta_sae_causal_best_rows.csv`

Interpretation:

- this is a **memory-efficiency rewrite**, not a methodological change
- if the uncapped rerun now fits, it should be valid to compare directly against
  the prior capped probe and against the R6.2 full causal results

## Final Uncapped Streamed Submission

Submitted on Anvil at `2026-07-04 03:40 EDT`.

Purpose: produce the final-table-valid full `r4.2` causal estimate using the
same receiver/donor estimand as the prior `r6.2` token-causal results.

Submission settings:

- `COMMON_MAX_RECEIVERS=0`
- `COMMON_MAX_CANDIDATE_DONORS=16`
- `COMMON_CAUSAL_MEM=360G`
- `COMMON_PATCH_CHUNK_SIZE=0` (`0` means use `BATCH_SIZE`, currently `8`)
- output root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_mb22_gc_on_stream_uncapped_v1`

Final causal jobs:

- `18836629 token_delta_causal`: `l18_m04_k04`
- `18836631 token_delta_causal`: `l18_m02_k04`
- `18836638 token_delta_causal`: `l18_m04_k08`

Final bootstrap jobs:

- `18836630 tok_boot_cpu`: after `18836629`
- `18836632 tok_boot_cpu`: after `18836631`
- `18836639 tok_boot_cpu`: after `18836638`

Verified Slurm submission detail on all three final causal jobs:

- `SubmitLine=sbatch --parsable --mem=360G --export=ALL slurm/eval_token_delta_sae_causal.template.sbatch`
- `ReqTRES=cpu=24,mem=360G,node=1,billing=1,gres/gpu=1`
- state at submission check: pending on priority
