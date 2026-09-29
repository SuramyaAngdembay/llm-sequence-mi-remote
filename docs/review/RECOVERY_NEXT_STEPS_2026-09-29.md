# Return to the claim-by-claim label recovery

## Intended approach

Repair the calendar target and trace its consequences through every paper claim.
Reuse verified artifacts when the intervention/training/scoring inputs are
unchanged. Recompute statistics when only labels/groups changed. Rebuild selection
or fitting when those depended on old labels. Run only necessary missing or
changed experiments. Keep original artifacts, failed gates and inconclusive
results visible. Synchronize all three papers from one accepted result registry.

This plan is a reconciliation of existing evidence, not a new independent
verification of all 46 claim families. Historical inventories are preserved.
Their initial `status` and later `cached_recovery_status` fields describe different
stages; reading the initial field alone wrongly marks completed work as pending.

## Completed scopes to preserve

- Labels and counts: both CERT datasets, with independent clock/event checks.
- Adapter training audits: existing main adapters remain usable for this bug;
  other families retain their recorded provenance qualifications.
- Cached LM results: C05–C09, C28 and C31 already have corrected reanalysis and
  an independent review. Comparator portions of C05/C06 remain separate.
- C33 primary time-series comparison and C34 fixed-sample history analysis were
  also recomputed; pending history-shuffle or broader causal claims are not cleared.
- C32 r4.2 behavior-only evaluation is now complete for all 42,468 examples and
  70 positive users. The older 60-user limitation is superseded for this result.
- C35 corrected causal pilot is complete as recorded diagnostics. Its original
  global matching gate remains failed; separate diagnostic analysis and a new
  control-policy feasibility probe are verified. Attack specificity is not
  established. This pilot does not repair every historical causal claim.

The numerical-validation branch served the corrected pilot. Its diagnosis is now
finished; further numerical or mathematical exploration is not the next label
recovery step.

## Next execution order

1. **Consolidate the result registry and manuscript map.** Reconcile all 46 claims
   across recovery stages; attach current values, source files, populations,
   scoring configuration, validation evidence and precise wording limits. Map
   each occurrence in the three paper versions. Do not recompute completed work
   because an old status field says `RECOMPUTE`.
2. **Audit remaining comparison-model and causal dependencies.** For C27/C45/C46,
   check actual training/scaler/fold membership against official insiders and
   identify saved checkpoints/predictions. For retained causal claims, compare
   exact adapter, tokenizer/text, dictionary/normalization, features, receiver,
   donors, edit, target and numerical configuration. A changed dictionary can
   eliminate exact reuse even when receiver labels are unchanged.
3. **Perform bounded CPU recovery first.** Prioritize reusable fixed-sample
   contrasts (C10/C36/C39) and optional cached associations (C30) when needed for
   the paper. Validate corrected groups and donor/partner eligibility. A
   fixed-sample correction is not complete-population coverage.
4. **Specify only necessary reruns.** Each job needs claim IDs, the unsupported
   inference it resolves, exact changed/missing inputs, frozen endpoints and
   validity checks, expected output, and a measured compute estimate. For C22,
   missing positives are only part of the problem: old label-selected negative
   pools also require correction. Do not blanket-resubmit historical jobs or
   interpret the historical ~200 GPU-SU spending as the repair bill.
5. **Synchronize and verify the manuscripts.** Update main, readable-full and
   plain-summary from the same registry. Preserve readable-full's linear
   exposition. Replace stale figures/numbers and qualify unsupported claims;
   successful execution alone does not reinstate a claim. Check repeated values,
   table/figure references and all three builds.

Claims outside the CERT date bug (TWOS, LANL and mathematical directions) retain
their own limitations and work ledger. They do not become cleared by this repair.
New novelty searches are parked while restoring the paper's evidential basis.

## Artifacts

The reconciled 46-row work plan and source-hash manifest are:

- `/Users/suramya/Documents/mi-paper-causal-validation-2026-09-29/current_recovery_plan.json`
- `/Users/suramya/Documents/mi-paper-causal-validation-2026-09-29/current_recovery_plan.csv`
- `/Users/suramya/Documents/mi-paper-causal-validation-2026-09-29/current_recovery_plan_manifest.json`

No experiments were launched while preparing this plan. It changes task
priorities and bookkeeping, not scientific targets or the status of failed runs.
