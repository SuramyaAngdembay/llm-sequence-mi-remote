# CERT label pipeline repair, 2026-09-29

Implementation branch: `repair/cert-label-pipeline-20260929`, based on Claude's
review branch at `4c67d7a`. This implements the user's request for Codex to own the
label repair. Thesis work is excluded.

## What is corrected

The canonical target remains `cert-master-window-calendar-v1`: official incident
windows intersecting session calendar dates, origin 2009-12-28. This change does
not revise that target, the frozen user split, the sampling priority, or endpoints.

1. **Completed class-cache writer.** Corrected `y`, `y_original`, calendar date and
   version are attached before saving. `example_idx` comes from the complete eval
   JSONL; old local indices are retained as `source_example_idx`. Exact coverage,
   unique identities, historical-label agreement, finite counts/losses, integer
   counts and token-class partitions are mandatory. The actual bridge cache is
   compared with the actual historical cache, including scoring manifest equality
   and frozen B1 input hashes. A boolean bridge result alone is insufficient.
   Publication uses a temporary directory and a final completion/hash manifest.

2. **Typed label bundle (`cert-label-bundle-v2`).** `y` always stores the calendar
   target. The sampling artifact has a separate `preserve_for_sampling` column.
   Both artifact roles include full identity and preserved historical labels.
   Consumers verify bundle checksums, the canonical sparse label artifact, exact
   evaluation indices, identities, label semantics and intended artifact purpose.
   Legacy three-column overrides are refused. Filenames retain the `v1` suffix
   because the target is unchanged; the containing bundle is `labels_v2`.

3. **Dictionary sampling and ranking.** Sampling uses the separate priority flag
   and preserves actual labels in output metadata. `--dictionary-only` fits and
   saves the SAE without computing a label-based ranking. Subsequent reselection
   uses discovery positives and held-out non-insider negatives under the existing
   protocol. Confirmation-positive labels therefore do not become an accidental
   ranking through the dictionary trainer. The dictionary still sees confirmation
   inputs; this is explicitly transductive, exploratory work.

4. **Pilot input/output validation.** Corrected pilots require typed target labels,
   official insider donor exclusions, and population/input hash agreement. The
   manifest records every intended receiver, pair, target and condition. Analysis
   requires precisely that grid, with no missing users, duplicate rows, nonfinite
   values, changing target counts or inconsistent baselines. It checks loss-view
   reconstruction, zero-edit and reconstruction tolerances, and a real alpha-zero
   feature-edit path. A completion manifest binds the analyzed files. Undefined
   dose ratios are reported as undefined. The five newly included positive users
   are no longer called fresh/unexposed.

5. **Extraction merge and JSONL loading.** Merge requires both completed shards,
   matching extraction configurations, complete canonical evaluation identities,
   verified target labels, finite deltas and exact per-example token positions.
   Every delta chunk is hashed. Output is published atomically and retains original
   labels. A shared evaluation loader cannot replace those corrected targets with
   historical JSONL labels. Original chunks are only symlinked, never rewritten.

## Real artifact verification

Artifacts are under `/Users/suramya/Documents/mi-paper-label-fix-2026-09-29`.
`real_artifact_validation.json` records hashes and full checks of canonical
metadata, corrected evaluation JSONLs, and all 26 existing corrected score caches.

| Dataset | Metadata rows | Corrected positive days | Positive users | Changed labels |
|---|---:|---:|---:|---:|
| r4.2 | 330,295 | 1,362 | 70 | 407 |
| r6.2 | 1,393,297 | 73 | 5 | 23 |

The new r4.2 bundle covers 42,468 evaluation examples. Both target and sampling
files contain all 1,362 true positive days. Exactly 695 discovery-positive days
have sampling priority. The priority mask and 35/35 user lists match the frozen
earlier artifacts byte/value for byte/value; no population redesign occurred.

The explicitly restricted class-score artifact has 40,519 rows, 1,343 positive
days and 60 positive users. Its 1,949 missing rows include 19 positive days and
10 positive users. It is **not** a complete C32 result.

### Completed scoring and extraction, later in this repair session

Anvil job `20956714` subsequently completed successfully (exit 0, wall time
7m28s, one allocated GPU). The 256-row bridge matches exactly, including all 29
non-local-index columns when compared directly against the historical full cache.
The 1,949-row extension passed the strengthened identity, provenance and partition
gates. The new complete C32 artifact covers **42,468 rows, 1,362 positive days and
70 positive users**. It preserves both label versions and canonical indices.

An independent sklearn calculation, including the subtraction identity for the
behavior score, reproduced **840 scalar fold metrics**, maximum absolute error
2.23e-16. Results use 70 equal-weight held-out-positive-user folds, each with all
79 held-out negative users. These are the batch-1 class-decomposition score pass.

| Score | Fold-mean day AUROC | Fold-mean day AP | Fold-mean user AUROC |
|---|---:|---:|---:|
| Full | 0.539588 | 0.004859 | 0.588969 |
| Behavior only | 0.819505 | 0.025238 | 0.824955 |
| Profile only | 0.482322 | 0.001353 | 0.475769 |

The mean fold prevalence is 0.000714; pooled prevalence is a different population
quantity, 0.032071. Behavior-minus-full day AUROC is **0.279918**, paired user-fold
bootstrap 95% interval **[0.203587, 0.356702]**. On the ten newly included positive
users, mean day AUROC is 0.828720 for behavior versus 0.126702 for full. This
supports score mitigation on the complete cohort. It does not establish removal
of internal profile use, a causal feature mechanism, or an unexposed holdout.

The Aquaman merge also passed: **83 chunks, 11,860,673 token rows, 42,468 examples**.
Its corrected score index contains all 1,362 positives and preserves 407 changed
labels. All chunk hashes are recorded in `aquaman_merge_report.json`. A separate
post-publication check verified the completion hashes and exercised the actual
causal JSONL loader: it retains corrected labels when reading historical text.
Remote output: `aquaman_l26_v1/merged_calendar_v2`.

## Validation evidence

Regression tests exercise the actual completion-writer and pilot-analysis CLIs,
including the BBS0039 boundary-label failure and a lost control row. Negative
fixtures cover duplicate/missing identities, wrong historical labels, invalid
counts, NaN losses/gates, wrong baselines, wrong artifact purpose and changed bytes.
CPU PyTorch tests exercise sampling, an actual tiny dictionary fit, and shard
merge success/failure. Test fixtures are not research results.

Final counts and machine-readable results are in `local_pytest_results.xml` and
`remote_pytest_results.xml`. PyTorch tests run on Aquaman with CUDA disabled;
pytest dependencies are installed in an isolated handoff directory, not the
shared model environment.

## Execution boundary and remaining work

- The historical LM adapters remain reusable for this label failure. This repair
  neither trains nor modifies adapters. Dictionary refitting/reselection and
  corrected causal experiments are separate work.
- Both Aquaman activation shards and the checked merge are complete. Preserve
  the merged output and original chunks; do not rebuild solely because the driver
  is held. A future resumed driver must verify this completed output before use.
- Anvil job `20956714`, the bridge, the 1,949-row extension and corrected complete
  C32 evaluation are complete. Do not resubmit or overwrite them. The command
  `complete_c32_when_ready.sh` remains as the exact reproducible invocation and
  intentionally refuses to overwrite the now-completed result.
- Automatic Stage C is held (exit 78). The original remote driver is backed up in
  `codex_label_handoff_20260929/run_stage_c.original.sh`. The source driver also
  documents and enforces the hold. This is an implementation gate, not a request
  for user permission.
- Label/output checks do **not** establish cached/live feature-state agreement or
  realized edit norms after hidden-state rounding. Those checks and a bounded
  small pilot remain necessary before full causal execution. `edit_balance` still
  measures proposed float32 shifts. The historical extractor did not record
  start-time model/code hashes; a post-extraction hash must not be presented as
  that missing attestation.
- Old causal results are not retroactively validated. Keep the prior claim-by-claim
  reuse inventory and historical artifacts; corrected label-dependent populations,
  dictionaries and effects need their own validated runs.

## Rebuilding labels

`build_stage_b_labels.py` now requires `--labels incident_calendar_labels.parquet`.
`select_corrected_pilot_populations.py` also requires that canonical artifact.
The frozen existing population can be retained after identity/hash validation.

For merge, supply `--eval-jsonl`, `--labels labels_v2/ranking_labels_v1.parquet`,
both `--shards`, and a new `--out-dir`. Never rerun it into an existing directory.
For completed class evaluation, `eval_r42_classes_complete.py --help` lists the
required actual bridge cache, three scoring manifests, frozen B1 input manifest
and JSONL files. It refuses old invocations that lack those provenance checks.
