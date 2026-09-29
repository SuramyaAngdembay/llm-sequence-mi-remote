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
- Both Aquaman activation shards completed successfully. Validate and merge them
  with the new script; do not infer completion from directory existence.
- Anvil job `20956714` is the already-submitted batch-1 bridge and 1,949-row class
  extension. At the last check it was pending unavailable/reserved nodes. Preserve
  the job and original inputs; do not submit a duplicate. Run the corrected writer
  only after its outputs and bridge checks exist.
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
