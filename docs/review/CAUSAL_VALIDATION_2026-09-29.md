# Corrected causal execution: implementation validation before effects

Authorized by the user's "okay go ahead" after the completed label recovery.
This amendment is written before any new intervention outcome. Existing adapter,
corrected user assignments, donors, 2M-row dictionary sampling, SAE architecture,
training seed/epochs, ranking policy and estimands remain fixed.

## Gates and sequence

1. Prepare dictionary rows using the frozen discovery-only priority mask. Split
   serialization into bounded chunks; this changes no sampled row or order.
2. Check the layer indexing and fresh batch-1 base/adapted hidden states against
   the stored extraction. Cached/live agreement is tested exactly, with errors
   reported descriptively rather than accepting an outcome-dependent tolerance.
   A failed exact match blocks using cached codes as the live receiver state.
3. Refresh states for the fixed pilot receivers and donors in the pilot's actual
   batch-1 configuration if needed. The dictionary/ranking can use the already
   declared extraction distribution; causal code/prototypes must use verified live
   states. Preserve both versions and disclose this numerical distribution shift.
4. Fit the corrected dictionary and rank on discovery positives and the declared
   non-insider negatives only. No confirmation-label ranking is produced during
   dictionary fitting.
5. Instrument every scoring hook: proposed float32 shift, cast shift and realized
   `after.float() - before.float()`. Record norms and token support separately.
   Zero and alpha-zero paths must reproduce unhooked scores within 1e-5. Exact
   shapes, finite values and matching pre-edit live states are mandatory.
6. Run a deterministic small engineering pilot before a full run. Its effects are
   diagnostics, not evidence of scientific success. Keep the selected feature
   sets/populations fixed regardless of signs. Match selected/random controls on
   realized token support, and report realized norm ratios. For expanding this
   pilot, require matching support and at most 5% relative per-token norm mismatch
   where the reference edit is nonzero. This is an engineering tolerance, not
   mathematical equivalence. Failed matching blocks a matched-dose causal claim;
   do not retune using outcomes. Selected-versus-control-feature comparisons are
   distinct policy comparisons and are not assumed to have equal doses.
   Each arm also needs at least one nonzero realized reference edit; an entirely
   inactive arm cannot establish matching. The diagnostic cohort is eight fixed
   pairs: indices 18, 25, 50, 81, 82, 85, 108, 118 in the original population
   manifest. It includes the five newly included users, the longest-token pair,
   and two further distinct users selected by a fixed hash. This selection was
   recorded before dictionary fitting, ranking, or edited scoring.

Original Stage C stays held until these gates pass. A run-completion marker proves
file completeness, not scientific validity. Unlabelled confirmation inputs remain
in dictionary training: results are exploratory and transductive.

## Compute envelope for this phase

Aquaman only; one GPU at a time. CPU preparation initially had a 45-minute wall cap;
it was extended to 60 minutes during serialization after all rows had been selected,
before any dictionary or intervention outcome. This avoids repeating the source
read and makes no sampling change. The
new GPU work has a total 135-minute cap (2.25 RTX GPU-hours), including failures:
state checks/refresh up to 10 minutes, dictionary fit 75, ranking 20, engineering
pilot 10, and a gated full pilot 20. Stop dependent work on any failed gate.
Record actual timings. These are RTX GPU-hours, not Anvil SUs. The previous
extraction and completed Anvil extension are separate already-spent work.

Resource-only amendment, before dictionary fitting or edited scoring: the
unhooked/zero replay took 57 seconds for 16 mostly long examples, so a full run
may need more than its original 20-minute suballocation. Unused time from earlier
steps may fund the full pilot, capped at 60 minutes and the remaining 135-minute
total. Estimate full runtime from engineering throughput and frozen token counts;
do not launch it if the remaining envelope is insufficient. This changes no
scientific sample, feature selection, endpoint, or validity tolerance.

### Ranking preparation amendment (after dictionary completion, before rankings)

The dictionary completed all 20 epochs (MSE 0.0895962054; checkpoint SHA256
`aa0fd14f3d7bd6e25db6a3c619fa6c223616e673ee06a351d78caa97129bcae1`).
The first ranking attempt hit its 1,200-second cap after loading 1,956,099 rows
(171,917 positives) and before writing a ranking. It produced no selected features
or intervention outcomes. Its full 1,206.1-second wall time remains charged.

Prepare and normalize the exact same ranking rows in a separate CPU-only job,
capped at 45 minutes. Preserve row identities, label roles, seed, order and
normalization operations in a checksummed artifact. Read the normalized matrix
with a memory map for the existing GPU evaluator, keeping its 8,192-row batches
and accumulation code. Test direct versus prepared evaluation for exact agreement
in feature scores, statistics and selected sets before launching the real job.
Do not refit the completed dictionary. The retry, engineering pilot and any full
pilot must fit the remaining 3,753.8 seconds of the original GPU-stage envelope;
the already failed attempt is not refunded. This is a resource correction, not a
change to the scientific ranking policy or an outcome-based selection.

On an 8GB card, score token losses in bounded chunks and use the same scoring
path for unhooked, zero, and edited runs. Do not infer a noise floor from a
comparison that changed batches, hardware, or loss arithmetic.

## Implementation evidence obtained before dictionary fitting

- Captured all 356 fixed receiver/donor examples at batch size 1; the layer-26
  hook output exactly equals `hidden_states[26]` in the capture check.
- Cached versus fresh deltas: 54/356 examples are exactly equal. Across examples,
  the median relative Frobenius-norm error is 0.05431448 and the maximum is
  0.57252239. This comparison establishes a numerical mismatch; it does not by
  itself isolate which extraction setting caused it. Fresh states are mandatory
  for the pilot. Dictionary training/ranking still use the declared cached
  extraction distribution; do not describe it as identical to live batch-1 data.
- A separate model load and the actual chunked scorer reproduced all 16 chosen
  receivers' pre-edit states exactly. Unhooked versus zero-edit mean NLL, class
  loss sums, and class counts all have maximum absolute difference 0.
- The first capture attempt stopped before recording states because the installed
  Transformers version lacks `hf_device_map`. The corrected check inspects actual
  parameter devices. Preserve that failed attempt and include its time in the
  compute accounting.

## Completed engineering validation

Execution plan v4 completed the CPU-prepared ranking retry in 741.94 seconds and
the fixed eight-pair engineering pilot in 416.36 seconds. Ranking used 1,956,099
rows, including 171,917 positive rows from all 35 discovery users, and negatives
from the 79 declared users. Every ordered row identity independently reproduced.
Selected features are `[3843, 3230, 3199, 3975, 1394]`; active low-gap controls are
`[1087, 3632, 1319, 3127, 72]`. The completed dictionary was reused.

All 240 receiver-condition outputs and 173,115 token-condition measurements are
complete. Live pre-edit states matched exactly throughout. Zero and alpha-zero
NLL differences are exactly zero. Reconstruction-identity residual is
3.814697265625e-6. All four random-control comparisons have identical token support
and no per-token norm error above 5%; the largest error is 0.581711%. A separate
CPU audit reproduced feature selection, corrected receiver labels and identities,
donor eligibility, complete grids, score-view arithmetic, and physical matching.
The scientific analyzer correctly rejects this engineering-only cohort.

Selected and control-feature arms edit 32 and 124 token positions respectively
out of 11,541. Matching applies to random controls within each arm; it does not
make selected-versus-control-feature effects equal-dose comparisons. These checks
establish implementation behavior on the engineering cohort, not anomaly
specificity, mediation, or validity on every full-cohort token.

The phase stopped at its compute gate without launching the full pilot. Charged
stage wall time is 5,504.51 seconds, including the failed ranking attempt.
Full-run prediction is 4,100.88 seconds, versus 2,580 seconds available and a
separate 3,600-second subcap. Do not describe the full pilot as running or complete.
A separate continuation needs a suitable new budget and must preserve all frozen
inputs, conditions and gates. The original Stage C driver stays held.

Evidence is archived at
`/Users/suramya/Documents/mi-paper-causal-validation-2026-09-29/REPORT.md` and
`/data/suramya/cert_review_20260928/causal_validation_20260929/`.

## User-authorized full-run budget extension

After reviewing the compute stop, the user explicitly instructed: "run it anyway
even if the budget is overboard unless it is not possible to run it" and then
"continue". The full pilot now has a separate 5,400-second (90-minute) allowance.
The previous 8,100-second phase cap and 3,600-second full-run subcap are superseded
for this authorized continuation; all previously charged time remains recorded.

`run_authorized_full_pilot.py` in the phase artifact directory launches the exact
previously proposed command: 316 receivers, 158 pairs, 34 users, 15 conditions.
It verifies prior evidence/code/input hashes, the engineering gates, the SAE and
ranking, and the cached base-model files before launch. Runtime intervention code
remains commit `b5242cd`; no scientific setting or validity tolerance changes.
After completion it checks full-cohort validity and runs the prespecified analysis
only if the realized-control gate passes. Output completeness alone does not
clear a causal claim. Execution status is recorded in `execution_full_v1.json`.
