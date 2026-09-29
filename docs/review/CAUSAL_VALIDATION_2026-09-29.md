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

Aquaman only; one GPU at a time. CPU preparation has a 45-minute wall cap. The
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
