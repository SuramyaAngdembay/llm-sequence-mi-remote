# Stage B/C protocol: corrected scoring reference, coverage repair, dictionary, selection and pilot

**Frozen 2026-09-28, before building any corrected population or scoring any edit.**
Code: branch `review/cert-independent-20260928`. The commit hash of this protocol's
implementation is recorded in each job's manifest. No rule below may change after a result
exists; any later change is a new, separately named analysis.

## 0. Claim chosen for the pilot, and why (decided before any pilot effect)

**C35/C25, stated narrowly:** at full strength, a residual-preserving edit of the selected
layer-26 SAE features toward a benign-colleague prototype raises **behavior-token** loss on
confirmation-user attack days more than:
- a norm- and position-matched random direction;
- the same edit of matched control features.

Reasons for choosing it:
- **Link to the scoring result.** It uses the behavior-only endpoint, so it connects directly to
  the behavior-only scoring result (C31/C32).
- **Coverage.** The r4.2 token-delta cache covers all 42,468 unseen days, including the ten
  newly positive insiders, so no extraction is needed.
- **Existing engine.** Pilot 1 already has same-batch zero baselines, a zero-edit exactness gate,
  measured edit sizes and matched random directions.
- **Label dependence is contained.** Its historical dependence on labels (dictionary sampling,
  ranking, receivers, matched days, donors) is exactly what this protocol replaces.

It is **not** a test of attack specificity, a complete circuit, or a profile mechanism.

## 1. Scoring reference (Stage B.A)

- **Scorer.** `scripts/score_token_class_decomposition.py`. Script sha256 `9a6ff58f…`,
  `token_class_decomposition.py` `00020e47…`, config sha256 `a0d7264a…`.
- **Model inputs.** Adapter `e0bf5112…`, tokenizer digest `794d46b3090bde93`. Batch size 1
  (no padding), fp32 cross-entropy, max length 2048.
- **Environment.** Collaborator environment `cert-qlora`: torch 2.5.1+cu121, transformers
  5.16.1, peft 0.20.0, bitsandbytes 0.50.2, accelerate 1.14.0, on an NVIDIA A100-SXM4-40GB.
- **Why this configuration.** It is the only one shown to be reproducible bit-for-bit across
  runs (review F2).
- **Bridge gate (hard).** Before any new row is accepted, rescore the 256 examples of the cached
  batch-1 pilot. Every one of `adapted_nll`, `loss_sum_*`, `n_*` and `n_tokens` must equal the
  cache exactly (0 differences). If not, the extension is refused and a separately versioned
  complete rerun is needed (about 1.4 A100-h); cached and new rows are never combined.

## 2. Coverage repair (Stage B.B)

- **r4.2 class cache.** Score the 1,949 missing unseen days of the 10 newly positive insiders
  with the reference configuration and the original `eval.jsonl` text (historical `y`
  retained). Text is checked byte-identical, and example IDs must be exactly the 1,949 missing
  keys. Accepted only after the bridge gate passes. The complete 42,468-row cache is a new file,
  `class_scores_complete_calendar_v1.parquet`; the 40,519-row cache is kept.
- **Evaluation.** Recompute C32 with the reviewer's evaluator on all 70 corrected positive users:
  - fold-mean day ROC and AP with prevalence;
  - user ROC;
  - behavior-only vs full paired bootstrap (10,000 draws, seed 42);
  - a breakdown by scenario.
- **Repaired 3B coverage (C22).** Not in this pilot. Only its cost is estimated, from the
  audit's missing-positive counts (211 r4.2 / 11 r6.2 per seed), which this review re-verifies
  before any extension is scheduled.

## 3. Corrected dictionary (Stage B.C)

| item | rule |
|---|---|
| deltas | r4.2 layer-26 token deltas, `pkg4_share/token_deltas` (the copy Package 4 and Pilot 1 used; chunk manifest and score index checked against the original cache) |
| population | all 42,468 unseen days (79 held-out benign users and 70 insiders); no training-user deltas exist |
| labels | `cert-master-window-calendar-v1` (review F1) |
| sampling | historical 2,000,000-row cap and formula. **Positive preservation is limited to corrected-positive days of discovery users.** Every other row (benign users, confirmation users, insiders' window-negative days) is sampled uniformly at the same rate |
| rationale | historical preservation used every user's (wrong) labels, including confirmation users'. Limiting it to discovery users keeps the method label-sampled while removing confirmation labels from the fit |
| normalization | mean and std of the selected sample (historical) |
| architecture | TopK SAE, d_in 4096, 8,192 features (m02), k = 4, 20 epochs, batch 4096, seed 42 (+ layer = 68, as historical) |
| outputs | new directory; the historical dictionary is kept as a fixed-dictionary reference only |

## 4. Discovery and confirmation users (selection)

- **The original 60 insiders.** Keep their historical 30/30 assignment
  (`user_splits_r42/*.txt`, seed 42).
- **The 10 newly positive insiders.** Order them by sha256(`cert-calendar-v1-split-2026-09-28|user`);
  the first 5 go to discovery and the rest to confirmation.
- **Exposure disclosure.** The 30 historical confirmation users were inspected in Package 4,
  Pilots 1–4 and H6, so they are **not a fresh scientific holdout**. The 5 new confirmation users
  are unexposed, and their results are also reported separately (descriptive, 5 users).

## 5. Feature ranking and controls

- **Positive rows.** Corrected-positive days of discovery users.
- **Negative rows.** Days of the 79 held-out benign users only, sampled at 0.25 with seed 42.
  Insiders' window-negative days are never negatives. Confirmation users are excluded entirely.
- **Ranking and sets.** `row_gap` = mean activation on positive rows minus mean on negative
  rows. Selected = top 5 by `row_gap`. Controls = `control5_active`: the lowest |row_gap| among
  features with activity ≥ 0.002, excluding the top 5 (historical rule).
- **Guards.** At least 2 discovery users must contribute positive rows. A non-finite gap is an
  error.

## 6. Pilot populations

- **Receivers.** Up to 6 corrected-positive days per confirmation user, in
  sha256(`pilot-calendar-v1-2026-09-28|recv|example_id`) order.
- **Matched days.** For each receiver, the same user's nearest **window-negative** day (earlier
  on a tie), each used once. A user without one is excluded, with the reason recorded.
- **Donors.** For each department, up to 8 **held-out benign (val-split, non-insider)** users
  in hash order, one day each in hash order. Official insiders are never donors (the historical
  Pilot 1 used 7 insider days as donors; review note). A receiver whose department has no donor
  is excluded, with the reason recorded.
- **Record keeping.** Every candidate user is listed with included / excluded and a reason.

## 7. Pilot conditions, magnitudes and endpoints

- **Engine.** Pilot 1's `scripts/pilot_intervention_specificity.py`, with corrected-label
  assertions and an alpha option.
- **Conditions, in scoring order:**
  - `zero`;
  - `rpU_S`, `randI_S`, `randD_S`, `rpU_C`, `randI_C`, `randD_C`;
  - `recon_S`, `recon_C`, `orig_S`, `orig_C`, `rpO_S`, `rpO_C`;
  - `rpU_S@0.5`, `rpU_C@0.5`.
- **Magnitudes.** Alpha 1 is primary; alpha 0.5 is a declared dose check. Edited codes are
  z + α(prototype − z) on edited coordinates.
- **Scoring rules.** All conditions are scored over identical batches (batch 12). Baselines are
  the `zero` condition in the same run, never a cached score.

**Gates (hard):**
- delta rows equal tokenizer length for every receiver;
- zero edit vs unhooked model ≤ 1e-5;
- `orig` = `rpU` + reconstruction error ≤ 1e-4 on the checked receivers;
- every donor is a non-insider;
- every receiver and matched day agrees with the corrected labels.

**Endpoints** (behavior_only; per receiver the patched-minus-zero mean over behavior targets;
per user the mean over receivers; estimate = mean of user means; 95% user-cluster percentile
bootstrap, 10,000 draws, seed 42):

- **Primary:** `rpU_S − randI_S` on attack days.
- **Co-primary:** `rpU_S − rpU_C` on attack days.
- **Specificity:** [`rpU_S − rpU_C`](attack) − [same](matched window-negative), within user.
- **Secondary:**
  - `rpU_S − randD_S` and `rpU_C − randI_C`;
  - `rpU_S − rpO_S` (writing into zero coordinates);
  - `orig_S − rpU_S − recon_S`;
  - the dose ratio `rpU_S@0.5` / `rpU_S@1`.
- **Diagnostics, named separately:** full and profile_only deltas for every condition; edit
  norms and supports (`edit_balance.json`); share of receivers with any edit.
- **Heterogeneity:** per user, by scenario, and fresh vs exposed confirmation users.

**Interpretation rules.**
- Report every estimate with its interval, including nulls, harmful signs and heterogeneity.
- No negligibility or equivalence claims.
- No change of features, cohorts, donors, score definition or magnitudes after results.
- A positive primary means the selected directions change behavior-token prediction more than
  matched random directions on these days, and nothing more.

## 8. Compute caps (recorded before execution)

Collaborator allocation `cis260991-gpu` (standing rule), A100 40 GB. An atomic `mkdir` lock
prevents duplicate execution.

| job | partition, wall limit | cap | expected |
|---|---|---|---|
| B1 bridge + class extension | gpu-debug, 30 min | 0.5 GPU-h | ~0.15 |
| B2 dictionary refit + ranking | gpu, 75 min | 1.25 GPU-h | ~0.55 |
| B3 delta cache for the pilot population | CPU (tra250034 shared), 60 min | 0 GPU | — |
| B4 pilot scoring | gpu-debug, 30 min | 0.5 GPU-h | ~0.3 |
| **total** | | **2.25 GPU-h hard cap** | ~1.0 |

If any gate fails, the dependent jobs are not submitted, and the failure is reported with its
evidence.

## Amendment A1 (2026-09-28 21:40 CDT, before any Stage B/C result exists)

**Why.** The collaborator's Anvil queues:
- **gpu and ai partitions:** estimated starts of 2026-10-15 to 2026-10-18.
- **CPU shared:** 2026-12-26.
- **gpu-debug and CPU debug:** 10–19 hours per job, one running job per user.
- **Moving the cached deltas instead:** the 2M presampled rows are 16 GB, and the
  Anvil→laptop→Aquaman route measures about 1.5 MB/s.

**Change.** Sections 3 and 5–7 run on Aquaman (2× RTX 3070 8 GB, `cert-venv`: torch
2.5.1+cu121, transformers 5.14.1, peft 0.19.1, bitsandbytes 0.49.2). They run on a **separately
versioned layer-26 token-delta cache, `aquaman-l26-v1`**:
- **Inputs.** The same adapter (sha256 `e0bf5112…`) and the same r4.2 `eval.jsonl` (sha256
  `9266ae43…`), 42,468 examples.
- **Extractor settings.** `scripts/extract_adapter_deltas.py` with the base pass through
  `disable_adapter()` (one model copy); layer 26 only; token unit; chunks of 512 examples; batch
  size fixed by the smoke test before the full run; two GPU shards at chunk-aligned global indices.
- **Downstream.** The dictionary refit, ranking, pilot delta cache and pilot all use this one
  cache and this machine, so the pilot's codes, edits and live model come from one configuration.
- **Unchanged.** Every rule of sections 3–7: sampling, seeds, discovery/confirmation, negatives,
  populations, conditions, endpoints, interpretation.
- **B1** (the A100 bridge and class-cache extension) stays queued on Anvil. It cannot move,
  because the bridge is defined as bit-exact on the A100 configuration.
- **Consequences, recorded as such.** Costs are measured on RTX 3070s and projected to A100 from
  the historical A100 rates. The new cache is compared descriptively with the historical A100
  cache (per-example NLL agreement, and deltas where available). It is not claimed to equal it.
