# Experiment plan after the 2026-09-24 review

All GPU work runs on the collaborator allocation (`ssh anvil-b`). Balances on
2026-09-24: `cis260991-gpu` 374.5 SU, `tra250034-gpu` 94.5, `tra250034-ai`
726.5. Anvil bills 1 SU per GPU-hour. No new seed sweeps, model families or
PCA experiments start until tiers 0 to 2 are resolved.

Each run is listed with the question it answers, what it needs and the gate
before its result is read.

## Tier 0 — already queued, no new spend

| run | question | gate before reading | compute |
|---|---|---|---|
| CERT package-4 smoke **20879548** (gpu-debug) | Do the per-class assertions hold in the collaborator environment? Does that environment reproduce the owner environment's intervention deltas? | In-run assertions pass. `compare_intervention_runs.py` against owner smoke 20840686 gives contrast Δ ≤ 1e-4 and per-row Δ ≤ 1e-3 | ≤ 0.5 SU |
| CERT package-4 full **20879549** (gpu) | Which token classes' losses does the published intervention change? | Smoke gate passed; addendum rules (`PREREGISTRATION_CERT_PACKAGE4.md`); `analyze_intervention_endpoints.py` | ~24 SU (committed) |
| TWOS held-out bounded **20890658** (gpu-debug) | Do discovery-selected features act on held-out confirmation users differently from controls, beyond reconstruction? | Gates in `TWOS_CORRECTED_INTERVENTION_PLAN.md` | ≤ 0.5 SU |

## Tier 1 — CPU, cheap, before any new GPU run

1. ~~Environment gate~~ **Passed 2026-09-24**: 1,186/1,186 rows matched, max per-row delta difference 9.5e-07, contrasts within 6e-09 (`results/cert_package4_env_gate/GATE_REPORT.txt`).
2. **Analyses** of 20879549 and 20890658 with the pre-specified analyzer (CPU).
3. ~~Copy the raw LANL files from purgeable scratch~~ **Done 2026-09-24.**
   The files are in `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/lanl2015_raw/`
   with checksums verified against the source, and a second verified copy is on Aquaman. See
   `docs/provenance/LANL_AND_ADAPTER_PRESERVATION.md`.
4. **LANL option A extraction** of never-sampled ordinary users, chosen by a
   new salted hash (CPU job on `cis260991` or `tra250034` CPU hours, a few SU).
   Gate: every extracted user is absent from `train.jsonl` and `val.jsonl`, and
   the population report passes.

## Scratch purge watch

Anvil deletes scratch files not read for 30 days, without warning, and scratch
is not backed up. Last reads, as of 2026-09-24:

| on scratch | last read | at risk from | action |
|---|---|---|---|
| LANL raw `auth.txt.gz`, `redteam.txt.gz`, `windows_full.jsonl` | 2026-09-07 | ~10-07 | copied to project space and to Aquaman, both checksum-verified |
| LANL adapters user_anon, host_anon, shuffle | 2026-09-11/12 | ~10-11 | copied to project space, checksums verified |
| Phase C 3B `p1/adapter_full` | 2026-09-18 | ~10-18 | copied to project space, checksums verified |
| LANL `adapter_full`, Phase C `adapter_targetmask` | 2026-09-21/23 | ~10-21 | copied to project space, checksums verified |
| staged CERT inputs `pkg4_share` (274 GB, a copy of project-space originals) | 2026-09-21/23 | ~10-21 | the queued job reads them when it starts; re-stage from project space if it slips past mid-October |
| LANL deltas and historical split files | recent | — | not copied: regenerable from `windows_full.jsonl` and the adapters |

## Tier 2 — bounded GPU runs that complete the current questions

| run | question | gate | compute |
|---|---|---|---|
| CERT α = 0 reconstruction control, confirmation receivers | How much of each arm's absolute per-class effect is dictionary reconstruction? This is required before any mechanistic reading of absolute effects. α = 0 does not depend on the donor, so 1 candidate per donor type suffices. | Tier-0 CERT gates passed | ~1 SU |
| TWOS δ-reproduction check (optional, gpu-debug) | Does the Anvil environment reproduce the TWOS deltas cached on Aquaman? It separates environment effects from reconstruction in the α = 0 arm | TWOS run 20890658 finished | ≤ 0.25 SU |
| LANL option A scoring, existing adapter | With user composition matched and no retraining, is ranking worse for users absent from training? Pooled and within-user AUC | Tier-1 extraction gate | ~3–5 SU |

## Tier 3 — only if tiers 0 to 2 leave the question open

| run | question | cost |
|---|---|---|
| TWOS seed 43, held-out ranking + bounded run | Does the corrected TWOS result depend on the adapter seed? Only if 20890658 passes its gates and shows an effect worth testing | ≤ 0.5 SU GPU + CPU ranking |
| LANL option B: salted resample, retrain, rescore | The originally intended random-fold seen/unseen comparison | ~14 SU training + ~5 SU scoring + CPU extraction |
| Phase C 8B training-mask arm | Does removing profile losses from training matter at the scale where the failure occurred? | ~20–25 SU (split 4-GPU training / 1-GPU scoring) |

## Deferred explicitly

- Further seed sweeps.
- New model families.
- A PCA-on-δ baseline for the SAE. It is sensible later, but it does not repair any defect listed here.
