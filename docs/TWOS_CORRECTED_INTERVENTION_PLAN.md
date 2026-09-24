# TWOS corrected intervention — plan, fixed before the run (2026-09-24)

This plan was committed before the run it describes was submitted. Nothing
below was chosen after seeing its outcome.

## Why it exists

The two TWOS token-class runs of 2026-09-19 (`results/twos_token_class_causal/`,
`results/twos_token_class_causal_s43/`) patched `top5 = [0, 1, 2, 3, 4]`. Their
run scripts (`/data/suramya/insider_mi/lm/run_twos_pkg4.sh`, `..._s43.sh` on
Aquaman) pointed `--frontier-dir` at the benign-only SAE frontiers
`twos_work/v3_sae_s42` and `v3_sae_s43`. Those frontiers ranked features on
their own benign-only training rows, so every gap was NaN, and sorting NaN
gaps returns feature ids in order. The controls were then the five most active
features, because the "low gap" key was undefined too. The original TWOS
intervention (`twos_work/v3_causal_s42`) did not have this defect: it read the
re-ranked frontier `v3_audit_s42/reselect_l24`, with finite gaps and
`top5 = [6036, 7375, 3218, 417, 22]`. That ranking used all 16 positive users,
however, so it was not held out from its own evaluation.

The re-runs' header comment claimed every parameter except the class schema
matched `v3_causal_s42`. Besides the frontier, they differed in batch sizes and
omitted `--exclude-same-user-donors`. The seed-43 comment cites a
`v3_causal_s43` run that never existed.

## Question

On the seed-42 TWOS adapter, do the five features most associated with attack
rows **in discovery users** change the per-class loss of **held-out**
confirmation users' attack rows differently from the control set, beyond what
the SAE reconstruction alone does?

It does not answer: stability across seeds; anything about CERT; what
information the features represent.

## Populations (exact files and hashes)

- Split: the 16 positive users sorted by `sha256("twos-discovery-v1|" + user)`;
  first 8 discovery, last 8 confirmation (`split_manifest.json`).
  - discovery: User1, User13, User17, User2, User20, User21, User3, User7
    (70 positive examples); file sha256 `44b1f032…`
  - confirmation: User12, User14, User15, User18, User4, User6, User8, User9
    (68 positive examples); file sha256 `a17e75c4…`
  - the 8 benign-only users are used only as benign rows and donors.
- Ranking (`reselect_token_sae_features.py`, CPU, Aquaman): discovery positives
  (7,539 rows, all 8 discovery users) against a 25% benign sample, **excluding
  every row of the confirmation users** (7,070 positive and 18,820 benign rows
  dropped). SAE: `v3_sae_s42/layer_24/m04_k08`, unchanged (benign-fitted).
  Ranking file sha256 `12b10690…`.

## Gates passed before submission

| gate | result |
|---|---|
| ranking population has both classes | 7,539 positive, 34,747 benign rows |
| positive rows come from ≥ 2 discovery users | 8 of 8 |
| every ranking statistic finite | 8,192 of 8,192 |
| top5 is the top of the gap ranking | `[6036, 7375, 1197, 7420, 3218]`, gaps 0.128 to 0.077 |
| control5_active meets its stated rule | `[7962, 972, 8082, 7232, 3157]`, gaps −4e-06 to −2.7e-05, each active ≥ 0.2% |
| control activity relative to selected | mean ratio 1.03; individual features 0.21% to 0.80% versus 0.24% to 0.81% |
| receivers disjoint from discovery | checked in the run by `build_selection_manifest` |

## Run

`slurm/anvil_friend/twos_bounded_corrected.sbatch`, collaborator account
`cis260991-gpu`, `gpu-debug`, 1 GPU, 30 min wall (≤ 0.5 SU). Layer 24, m=4,
k=8, context mode `team`, `top5` versus `control5_active`, alphas **0.0 and
1.0**, up to 16 candidate donors per donor type, same-user donors excluded,
`--token-class-schema cert`, receivers restricted to the confirmation users.
Alpha 0 decodes the unedited sparse code: it is the reconstruction-only
control for each arm, on exactly the receivers that arm patches (a receiver
day is patched, all tokens reconstructed, when any token has an active feature
of that arm).

## Pre-specified analysis

`scripts/analyze_intervention_endpoints.py <out> --alpha 1.0`

- **Primary**: benign donors, alpha 1.0, per receiver the mean over all
  candidate donors (not the best one), selected minus control, for the
  `behavior_only` and `profile_only` views; mean of receiver-user means (8
  users), 95% cluster bootstrap over users.
- Reported with it, not instead of it: the unpatched base, both arms' absolute
  deltas, the alpha-0 reconstruction-only deltas, the edit effect net of
  reconstruction for each arm and their difference, anomalous-donor results,
  the direct profile-minus-behaviour contrast, and the historical best-candidate
  donor difference-in-differences for continuity.
- Reading rules: 8 users make every interval descriptive. An interval crossing
  zero is inconclusive, not equivalence; no equivalence margin is declared, so
  no "negligible" claim will be made. Two alphas give no dose-response.

## Gates before any reading

1. Per-batch reconstruction and count assertions pass (the run stops otherwise).
2. `feature_selection_manifest.json` records `held_out_from_selection: true`
   and the ids above.
3. Receivers are the 68 positive examples of the 8 confirmation users.
4. Tokenization on the collaborator environment matches the environment that
   extracted the deltas, record by record (checked on CPU before submission).

## After this run

Only if the gates pass and the run completes: decide whether a seed-43 run
with its own held-out ranking is worth its cost. No further seeds before that.
