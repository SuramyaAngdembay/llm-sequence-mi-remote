# Hypothesis ledger — interpretability exploration round 1 (2026-09-25)

Written before any pilot in this round was launched. Every branch proposed,
run, stopped or failed is recorded here, including nulls. Everything in this
round is **exploratory**: the CERT confirmation receivers were already
inspected in package 4, so no result here is a fresh confirmation.

Central question (task specification): what internal computations distinguish
identity familiarity, ordinary behavioural prediction and context-dependent
anomalous behaviour, and can the analysis tell them apart?

Frozen objects for every pilot: Qwen3-8B r4.2 adapter
(`pkg4_share/adapter`, fingerprint-checked), discovery-split SAE at layer 26
(m=2, k=4; `pkg4_share/frontier`, ranking sha256 recorded in the package-4
manifest), selected set `[4596, 7693, 2302, 3673, 3455]`, control set
`[6596, 8017, 6608, 2765, 886]`, the 30 confirmation users.

Classification: **M** = possible methodological contribution, **E** =
empirical discovery, **V** = validation/correction. Controls and established
patching techniques are not counted as new methods.

| id | question | competing explanation | distinguishing experiment | controls | cost | closest work | stopping rule | class | status |
|---|---|---|---|---|---|---|---|---|---|
| H1 | Do the selected directions change behaviour-token loss beyond what the actual perturbation size and edit positions predict? | Generic disruption: any vector of that norm at those positions raises loss as much. | Pilot 1. Residual-preserving edits `h + dec(z') − dec(z)` against norm- and position-matched random directions (isotropic, and drawn from other dictionary directions). Same frozen model, same receivers, same batches. | Measured decoded edit norms; token support and token classes; true zero edit; matched benign receivers | ≤ 0.5 GPU-h | Zhang & Nanda 2023; Heimersheim & Nanda 2024 (patching practice); Marks et al. 2024, Sparse Feature Circuits | If selected minus matched-random is within ±20% of the selected effect with an interval covering 0: no direction-specific effect; stop the feature-specific branch. | V (+E) | selected, run |
| H2 | Is the package-4 effect a property of the model or of the patch implementation? | The union-of-active-positions edit writes donor values into coordinates that were zero, an off-distribution "dormant direction" injection; or reconstruction interacts nonlinearly with the edit. | Pilot 1 conditions: original procedure, residual-preserving with union support, residual-preserving with each feature's own support, and single-feature own-support edits (joint versus sum of singles). | As H1 | inside Pilot 1 | Makelov, Lange & Nanda 2023 (subspace-patching illusions via dormant directions) | Report the decomposition whatever it shows. No stopping rule is needed for a decomposition. | V | selected, run |
| H3 | Does fine-tuning make behavioural predictions depend on the profile, and do the selected features read that profile? | The base model already conditions on the profile text (ordinary in-context dependence). Or the features respond to session content only. | Pilot 2. Swap the profile lines (DAY/PSY) for a same-department or a different-department user's, keeping `week` and every session line. Measure behaviour-token loss under the base (adapter off) and adapted model, and the selected/control feature activations on session tokens. | Base versus adapted on identical inputs; same-department versus different-department partner; malicious versus matched benign days; on-the-fly δ checked against the cache | ≤ 0.5 GPU-h | Prakash et al. 2024 (fine-tuning enhances existing mechanisms); Jain et al. 2024; Vig et al. 2020 (causal mediation) | If adapted-minus-base swap sensitivity has an interval covering 0 and the selected features' swap response equals the controls': no adaptation-specific identity pathway at this resolution; do not run H6. | E | selected, run |
| H4 | Do the high-gap features track the adapted model's own surprise rather than malicious content? (unconventional) | They encode session content that happens to be rarer on malicious days. | Pilot 3 (CPU, on Pilot 2's saved per-token data). Within-day association between a feature's activation at t and the adapted model's loss at t+1, on benign and malicious days separately, against controls; base-model loss as an intrinsic-difficulty alternative. | Per-user association, controls, base-model loss | 0 GPU | Stolfo et al. 2024 (confidence-regulation neurons); Gurnee et al. 2024 (universal/entropy neurons) | If association with adapted loss is no stronger than with base loss, or no stronger than for controls: no surprise-tracking claim. | E | selected, run |
| H5 | Do other SAE seeds find functionally equivalent directions, beyond decoder alignment? | Similar directions but different causal roles (same location, different function). | Repeat Pilot 1's residual-preserving edits with seed-43/44 dictionaries' best-matching features. | Decoder-alignment baseline (already 0.88–0.96 against 0.64 chance) | ~0.5 GPU-h | Paulo & Belrose 2025 (different features across seeds); Lan et al. 2024 | Not run in round 1: a functional comparison is moot unless H1 finds direction-specific effects. | V | proposed, deferred |
| H7 | *(added 2026-09-25 before any scoring; see attempt log)* Does the adapter key its behaviour predictions on identities it was trained on? | Profile text changes predictions only through generic attribute values (department, role, scores), identically for familiar and unfamiliar people. | Inside Pilot 2: also swap in the DAY/PSY lines of a same-department **training** user (familiar to the adapter) and compare with the same-department eval user (unfamiliar). Contrast: [swap_train − swap_same] behaviour loss, adapted minus base. | Base model (never saw either user); same department for both partners; malicious and matched benign days | inside Pilot 2 | Carlini et al. 2021 and Tirumala et al. 2022 (memorization in LMs); Geva et al. 2021 (feed-forward layers as key-value memories) | If the adapted-minus-base contrast has an interval covering 0: no evidence of identity-keyed prediction at the level of behaviour-token loss. | E | selected, run |
| H6 | Through which layers does profile information reach session positions? | Direct attention from session tokens at the patched layer, or early mixing. | Path patching of profile-position residuals from swapped runs into original runs, across a small layer set. | As H3 | ~0.5 GPU-h | Goldowsky-Dill et al. 2023 (path patching); Geva et al. 2023 (attention knockout); Wang et al. 2022 | Run only if H3 finds an adaptation-specific profile dependence. | E | proposed, conditional |

Not proposed in this round: detector comparisons (tabular models), PCA baselines and new
training runs. None distinguishes the mechanistic hypotheses above.

## Budget (8 aggregate GPU-hours cap for this round)

| item | partition | wall limit | budget |
|---|---|---|---|
| Pilot 1 (validity phase + main) | gpu-debug, 1 A100 | 30 min | 0.5 GPU-h |
| Pilot 2 (validity phase + main) | gpu-debug, 1 A100 | 30 min | 0.5 GPU-h |
| Reruns if a job fails or times out | gpu-debug | 30 min each | ≤ 2 GPU-h |
| **Round total planned** | | | **≤ 3 GPU-h** |

## Attempt log

(dated entries appended as the round proceeds)

**2026-09-25, inventory.** No commits since `99c3c40`; no queued or running jobs
on either account. The package-4 rows are unchanged (sha256 in
`results/cert_package4_token_class/INPUT_HASHES.txt`).

**2026-09-25, populations frozen before any pilot scoring**
(`scripts/pilot_select_populations.py`, salt `pilot-2026-09-25`). Up to 10
attack days per confirmation user in hash order: 207 malicious days, 30 users,
each with the same user's nearest benign day (207). Donor policy: up to 8 benign
days of distinct never-malicious users per department (hash order). Department
13 has no never-malicious eval user, so its only receiver user (JJM0203, 10
pairs) has no donor policy and is excluded from Pilot 1 (rule fixed before
scoring); in Pilot 2 that user lacks only the same-department swap.

**2026-09-25, H7 added before any scoring.** Checking the split showed that the
149 eval users and the 851 training users are disjoint: every confirmation user
and every swap partner is unfamiliar to the adapter, so the planned swaps
cannot test identity familiarity, which is part of the central question. The
selector gained `--train-jsonl` (a same-department training user per receiver
user, hash order, one of that user's training days). All previously frozen keys
are identical in the regenerated manifest
`manifests/pilots_2026_09_25/pilot_populations.json` (sha256 `e75204b0…`); the
earlier file was `1afd6efd…`.

**2026-09-25, delta cache.** The layer-26 deltas of the 460 pilot examples
(137,068 token rows) were copied unchanged (float16) from 60 of the 83 cache
chunks into one file (`scripts/pilot_cache_deltas.py`; metadata
`manifests/pilots_2026_09_25/pilot_deltas_l26.json`, sha256 `9bdb5447…`), so
the GPU jobs do not read 66 GB of pickled chunks. Rows per example equal the
extraction's token counts.

**2026-09-25, Pilot 1 CPU dry run (no language model; before any scoring).**
`results/pilots_2026_09_25/pilot1_dryrun/`. Validity: delta rows equal the
tokenizer length for all 394 receivers; right padding; `orig = rpU + (dec(z) −
δ)` to 9.5e-07. Measured edits (mean per receiver day, benign / malicious):

| condition | days edited | tokens edited | total L2 (residual units) | L2 per edited token | DAY / PSY / SES tokens |
|---|---|---|---|---|---|
| rpU_S (union) | 100% | 3.5 / 3.6 | 203 / 210 | 102 / 103 | 1.2 / 0.1 / 2.2–2.4 |
| rpO_S (own) | 100% | 3.5 / 3.6 | 155 / 160 | 77 / 78 | same |
| rpU_C | 100% | 4.8 | 49 / 52 | 17 / 19 | 1.1 / 0.8 / 2.6–2.7 |
| recon (both sets) | 100% | all (~293) | 1,005 / 1,107 | 39 / 41 | all |
| single 2302 | 100% | 1.3 | 73 | 63 | 1.2 / 0.1 / 0.0 |
| single 4596 / 3673 / 3455 | 56–80% | 0.7–0.8 | 61–74 | 61–68 | SES |
| single 7693 | 0.5% | 0.005 | ≈0 | — | — |

Observations recorded before scoring: (i) feature 2302 is active on about one
DAY token of every day, and its department prototypes are 17–27 while the other
selected prototypes are 0–3; (ii) under union support the edit therefore also
writes 2302's prototype into session positions where it was zero, which is the
difference between rpU_S and rpO_S; (iii) selected edits are about 4× the
control edits in total norm, so E3 (selected versus control) is size-confounded
and E1 (versus norm- and position-matched random directions) is the
specificity test; (iv) the reconstruction replacement of the published
procedure is about five times larger than the selected edit; (v) 7693 does not
fire on these receivers, so the joint set is effectively four features here.

**2026-09-25, changes after the dry run and before submission.** Added a
per-feature record of donor-minus-receiver coefficient changes (own and union
support; `coefficient_changes.json`), because the package-4 prototype averages
each feature over tokens where *any* set feature is active, and the dry run
suggests that 2302's DAY activity dilutes the other features' prototypes, which
would make "toward a benign colleague" mostly a reduction for session
features. Conditions are now scored in priority order (E1–E3 first). Pilot 2's
delta validity rule: abort on a shape mismatch or a median relative error above
0.5 (a wrong layer or misalignment); the earlier 0.1 threshold would have
aborted on bf16 batch-shape noise, which a difference of two hidden states
amplifies. The median and 95th-percentile error and the SAE active-set Jaccard
are recorded.

A login-node process-listing mistake caused a second, 12-pair dry run to
write into the same directory as the still-running full dry run. The full
run's files are the later ones (04:02 EDT, 394 receivers); the small run
completed first and changed nothing that is reported.

**2026-09-25, submitted.** Pilot 1 = job 20907022, Pilot 2 = job 20907024
(gpu-debug, `cis260991-gpu`, 1 A100, 30 min each; code at commit `e77b1d6`,
md5-checked on the cluster). Queue estimate at submission: a week, against
8–24 h observed for recent gpu-debug jobs; the H100 partition estimate was
later still and its balance is reserved for Phase C, so the jobs were left in
place.

**2026-09-25, while queued (no output exists).** Pilot 2 now also saves the
token ids of the original inputs in `pilot2_orig_tokens.npz`. Reason: the H4
analysis must compare feature activity with next-token loss *within the same
predicted field*, and without token ids it cannot separate a surprise-tracking
feature from one that fires on intrinsically hard fields. Nothing else in the
job changed.

**2026-09-25, interpretation map (written while both jobs are queued; no
pilot output exists).** What each outcome would and would not mean. All
readings stay exploratory; the receivers were inspected in package 4.

*H2 sharpened (from the dry-run balance, before results).* Under union
support, every day's organisation-line token carries feature 2302, so every
session token where another selected feature fires also receives 2302's
department prototype (17–27, about ten times the other prototypes). And
because the package-4 prototype averages each feature over tokens where *any*
set feature fires, the session features' prototypes are diluted by those
organisation tokens, so their "move toward a benign colleague" edits are mostly
reductions. Alternative explanation of the package-4 behaviour effect: it is
driven by injecting an organisation-line direction into session positions
(a dormant-direction write in the sense of Makelov et al. 2023), not by
editing the session features. Predictions if true: E5 (rpU_S − rpO_S) carries
most of rpU_S's behaviour effect; rpO_S and the single session-feature edits
are small; single_2302 (own support: organisation tokens only) has a
behaviour effect no larger than its random-direction equivalent.
`coefficient_changes.json` shows the sign and size of the coefficient changes.

| outcome | reading | not licensed |
|---|---|---|
| E1a and E1b > 0, intervals exclude 0, beyond 20% of rpU_S | the selected joint edit changes behaviour loss more than equally large random and other-dictionary perturbations at the same positions | feature semantics; anomaly specificity (needs E6); a circuit |
| E1a > 0 but E1b ≈ 0 | an on-dictionary direction effect, not specific to these features | "these features matter" |
| E1a and E1b within ±20% and covering 0 | generic disruption explains the effect; stop the feature-specific branch (declared rule) | anything feature-specific |
| E4 ≠ 0 | reconstruction interacts with the edit, so package-4 contrasts contain an interaction term | a sign for the interaction elsewhere |
| E5 large, rpO_S small | the package-4 effect is mainly an implementation effect of union support (H2 sharpened) | a model mechanism |
| E6 ≈ 0 with E1 > 0 | the direction effect is not specific to malicious days | anomaly specificity |
| E7 ≠ 0 | the downstream response to the summed single edits is non-additive (the hidden-state edits themselves add exactly) | which features interact, without further decomposition |
| F1 > 0 (adapted − base) | adaptation increased how much behaviour predictions depend on the profile lines | where in the network this happens (H6) |
| F1 ≈ 0 with both Δ > 0 | the base model already conditions on the profile; adaptation did not change the dependence | "the adapter introduced identity use" |
| F4 > 0 (adapted − base) | a familiar training profile changes behaviour predictions more than an unfamiliar same-department one: evidence of identity-keyed prediction | memorisation of specific sessions |
| F3 selected > control | the selected features' activity on session tokens depends on the preceding profile (represented context) | that this dependence causes the behaviour-loss change (needs mediation, H6) |
| H4: within-field association with adapted loss > base loss and > controls | the features co-occur with the adapted model's own surprise beyond field difficulty | causation in either direction |
