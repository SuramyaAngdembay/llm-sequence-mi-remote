# Contamination audit: is this project p-hacking, and which results are clean?

Started 2026-09-19 at Suramya's instruction: research p-hacking properly, then
check every metrics table harshly, mark as clean **only** what is clean of
everything, and recheck rather than assert.

**Status: IN PROGRESS. Several items are unresolved and are marked so.** This
file is not a clearance. It is a ledger.

> **The audit's prediction was tested and confirmed the same day.** This file
> flagged the TWOS package-4 result as exposed on three counts: 16 clusters, 24
> uncorrected intervals, and a single seed. An independent seed-43 replication
> then **reversed every sign**, and the directional claim has been withdrawn.
> See `results/twos_token_class_causal/README.md`. This is the strongest
> evidence in the project that these exposures are not theoretical.

## Standard applied

From the primary literature rather than from intuition:

* **Researcher degrees of freedom** (Simmons, Nelson & Simonsohn 2011): flexible
  choice of outcome measure, sample size, exclusions, covariates, subgroups, and
  which conditions get reported. Their finding is that ordinary, well-intentioned
  flexibility pushes the false-positive rate far above its nominal level.
* **The garden of forking paths** (Gelman & Loken 2014): the inflation does not
  require fishing. It is enough that the analysis chosen was *contingent on the
  data seen*, even if only one analysis was ever run.
* **Test-set reuse / adaptive overfitting** (ML-specific): a fixed evaluation set
  consulted repeatedly across months, with each experiment chosen in response to
  the last, is the same pathology under a different name.
* **Selection on the dependent variable**: choosing a configuration by the sign or
  size of the effect it produces.

Each result family below is checked against all four. **A family is marked CLEAN
only if it passes all four and nothing about it is unverifiable.**

## Verdict summary

| result family | arithmetic | selection on outcome | multiplicity | test-set reuse | verdict |
|---|---|---|---|---|---|
| History-prefix probe (pkg 3) | **verified** | none in-run | 72 uncorrected intervals | inherits project-wide reuse | **QUALIFIED** |
| TWOS token-class causal (pkg 4) | **verified** | none in-run (config matched) | 24 uncorrected intervals | TWOS reused separately | **CLAIM WITHDRAWN — failed replication** |
| Time-series probe (pkg 2) | **verified** | none found | 12 uncorrected, 4 clusters | new data build | **QUALIFIED** |
| r4.2 SAE causal headline | not yet recomputed | **YES — documented** | 12 configs compared | 40+ evaluations | **CONTAMINATED (selection)** |
| r6.2 SAE causal headline | not yet recomputed | likely same rule | 12 configs | 40+ evaluations | **UNVERIFIED** |
| r4.2 scoring-mitigation (pkg 1) | fold CSVs reproduce to 3.6e-15 | n/a | 60 folds | raw scores never re-verified | **QUALIFIED** |
| LANL replication | not yet checked | unknown | unknown | unknown | **UNVERIFIED** |

**No family is marked CLEAN.** See "Why nothing is clean" below.

## Definite checks performed (recomputation, not reading)

These were recomputed from raw per-row artifacts by an independently written
script, not read from the write-up:

| what | result |
|---|---|
| All 17 published history-prefix figures (means + bootstrap intervals), across 4 runs | **0 mismatches** |
| All 24 published TWOS top5-minus-control contrasts, across 6 views × 4 alphas | **0 mismatches** |
| The 4 TWOS intervals quoted in prose at α = 1.0 | **0 mismatches**, agreeing to 5 decimals |

Arithmetic is therefore not the problem. Every problem below is about *which*
numbers were computed and *why those*.

## CONTAMINATED — selection on the dependent variable

**`docs/HANDOFF_2026-07-05_R42_NATIVE_TOKEN_SEARCH.md`, "Selection Rule" and
"Success Criterion", lines 168–190.** Quoted:

> Choose them by: 1. frontier proxy selectivity 2. control activity quality
> 3. then full uncapped causal patching

> The r4.2 native remote search is successful only if at least one remote
> token-causal config becomes: positive on `top_minus_control_advantage` …
> beat `0.0` robustly first

This is selection on the outcome, written down as policy. The search space was
**12 configurations** for the 8B (layers 18/26/34 × latent_mult 2/4 × k 4/8), and
the winner — layer 26, latent_mult 2, k 4 — is the headline configuration. The
3B pilot frontier compared **36** configurations. The frontier summary carries
**8 label-dependent selectivity columns**, including
`top5_minus_control3_advantage_proxy`, which is a preview of the very contrast
the paper reports.

Being written down is not a defence; it is what makes the finding checkable.

**The intended mitigation, and its current status.** The project does have a
discovery/confirmation split of malicious users
(`scripts/make_positive_user_split.py`, whose docstring says it exists "to
separate mechanism discovery (feature ranking, config choice) from"
confirmation), and the confirmation submitter
(`scripts/submit_r42_confirmation_anvil.sh`) exports
`RECEIVER_USER_FILE=…/confirmation_users.txt`. If the selection used discovery
users only and the headline uses confirmation users only, the headline survives.

**That cannot currently be verified, and one plausible check turned out to prove
nothing.** Both the confirmation run and the earlier unrestricted run at the same
configuration report `n_positive_receivers: 1309`. Reading the code
(`eval_token_delta_sae_causal.py:1298`) shows that field is computed from the
full example table **before** the receiver filter is applied at line 432, so the
equality is expected and is *not* evidence of contamination. The indirect
evidence that the restriction was applied is that `candidate_rows` falls from
1,086,704 to 530,792, close to half. That is consistent but not proof.

**Provenance gap found:** no job record in `results/` captures
`receiver_user_file`. The run scripts write a JSON of their parameters and that
key is absent from all of them, so no archived artifact states which users a run
scored. The proof exists only in the Anvil job log, which prints
`[receivers] restricted to N users from FILE` (line 980) and is unreachable.

## Why nothing is marked CLEAN

**Accumulated evaluation-set reuse.** The repository records **36 distinct Anvil
jobs**, **37 distinct job ids**, **129 result directories** and **38 r4.2 summary
files**. The CERT evaluation data has been consulted on the order of forty-plus
times across months, with each experiment chosen in response to the previous
one. This is the textbook adaptive-overfitting setting, and it applies to every
CERT number in the project including the ones whose arithmetic I verified today.

This does not mean the results are wrong. Empirical work on benchmark reuse finds
adaptive overfitting is often milder than theory predicts. It does mean no CERT
result can honestly be called clean of everything on internal evidence alone.
The only instrument that settles it is a genuinely fresh evaluation set or an
external replication.

## A prediction of this audit, tested

The TWOS package-4 run was marked QUALIFIED on 16 clusters, 24 uncorrected
intervals and one seed. Replicating it on seed 43 — different adapter, delta
cache and SAE; identical configuration, data, receivers and candidate-row count
— produced:

| view | seed 42 | seed 43 |
|---|---|---|
| profile only | −0.0074 [−0.0149, −0.0005] | **+0.0109** [+0.0024, +0.0203] |
| behaviour only | −0.0001 [−0.0005, +0.0002] | **−0.0024** [−0.0053, −0.0003] |
| full | −0.0032 [−0.0062, −0.0003] | **+0.0030** [+0.0003, +0.0065] |

Every sign reverses, and several intervals exclude zero in **both** directions.
The directional claim is withdrawn.

The casualty beyond that result: **a monotone dose-response across patch
strength no longer counts as evidence of a real causal handle here.** It was the
strongest argument offered for seed 42. Seed 43 produces an equally clean
dose-response the other way.

## Selection surface, enumerated

A full inventory of the project's reported tables gives the size of the garden.
Disclosed selection dimensions and their counts:

| dimension | values searched |
|---|---|
| SAE configs | 36 (3B pilot), 12 (8B r6.2 frontier), 8 (r4.2, stated in the paper), 4 (r4.2 native search) |
| layers | {12,18,24} 3B · {18,26,34} r6.2 · {18,26} r4.2 · {12,18,24} TWOS |
| latent_mult / k | {2,4,8} × {4,8,16,32} |
| control sets | control1, control3 (found inert), control5_active, control5_matched |
| context modes | 4–5 per dataset |
| alphas | 4, all reported |
| SAE seeds | 42, 43, 44 |
| adapter seeds | 42, 43 (3B, TWOS); one seed per arm for the 8B factorial and LANL |
| token-class views | 6 |
| serialization conditions | 4 CERT factorial · 4 LANL · 3 masking |

**In the project's favour:** the manuscript states its search grids explicitly
(layers {18,26,34} × m{2,4} × k{4,8}), reports all four alphas and all context
modes rather than a winner, and `docs/FINAL_TABLE_ROW_POLICY.md` pre-declares
which result files are paper-safe and which are superseded. That is real
disclosure, and it is what made this audit possible at all.

**Against:** the r4.2 configuration was still chosen by a rule that reads the
outcome, and single-seed arms — the 8B factorial and LANL both use one adapter
seed per condition — are now known to be exactly where this project's
measurements are unstable.

## Additional exposures found

* **Multiplicity, uncorrected, everywhere.** The TWOS table reports 24 intervals;
  the history-prefix READMEs report 72 across four runs. No correction is applied
  and none is mentioned. Effects like psychometric B−A (−3.2 nats, ~100× the
  interval width) are unaffected by multiplicity. Small ones near the boundary —
  `behavior_ses_only` B−A at −0.002 [−0.002, −0.001] — are exactly the kind that
  multiplicity manufactures, and should not be cited as findings.
* **An undisclosed pilot.** A 40-user history-prefix pilot ran on the same
  evaluation data before the reported 200-user runs, and is not mentioned in
  `results/history_prefix/README.md`. It informed both the decision to scale up
  and the decision to add the week variant. Under Simmons et al.'s disclosure
  rule this belongs in the write-up. **Action owed.**
* **Post-hoc protocol change**, already recorded: the week-prefix variant was
  written after an unfavourable day-view result.
* **16 clusters (TWOS) and 4 clusters (time-series)** for bootstrap intervals.
  Both are too few for inferential reading; both are labelled descriptive. The
  TWOS replication failure shows what "too few" costs in practice.
* **Single-seed arms elsewhere.** The 8B CERT factorial and the LANL replication
  each use one adapter seed per condition. Given that a seed change reversed the
  TWOS result, these are now priority candidates for replication, not
  afterthoughts.

## Checked and found NOT contaminated

* **No dropped variants in this session's work.** The Aquaman run directories
  contain exactly the runs reported plus clearly-labelled smoke and pilot runs:
  time-series has `w1`, `w7`, `fc7` and all three are reported including `fc7`,
  which has the worst user ROC. No silent variant selection.
* **The TWOS package-4 configuration was matched, not chosen.** Layer 24,
  latent_mult 4, k 8 were read from the prior run's summary JSON rather than
  searched, so that run contains no selection on outcome. The *original* TWOS
  configuration choice is a separate, unaudited question.
* **A false-positive trap avoided.** `paper/main.tex:978` matches the wording of
  a withdrawn claim but is a different claim about a different experiment.

## Resolved 2026-09-21, when Anvil returned

Three open items closed by direct check, and one new finding that the checks
produced.

**CLOSED — the discovery/confirmation split is real and disjoint.**
`comm -12` on the two user files: **30 discovery users, 30 confirmation users,
0 overlap, union 60.** The split exists and partitions the malicious users
exactly.

**CLOSED — the confirmation run really was restricted.** Its job log contains
the line this audit said was the only thing that would settle it:

```
[receivers] restricted to 30 users from .../user_splits_r42/confirmation_users.txt
```

The earlier suspicion, raised because both runs reported `n_positive_receivers:
1309`, is fully retired: that field is a pre-filter count, and the log proves
the filter applied.

**CLOSED — adapter provenance verified, not assumed.** Content fingerprints of
the Anvil and Aquaman adapters:

| adapter | weights digest | verdict |
|---|---|---|
| r4.2 | `49bbbd4b0bf0cfe7…` both sides | **identical weights and configuration** |
| r6.2 | `5a6c2ffa1ffa8ea6…` both sides | **identical weights and configuration** |

Work package 3 ran on the same adapters that produced the published numbers.
That criterion now passes for every result built on them.

**NEW FINDING — the split protected feature selection but NOT configuration
selection.** Checking `[receivers] restricted` across the **25** archived causal
job logs:

| runs | receiver restriction |
|---|---|
| job 19379904 (the confirmation run) | restricted to 30 users |
| **the other 24, including the entire configuration search** | **ALL receivers** |

So the layer/latent_mult/k choice — 1 of 12 in the frontier, then 1 of 4 in the
native search — was made from runs that scored **all 60 malicious users**,
including the 30 later designated confirmation. The confirmation run is a valid
held-out test of the *features*, which were re-selected on discovery users, and
is **not** a held-out test of the *configuration*, which had already seen those
users.

This is weaker than "contaminated" and stronger than "clean". Stated precisely:
the r4.2 mechanistic headline rests on features selected on held-out-disjoint
users, evaluated on 30 users never used for feature selection, under a
configuration chosen with those users visible.

## Open items, with what would settle each

| item | what settles it | blocked on |
|---|---|---|
| Whether frontier selectivity proxies used discovery users only | the frontier job's user file | **answered indirectly: no causal job except the confirmation restricted receivers** |
| Outputs of jobs 20827646 (r4.2 portability) and 20827647 (LANL) | reading them | the other account — `/anvil/scratch/x-bbhusal1` is mode 700 and unreadable from x-sangdembay |
| r4.2 scoring-mitigation raw scores | re-score and compare | Anvil — now reachable, not yet run |
| LANL replication | not yet examined | — |
| r6.2 configuration selection rule | find the equivalent handoff doc | — |

## Actions taken

1. **Done.** The causal sbatch template now records `receiver_user_file` and
   `exclude_same_user_donors` in the job JSON, so future runs state which users
   they scored. This does not repair past records.
2. **Done.** `results/history_prefix/README.md` now carries an
   "Everything that was run" table disclosing the 8-example smoke and the
   40-user pilot, and says plainly that the reported runs are **not independent
   of** the pilot because it prompted both the scale-up and the week variant.
3. **Done.** Multiplicity counts now sit beside every interval table: 72 for
   history-prefix, 24 for TWOS, 12 for the time-series probe, each stating that
   no correction is applied and which specific conclusions are and are not
   robust to that.
4. **Done.** `results/history_prefix/README.md` now states that
   `behavior_ses_only` B−A (−0.002) **should not be cited as a finding** — it is
   exactly the size of effect that 72 uncorrected intervals manufacture.

## What would actually clear this project

Nothing internal. The instruments that would settle it:

1. **A fresh evaluation set** — CERT users never scored during development.
   This is the only real answer to forty-plus adaptive evaluations.
2. **The Anvil log lines**, to prove the discovery/confirmation separation held.
3. **An external replication** of the headline causal result by someone who did
   not run the configuration search.

Until at least the second, the r4.2 mechanistic headline should be described in
the paper as a result whose configuration was selected on a related outcome, with
the discovery/confirmation design stated as the intended control and its
verification stated as pending.
