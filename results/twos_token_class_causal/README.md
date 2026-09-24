# Which tokens does an SAE intervention actually repair?

> ## ⛔ INVALID FEATURE SELECTION — correction of 2026-09-24. Read before anything below.
>
> This run patched `top5 = [0, 1, 2, 3, 4]` against `control5_active =
> [505, 5342, 561, 4137, 3967]`. Its script (`lm/run_twos_pkg4.sh` on
> Aquaman) pointed `--frontier-dir` at the **benign-only** frontier
> `twos_work/v3_sae_s42`, which ranked features on benign rows only: all 8,192
> gaps are NaN, so sorting returned ids in order, and the "low-gap" controls
> were simply the five most active features (active on 2.0% to 2.2% of rows,
> against 0.0% to 1.0% for the "selected" set; feature 0 never fires in the
> benign rows it was ranked on). The
> published TWOS intervention (`twos_work/v3_causal_s42`) read the re-ranked
> frontier instead (`top5 = [6036, 7375, 3218, 417, 22]`). So the claim below
> that "every parameter except `--token-class-schema`" matched it is false: the
> frontier, the batch sizes and `--exclude-same-user-donors` all differ.
>
> Consequences: the per-class numbers are arithmetic on an arbitrary pair of
> feature sets. They say nothing about attack-associated features, and the
> seed-42/seed-43 difference is a measured difference between two runs with
> invalid selection, not evidence that correctly selected features are
> unstable across seeds. The `delta`, `repair` and summary columns also use
> the cached base (8,162 repair signs disagree with `delta_recomputed`).
> Root cause, fix and the corrected held-out rerun:
> `docs/REVIEW_2026-09-24_ISSUE_LEDGER.md`, `docs/TWOS_CORRECTED_INTERVENTION_PLAN.md`.


> ## ⚠ THE DIRECTIONAL CONCLUSION BELOW DOES NOT REPLICATE. READ THIS FIRST.
>
> An independent replication on **seed 43** — a different adapter, a different
> delta cache and a different SAE, same configuration and same data — **reverses
> every sign**. The seed-42 conclusion that repairs act on identity tokens and
> not on behaviour is **NOT supported**. Details in "Replication failure" below.
> Results for seed 43 are in `results/twos_token_class_causal_s43/`.
>
> What survives: the per-class decomposition machinery, whose assertions passed
> on both runs, and the finding that the pooled score hides which token class an
> intervention acts on. What does not survive: any claim about *which* class.

Work package 4, run on the **TWOS** replication. 2026-09-19 on Aquaman
(1× RTX 3070), **zero cluster SU**, while Anvil was down.

## Why this run exists

The published causal result says patching the top delta-SAE features lowers the
anomaly score more than patching an activity-matched control set. That score is
a mean over **all** scored tokens, so the result cannot say whether the repair
lands on the profile tokens, the behaviour tokens, or both. That distinction is
the difference between "these features encode the identity shortcut" and "these
features encode behaviour," and the pooled number cannot separate them.

`eval_token_delta_sae_causal.py --token-class-schema cert` now accumulates the
same token losses into per-class sums and counts, so every intervention is
scored against the full, profile-only, behaviour-only, psychometric-only and
day-field-only views at once.

**This is the TWOS dataset, not CERT.** CERT's token-delta cache and SAE
frontier live only on Anvil. CERT remains the headline case and is still
pending.

## Provenance

Every parameter except `--token-class-schema` matches the run that produced
`twos_work/v3_causal_s42`, whose settings are recorded in that directory's
`token_delta_sae_causal_summary.json` and were read from it rather than
remembered: layer 24, latent_mult 4, k 8, `top5` against
`control5_active`, alphas 0.25/0.5/0.75/1.0, context mode `team`,
`patch_chunk_size` 2. The new columns therefore describe the **same
interventions** as the published TWOS causal result.

35,328 candidate rows over 2,275 examples, 138 positive receivers, **16 receiver
users**.

## The base score is recomputed, and that was not optional

Deltas are taken against a base rescored through the identical code path and
batch composition as the patched score, not against the cached `adapted_nll`.
The two differ by a mean of **1.045e-02** nats (max 3.785e-02) — **larger than
every effect measured below**. Using the cache would have buried the result in
a batching artifact. The run also asserts, per batch, that the per-class sums
reconstruct the scalar NLL (max 2.4e-07) and that patched per-class target
counts equal the base's, since a patch changes hidden states and never
tokenization.

## Result — top5 minus activity-matched control, clustered by receiver user

**Which comparison is primary.** The pre-declared test is `profile_only`
against `behavior_only`. The day-field and psychometric views are
**decompositions of the profile view** and are secondary: they may explain a
primary result, they do not stand in for one. This table leads with the
day-field row because it is the largest effect, and that ordering was chosen
**after** seeing the numbers — the pre-declared comparison is stated here so
that emphasis cannot be mistaken for the test. The pre-declared comparison also
holds on its own terms: −0.0104 [−0.0204, −0.0014] for profile against
−0.00008 [−0.00049, +0.00031] for behaviour at α = 1.0.

Negative means the selected features lower that view's loss **more than the
control set does**. The control is the comparison; a raw delta against zero
would only say that some patch moved the score.

| view | α = 0.25 | α = 0.5 | α = 0.75 | α = 1.0 |
|---|---|---|---|---|
| **day fields only** | −0.0093 [−0.016, −0.003] | −0.0167 [−0.028, −0.006] | −0.0240 [−0.039, −0.010] | **−0.0272** [−0.045, −0.010] |
| **profile only** | −0.0031 [−0.009, +0.002] | −0.0066 [−0.013, −0.000] | −0.0097 [−0.018, −0.002] | **−0.0104** [−0.020, −0.001] |
| full | −0.0014 [−0.004, +0.001] | −0.0028 [−0.005, −0.000] | −0.0041 [−0.007, −0.001] | −0.0044 [−0.009, −0.001] |
| psychometric only | +0.0006 [−0.007, +0.009] | −0.0006 [−0.010, +0.008] | −0.0013 [−0.014, +0.009] | −0.0006 [−0.016, +0.012] |
| **behaviour only** | −0.00012 [−0.0005, +0.0002] | −0.00013 [−0.0005, +0.0002] | −0.00010 [−0.0005, +0.0003] | −0.00008 [−0.0005, +0.0003] |
| behaviour, SES lines only | −0.00013 | −0.00015 | −0.00012 | −0.00009 |

## Multiplicity

The table above is **24 intervals** (6 views × 4 alphas) with **no multiplicity
correction**. The day-field and profile contrasts are large, monotone in patch
strength, and consistent across alphas, which multiplicity does not produce. The
behaviour conclusion rests on an interval that *spans* zero, which multiplicity
can only make more likely, not less — so it is conservative in the direction
claimed.

## Replication failure — seed 43 reverses every sign

Both runs use layer 24, latent_mult 4, k 8, `top5` against `control5_active`,
the same four alphas, the same context mode, the same 2,275 examples and the
same 16 receiver users. They differ only in the seed of the whole pipeline: a
separately trained adapter, its own delta cache, its own SAE. `top5` is
therefore a different set of five features in each.

top5 minus activity-matched control, pooled over alphas, clustered by receiver
user:

| view | seed 42 | seed 43 | agree? |
|---|---|---|---|
| day fields only | **−0.0193** [−0.0320, −0.0075] | **+0.0088** [−0.0082, +0.0290] | **no — sign flips** |
| profile only | **−0.0074** [−0.0149, −0.0005] | **+0.0109** [+0.0024, +0.0203] | **no — flips, both exclude zero** |
| psychometric only | −0.0005 [−0.0114, +0.0097] | **+0.0120** [+0.0014, +0.0252] | **no** |
| full | **−0.0032** [−0.0062, −0.0003] | **+0.0030** [+0.0003, +0.0065] | **no — flips, both exclude zero** |
| behaviour only | −0.00011 [−0.00047, +0.00023] | **−0.0024** [−0.0053, −0.0003] | **no — null becomes significant** |
| behaviour, SES only | −0.00012 | **−0.0028** [−0.0061, −0.0004] | **no** |

Seed 43 has a monotone dose-response too, in the opposite direction:
profile-only runs +0.0087 → +0.0104 → +0.0111 → +0.0133 across α, and
behaviour-only sits at −0.0024 → −0.0023 → −0.0024 → −0.0026 with every
interval excluding zero.

**So on seed 43 the selected features improve BEHAVIOUR tokens and make PROFILE
tokens worse — the exact opposite of seed 42.** A dose-response was the main
evidence offered for seed 42 being a genuine causal handle. Seed 43 shows a
dose-response is reproducible in either direction and therefore cannot
distinguish them.

### What this costs the earlier reading

Struck out, not softened:

* ~~"The repair lands on identity tokens, not behaviour tokens."~~ Not supported.
* ~~"There is a clean dose-response, and only on the identity views."~~ Seed 43
  has one on the behaviour views.
* ~~"The pooled result is real but understates where it comes from."~~ The pooled
  result itself flips sign between seeds.

### The honest reading

Which token class a `top5` feature set acts on **depends on which features the
pipeline happened to find**, and that varies with the seed. With 16 receiver
users, 24 uncorrected intervals per run and one seed, the seed-42 result was
within the range that this design produces by chance — which is exactly what
the contamination audit predicted before the replication was run.

Two runs is not enough to characterise the distribution either. The correct
next step is more seeds, not a choice between these two.

## A second, independent error in the seed-42 reading: the difference was not what I said it was

Found by external review and confirmed by recomputation from the 35,328 saved
rows. Reporting only `top5 − control` hid what each patch actually did. At
α = 1.0, per receiver user, **positive means the patch made prediction worse**:

| token group | selected-feature patch | control patch | difference |
|---|---|---|---|
| organizational / DAY | **−0.00143** | **+0.02574** | −0.02717 |
| all profile tokens | −0.01113 | −0.00079 | −0.01035 |
| psychometric | −0.01709 | −0.01652 | −0.00057 |
| behaviour tokens | **+0.00338** | **+0.00346** | −0.00008 |
| full | −0.00246 | +0.00189 | −0.00435 |

Two things follow, and both contradict how I wrote this up:

**1. The large day-field number is mostly the control degrading, not the
selected features helping.** The selected features improve day-field loss by
0.00143. The control *worsens* it by 0.02574. About 95 % of the −0.02717
"advantage" is the control arm, so describing it as a repair produced by the
selected features was wrong.

**2. Both patches worsen behaviour-token prediction, by almost the same
amount.** A near-zero difference means the two patches are *similar* on
behaviour, not that neither touches it. Writing that the intervention "does not
affect behaviour" inverted the meaning of a small contrast.

### What the seed-42 run actually supports

> On this run, the selected-versus-control difference is concentrated in
> profile-token loss, with little detectable difference between the two patches
> on behaviour-token loss.

That is real mechanistic evidence and it is worth reporting. It does **not**
establish an exclusively identity-related circuit, and the earlier claim that no
confound could produce the dose-response was unjustified — a confound acting on
the control arm would produce exactly this pattern.

Together with the seed-43 reversal above, the position is: the contrast is not
stable across seeds, and on the seed where it was largest it was driven mainly
by the control.

## What the seed-42 run showed (NOT replicated, and the reading above supersedes this — retained for the record)

**1. The repair lands on identity tokens, not behaviour tokens.** At full patch
strength the day-field view improves by −0.0272 while the behaviour view moves
by −0.00008 — a factor of roughly **340**, with the behaviour interval tightly
bracketing zero at every strength. The same holds for the SES lines alone.

**2. ~~There is a clean dose-response, and only on the identity views.~~**
~~A confound that merely perturbed the model would not produce a dose-response on
one token class and a flat line on another.~~ **Both halves of this are
withdrawn.** Seed 43 produces a dose-response on the behaviour views, and a
confound acting on the control arm produces exactly this pattern — which the
three-column table above shows is what happened.

**3. The pooled result is real but understates where it comes from.** The full
view's −0.0044 is the number the published pooled metric would report. It is
almost entirely the profile component; the behaviour component contributes
essentially nothing.

**4. On TWOS the effect is in the organizational fields, not the psychometric
ones.** The psychometric contrast spans zero at every alpha. TWOS `DAY` lines
carry `team`, `leader` and `machine`, and `machine=` is a near-unique per-user
string — a strong identity token. CERT's profile splits differently, so this
particular split should not be assumed to transfer.

## How this fits the other packages

* Package 2: a non-language detector given the same fields puts **−3.7 % to
  +10.2 %** of its score gap into the profile channels. The shortcut is not
  forced by the field set.
* Package 3: the profile functions as an **identity key** — the user's own
  profile collapses psychometric loss by 3.2 nats but buys ≤ 0.016 nats of
  behavioural predictive value, while a wrong profile hurts behaviour.
* Package 4 (here): the SAE features that causally move the score act on the
  **identity tokens**, with a dose-response, and not on behaviour.

## Limits

* **16 receiver users.** Intervals bootstrap over 16 clusters. They exclude
  zero for the day-field and profile contrasts at α ≥ 0.5, but this is a small
  number of clusters and the result should be read as strong direction with
  modest precision.
* **TWOS, not CERT.** CERT is the headline dataset and needs Anvil.
* **Absolute effects are small**, at most 0.027 nats. The claim is about *where*
  the repair lands, not about its size as a detection intervention.
* **One seed (42) and one layer (24)**, matching the published TWOS run.
* Two artifact problems were found and fixed rather than worked around: the
  delta cache was built from `audit_pool_v3` (2,275 rows) rather than
  `session_jsonl_v3` (10,354), and the adapter's tokenizer config stored
  `extra_special_tokens` as a list that current transformers cannot parse. The
  tokenizer was corrected in a symlinked overlay leaving the original untouched,
  and verified to produce **token ids identical to the base tokenizer**.

## Files

`token_delta_sae_causal_candidate_rows.csv` carries, per intervention,
`base_/patched_/delta_` for all six views, per-class target counts, and
`base_cache_minus_recomputed`. Reproduce the tables with
`scripts/analyze_token_class_causal.py results/twos_token_class_causal
[--by-alpha]`.
