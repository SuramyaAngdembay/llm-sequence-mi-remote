# Does the adapted model copy the profile from its context?

Work package 3. Run 2026-09-19 on Aquaman (2× RTX 3070), **zero cluster SU**,
while Anvil was down.

## The question and the design

The delta-SAE audit says the adapter carries identity information, and the
token-class decomposition says the profile tokens carry a large share of the
anomaly score. Neither says *how* the model gets the profile tokens cheap.
Two accounts fit: the adapter **memorized** each user's profile, or the model
**copies** an identity block that is present in its context.

Three conditions on one frozen adapter, scoring identical targets:

| | prefix |
|---|---|
| **A** | none — the published scoring condition |
| **B** | the same user's static profile, from a strictly earlier record |
| **C** | a **length-matched different user's** profile, same format |

Two design points decide whether any of this means anything, and both are
enforced in code and checked offline in `scripts/tests/test_history_prefix*.py`
(32 + 10 checks):

* **The current day's token ids are identical across conditions.** The prefix
  and the current text are tokenized separately and their ids concatenated.
  Re-tokenizing a joined string could merge tokens across the boundary.
* **The scored target set is identical across conditions.** The current day's
  *first* token is excluded as a target in **all three**, so adding a prefix
  cannot add a target that A never had. The runner asserts per batch that the
  three conditions produce identical per-class target counts.

Declared in `scripts/history_prefix.py` (`choose_length_matched_donor`, line 108)
and committed in `007ca54`/`8a26446` before any loss existed: donors come from
each donor's earliest record in a seeded hash order that depends only on the recipient id; the first
donor whose prefix tokenizes to **exactly** the recipient's prefix length is
taken; an example with no match within 200 candidates is **excluded and
counted**, never approximated.

## What condition B actually tests, measured not assumed

The static-profile overlap between a user's earlier record and their current
one is **1.0000** in every run: CERT profiles do not change. So B is a test of
**copying an identity block already in context**, not of recalling a different
past. Donor profiles overlap the recipient's on 0.27–0.31 of fields.

## Everything that was run, including what is not a result

Disclosure in the sense of Simmons, Nelson & Simonsohn (2011): all runs on this
probe, not only the reported ones.

| run | scale | reported? |
|---|---|---|
| smoke | 8 examples, r6.2 val split | no — mechanics check |
| **pilot** | **40 users × 5 days, r6.2 eval, no-week prefix** | **no — but it informed two decisions below** |
| r62_noweek / r62_week / r42_noweek / r42_week | 200 users × 15 days | yes, all four |

The pilot matters and was previously undisclosed. It ran on the **same
evaluation data** as the reported runs, and it is what prompted (a) scaling to
200 users and (b) writing the week-prefix variant after its day-field view moved
the wrong way. The reported 200-user runs are therefore **not independent of**
the pilot: this is a garden-of-forking-paths dependency, not a fresh test. The
pilot's own numbers agreed with the final ones in direction and magnitude
(profile B−A −1.06 against −1.10 final), which is reassuring but is not
independence.

## Multiplicity

Each run's table reports 6 views × 3 contrasts = **18 intervals**; four runs give
**72**, with **no multiplicity correction applied**. Read accordingly:

* The psychometric and profile effects are 50–100× their interval widths. No
  plausible correction touches them.
* `behavior_ses_only` B−A, at −0.002 [−0.002, −0.001], is precisely the size of
  effect that 72 uncorrected intervals manufacture. **It should not be cited as
  a finding.** The behaviour-view conclusion rests on the effect being
  *negligible*, which does not depend on its sign.

## Result — clustered by user, which is the unit of evidence

Days from one user share that user's profile, so the bootstrap resamples
**users**, not examples. Sample: 200 users capped at 15 eligible days each,
users with any positive day always retained.

### r6.2, training-format prefix (2,982 examples, 199 users)

| view | B − A (own profile) | C − A (stranger) | B − C |
|---|---|---|---|
| psychometric only | **−3.226** [−3.352, −3.102] | −0.413 [−0.548, −0.274] | −2.813 [−2.910, −2.717] |
| profile only | **−1.304** [−1.362, −1.248] | +1.122 [+1.044, +1.199] | −2.426 [−2.489, −2.365] |
| day fields only | +0.021 [−0.016, +0.055] | +2.179 [+2.096, +2.259] | −2.158 [−2.236, −2.081] |
| behaviour only | **+0.016** [+0.014, +0.018] | +0.133 [+0.125, +0.141] | −0.117 [−0.124, −0.109] |
| behaviour, SES lines only | −0.002 [−0.002, −0.001] | +0.124 [+0.116, +0.132] | −0.126 [−0.134, −0.118] |

Absolute means for the psychometric view: **3.373 → 0.148** with the user's own
profile in context, versus 2.961 with a stranger's.

### Replication

| release / format | profile B − A | psychometric B − A | behaviour B − A |
|---|---|---|---|
| r6.2, no week | −1.099 [−1.157, −1.042] | −3.306 [−3.428, −3.183] | +0.006 [+0.005, +0.008] |
| r6.2, with week | −1.304 [−1.362, −1.248] | −3.226 [−3.352, −3.102] | +0.016 [+0.014, +0.018] |
| r4.2, no week | −0.847 [−0.912, −0.783] | −3.140 [−3.287, −2.994] | +0.011 [+0.009, +0.014] |
| r4.2, with week | −1.166 [−1.234, −1.098] | −3.058 [−3.208, −2.908] | +0.005 [+0.003, +0.007] |

## What this establishes

**1. The model copies profile tokens from context, and the copying is
identity-specific.** Psychometric loss falls by about 3.2 nats — from 3.37 to
0.15 — when the user's own profile is in context, in **100 % of users** in
every run. A length-matched stranger's profile does not produce that drop. The
profile-view gap between the two conditions is −2.43 [−2.49, −2.37].

**2. The correct profile does not help predict behaviour.** The behaviour view
moves by **+0.016 nats or less** in every run, and for the SES lines alone the
effect is −0.002, i.e. indistinguishable from nothing. Against −3.2 nats on the
psychometric view, that is a factor of roughly 200. Having the user's identity
block in context buys essentially zero behavioural predictive value.

**3. A wrong profile does hurt behaviour.** The stranger condition raises
behaviour loss by +0.12 to +0.21 nats, interval excluding zero, in 99 %+ of
users. So the model is conditioning behaviour prediction on the profile as an
identity key: supplying the wrong key disrupts it, while supplying the right
key adds nothing the current day's own profile line did not already provide.

**4. Organizational fields and psychometric fields dissociate sharply.** With
no prefix, day-field loss is already low (0.44) while psychometric loss is high
(3.37). The organizational attributes are cheap from the model's priors; the
Big Five integers are arbitrary per user and are only cheap when copyable. Any
account of "the model memorized the profile" has to explain why it applies to
one half of the profile and not the other.

## A confound found, and turned into a control — with the order of events stated

In the first run the day-field view moved the **wrong** way under the own-
profile condition: +0.42 [+0.38, +0.46]. That is a format surprise, not failed
copying. The prefix omitted `week`, so its DAY line differed in format from
every DAY line seen in training. Re-running with the earlier record's own week
value restored moves that contrast to **+0.021 [−0.016, +0.055]** — an interval
that now includes zero.

**The order matters and is stated plainly, because this is a post-hoc protocol
change.** The no-week variant ran first. The week variant was written *after*
seeing the no-week day-view anomaly. It was not pre-registered: commit `93f2d0a`
is where it was added, and that commit's message records the pilot number that
prompted it. An earlier
version of this file claimed "neither was selected after seeing the outcome",
which was **false**, and is corrected here.

The ordering is checkable rather than asserted: commit `93f2d0a` is the commit
that added `--prefix-include-week`, and its own message records the pilot result
(+0.42 nats on the day view) that prompted it. The results commit `e755e12`
follows it. `git log --format='%h %ad %s' -- scripts/history_prefix.py` shows the
sequence.

What keeps this from being result-shopping, and what the reader should check:

* Both variants are reported in full, in the replication table above, and
  neither is suppressed.
* The change was driven by a *diagnosed mechanism* — a format mismatch that is
  visible in the serialization without looking at any loss — and it makes a
  falsifiable prediction that was then borne out: restoring the format should
  move the day-field contrast to zero, and it did.
* The headline conclusion does not depend on it. The PSY line is built outside
  the `include_week` branch (`scripts/history_prefix.py` line 86), so it is
  byte-identical in both formats, and the psychometric effect is −3.31 (no week) versus −3.23
  (week).
* The primary table shows **r6.2 with week**, which is also the **largest** of
  the four profile effects (−1.304 against −1.099, −1.166, −0.847). It is
  presented as primary because restoring the training format is the
  methodologically correct choice, not because it is largest — but a reader
  should weigh that coincidence and use the four-row replication table.

## Limits

* **Condition B is a copying test, not a memory test.** CERT profiles are
  static (overlap 1.0000), so B re-presents information that is also present in
  the current day's own DAY/PSY line. This says the copy mechanism exists and is
  identity-specific. It does **not** establish that the *deployed* profile-token
  loss in condition A is produced by copying, because in deployment no earlier
  profile is in context.
* **The r4.2 with-week run excluded 908 examples** for want of a length-matched
  donor, against 15 without week, because including `week` widens the spread of
  prefix lengths. That variant is therefore on a different and smaller eligible
  set; it is reported as a replication of direction and magnitude, not as a
  paired comparison with its own no-week run.
* **Only 4 users with a positive day** are in the r6.2 sample. Nothing here is
  a detection claim, and no metric in this file is a detection metric.
* Loss differences are in nats per token, not detection performance.

## Files

Per tag (`r62_week`, `r62_noweek`, `r42_week`, `r42_noweek`):
`history_prefix_per_example.csv`, `history_prefix_summary.csv` (example-level),
`user_clustered_summary.csv` (user-level with bootstrap intervals, the
authoritative table), `history_prefix_meta.json` (sampling, exclusions,
overlaps, declared rules).

Reproduce: `scripts/run_history_prefix_probe.py` then
`scripts/analyze_history_prefix.py <tag-dir>`.
