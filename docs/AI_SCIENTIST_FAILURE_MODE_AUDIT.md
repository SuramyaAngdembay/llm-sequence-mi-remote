# Self-audit against Luo, Kasirzadeh & Shah, "The More You Automate, the Less You See"

arXiv:2509.08713. Audited 2026-09-19 against this repository's artifacts, not
from memory. Every claim below names the file or commit that supports it.

The paper's central empirical result is that reviewing the **paper alone**
detects these failures at 51.4 % accuracy, while paper **plus execution logs and
code** reaches 74.0 %. This audit is therefore written against the logs and
code, and it is the right document to hand a reviewer alongside the manuscript.

**The most important finding of this audit, stated first:** the errors that
mattered most in this project were caught by **external review** (2026-09-19),
not by self-audit. An earlier version of this file called that an "external
human audit". I had no evidence of who or what wrote it, and a later review
indicated it came from a Codex review rather than a person. The claim is
corrected, and it is an instance of exactly the failure this document is about:
asserting a fact about provenance from assumption. That audit found a population mismatch that had
produced a wrong headline claim, a forecasting leak, an average-precision figure
paired with the wrong prevalence, a miscounted fold tally, and a mechanism
explanation that contradicted our own earlier finding. A self-written trace is
necessary but demonstrably not sufficient, which is exactly the paper's thesis.

---

## 1. Inappropriate benchmark selection

*Paper's definition: systems cherry-pick datasets yielding strong performance
rather than choosing on relevance, difficulty or representativeness. Detected by
tracking which of 20 difficulty-ordered datasets a system selects.*

**Exposure found — benchmark chosen by availability, and it produced the clean
result.** Work package 4 was specified for CERT r4.2. It ran on **TWOS**,
because the CERT token-delta cache and SAE frontier live only on Anvil, which
was down. TWOS then produced a clean positive result with a dose-response. The
causal story is an outage, not preference, but the *shape* is the failure mode:
the dataset that was reachable is the dataset that got reported.

Mitigations in place: `results/twos_token_class_causal/README.md` states in its
second paragraph that this is not CERT and that CERT remains the headline case;
the CERT submitter exists and is unrun; this document records the substitution.
Mitigation still owed: the CERT run itself.

**Exposure found — configuration chosen because a prior run existed there.**
Within TWOS, seed 42 was used because `twos_work/v3_causal_s42` already existed
to match parameters against. A seed-43 replication was launched for exactly this
reason and is running.

**Clean, and checkable.** CERT r4.2 and r6.2 are the project's premise, fixed
long before any of this work. The order of running was r6.2 **first**, r4.2 as
replication — the reverse of what result-shopping would produce. Work package 2
reports all three numerical detectors including `fc7`, which has the **worst**
user ROC of the three (0.7389 against 0.7894 and 0.7851); none was dropped.

---

## 2. Data leakage

*Paper's definition: the system accesses test-set information during training or
validation. Detected by injecting label noise and checking whether reported
accuracy exceeds the theoretical ceiling. The paper's reviewer advice: "examine
logs for unexpected dataset creation or subsetting."*

**Five defects were found, and calling them all "leaks" was wrong.** External
review pointed out that these are distinct failure types and that grouping them
inflates the leakage claim. Only the first is leakage in the train/test sense.
Reclassified:

| defect | type | status |
|---|---|
| Forecast first day had itself as its own context (4,000 rows) | **leakage** — target visible in its own input | found, fixed, now asserted as an exact set equality (V26) |
| Raw integer `user` column would have carried identity into the time-series model | **feature validity** — a prevented identity shortcut, not a leak that occurred | excluded before training |
| `val.jsonl` is a strict subset of `eval.jsonl` — 16,992 rows double-counted | **population definition** — double counting, not leakage | found, dedup added |
| Split keyed on post-merge positives (4) rather than the answer key (5) | **split construction** | found, fixed |
| TWOS delta cache built from `audit_pool_v3` (2,275 rows), not `session_jsonl_v3` (10,354) | **artifact mismatch** | caught **by an assertion in the code**, 2026-09-19 |

The last one is the paper's own advice working: an unexpected subset was caught
because the code refuses to proceed when row counts disagree.

**Exposure found — label-dependent sample construction.** The history-prefix
sampler's `--user-limit` retains every user with a positive day. That uses
labels to build the sample. It is harmless here because **no detection metric is
computed on that sample** — the probe reports loss differences only — but it
would be a leak the moment anyone computed an AUC on it. Stated in the code's
own help text; now stated here as a standing constraint.

**Exposure found and UNRESOLVED — adapter provenance is assumed, not verified.**
Work package 3's results come from adapters on Aquaman
(`cert-data/{r42,r62}_adapter`). Their md5s are recorded (V34). Whether they are
byte-identical to the Anvil adapters that produced the published numbers is
**unverified**, because Anvil is unreachable. The `base_model_name_or_path`,
LoRA rank, alpha and target modules all match, which is consistent but not
proof. This must be checked when Anvil returns before package 3's numbers are
put in the paper.

---

## 3. Metric misuse

*Paper's definition: selectively reporting favourable metrics rather than those
reflecting the research objective. Their case: one system substituted the
specified metric with F1 and training loss.*

**The metric SET was recorded before any result existed (commit `007ca54`);
that is narrower than "metric use is clean", and a commit proves only that the
definitions existed then, not that anyone was bound by them.** External review is right that fixed definitions do not guarantee
correct baselines, populations, aggregation or interpretation — and the TWOS
baseline defect below proves the point. What is checkable is only this: the
token-class views (`full`,
`profile_only`, `behavior_only`, `behavior_ses_only`, `psy_only`, `day_only`)
were frozen in `scripts/token_class_decomposition.py` on **2026-09-17, commit
007ca54**, and `git log -S CERT_VIEWS` shows **no subsequent edit**. Every view
is reported in both results READMEs. The metric set was fixed before the results
existed, and that is verifiable from the history rather than asserted.

**Clean — the wrong metric is not merely avoided, it is tested against.** The
naive `s_full − s_profile` was prohibited in the task specification.
`scripts/tests/test_eval_metrics_core.py` contains a check asserting that this
quantity *reorders examples* relative to the true conditional behaviour score,
and that `s_full − w_P·s_P` does not. The prohibition is enforced by a test, not
by discipline.

**Clean — adverse and null results are reported.** Attack localization is 0 of 4
in every model and both main views. The behaviour contrast in package 4 is flat
and spans zero at every patch strength. The psychometric contrast on TWOS is
null. The day-field view initially moved the wrong way. A previous "ranks users
near chance" claim is marked **withdrawn** in
`results/timeseries_probe/README.md` rather than quietly deleted.

**Exposure found — emphasis is post-hoc even though the metric set is not.** The
TWOS write-up leads with the day-field contrast (−0.0272), the largest of the
six. The **pre-declared** comparison is `profile_only` against `behavior_only`;
`day_only` is a decomposition of the profile view, offered as explanation. The
pre-declared comparison also holds (−0.0104 against −0.00008), so the conclusion
does not depend on the emphasis — but the emphasis was chosen after seeing which
number was biggest. Commit `9fb1152` is the write-up in question; the fix landed
in `c0131c4`.

**Previously corrected, listed for completeness.** Average precision paired with
the wrong prevalence (fold-average 0.0164 belongs to fold prevalence 0.00080,
not the pooled 3.23 %); a fold tally reported as 54 improved when it was 51
improved, 3 tied, 6 worse; recall-at-FPR presented as if it were a deployment
threshold when it is an empirical operating point calibrated on the same
negatives it is evaluated against. All three were external-audit findings.

---

## 4. Post-hoc selection bias

*Paper's definition: evaluate multiple candidates on test data, then report only
the strongest — p-hacking. Detected by inverting test rankings and seeing
whether the system's selection follows.*

**Exposure found, and a false claim of mine corrected.** The history-prefix
probe first ran **without** the `week` field in the prefix. The day-field view
moved the wrong way. I then wrote the `--prefix-include-week` variant and ran
it. That is a protocol change made after seeing an unfavourable result. Worse,
the results README claimed "neither was selected after seeing the outcome",
which was **false**. Corrected in `results/history_prefix/README.md` this
session, which now states the order of events explicitly.

What keeps it from being result-shopping, and what a reviewer should check: both
variants are reported in the four-row replication table; the mismatch it fixes
can be confirmed from the serialization alone, though it was noticed only because
the pilot's loss moved the wrong way (commit `93f2d0a` records this); it made
a falsifiable prediction (restoring the format should send the day contrast to
zero) that was borne out (+0.42 → +0.021, interval now spanning zero); and the
headline psychometric conclusion is −3.31 versus −3.23 across the two variants,
so it does not depend on the change.

**Exposure found — the primary table is also the largest effect.** The package-3
primary table is r6.2-with-week, whose profile effect (−1.304) is the largest of
the four runs (−1.099, −1.166, −0.847). The justification is methodological: the
with-week prefix restores the training format. That justification is real, but
it was formed after the runs, and the coincidence is now stated in the README
with a pointer to the full replication table.

**Clean.** All four patch strengths are reported with their dose-response; none
was dropped. Both feature sets are reported. The bootstrap seed (42) and draw
count (2,000) were fixed when the evaluator was written on 2026-09-18 and never
changed. The r4.2-with-week run excluded 908 examples for want of a
length-matched donor, and that is disclosed in the README rather than absorbed.

**Structural weakness — there is no pre-registration.** The progress record
(V1–V35) is a chronological trace, which is what the paper says enables
detection. But it is written by the same agent doing the work, so it can be
self-serving in emphasis even when every individual entry is true. The remedy is
to pre-register the analysis *before* the data exists. Done for the one
experiment still ahead: `docs/PREREGISTRATION_CERT_PACKAGE4.md`, committed while
Anvil is unreachable and the run is therefore impossible.

---

## Where this workflow is genuinely stronger than the paper's subjects

* **Gates refuse to proceed rather than warn.** Cached scores must reproduce
  before they are used; a scoring gate refused a full run over a hardware
  mismatch; a loss-path gate aborted a training job over a 4.0× normalization
  error; the checkpoint gate refused a run whose window did not match its
  checkpoint. These are failures that were prevented, not detected afterwards.
* **Correctness checks are cross-implementation.** `token_class_nll` is checked
  against an independent Python implementation of the same quantity, not only
  against itself.
* **Negative results are load-bearing in the write-up**, not appended.

## Where it is weaker

* **No pre-registration** until today, for a project this far along.
* **The trace is self-authored.** Every serious correction this project has made
  came from external review.
* **Effect emphasis drifts toward the largest number** even when the metric set
  is fixed, as in the day-field lead.
* **Provenance of a key artifact is assumed** — the Aquaman adapters.

## Actions taken as a result of this audit

1. Corrected the false pre-registration claim in
   `results/history_prefix/README.md`.
2. Wrote `docs/PREREGISTRATION_CERT_PACKAGE4.md` before the run is possible.
3. Stated the pre-declared contrast in the TWOS README so the day-field lead
   cannot be mistaken for the pre-declared test.
4. Launched the TWOS seed-43 replication.
5. Recorded adapter-provenance verification as a blocking item before package 3
   enters the paper.

---

## Damage-radius trace (2026-09-19)

Applying step 2 of the `research-claim-triage` skill to this project's own
corrections: a withdrawn number does not stay where it was withdrawn. Every
corrected figure was traced through the repository, the manuscript sources and
the shared notes repo.

| withdrawn / corrected figure | found outside its correction notice? |
|---|---|
| Time-series user AUC 0.477 / 0.508 / 0.296 | **No.** The only matches are coincidental digit strings inside per-row CSV dumps in unrelated causal reports. |
| "ranks users near chance" (time-series, user level) | **No.** `paper/main.tex:978` says "near chance" about a *different* claim — the **masking ablation** at the **day** level on r6.2 (ROC 0.547 / 0.575 / 0.471), already hedged as descriptive with four clusters. Correctly left alone. |
| Fold tally "54 improved" (true: 51 improved, 3 tied, 6 worse) | **No.** Only the audit's own listing of the error. |
| Average precision paired with the wrong prevalence | **No.** |

**The honest reason the radius is clean: the new work has not reached the
manuscript yet.** Searching `paper/main.tex` and `paper/main2.tex` for the
probe's own figures (0.7894, 0.6739, 0.7061, 3.226, 1.304) returns nothing, and
neither manuscript mentions the history-prefix probe or the token-class
decomposition at all. This is not evidence of vigilance; it is evidence that the
corrections happened before the writing.

**Forward-looking consequence.** The next manuscript revision is precisely where
these numbers will be transcribed for the first time. The corrected versions —
not any figure quoted in an earlier chat, email, or handoff note — are the ones
in `results/*/README.md`. The near-miss to avoid is quoting a number from
conversation rather than from the artifact.

One live trap was found and left deliberately untouched: `paper/main.tex:978`
matches the withdrawal's *wording* but not its *claim*. Patching it would have
introduced an error while "fixing" one. That is the case the skill's
classification step exists to catch.

---

## External review of this audit (2026-09-19), and what it corrected

This document was itself reviewed. Every empirical claim in that review was
recomputed here before being accepted; all were correct.

| review finding | verified? | action |
|---|---|---|
| The TWOS day-field "advantage" is mostly the **control degrading**: selected −0.00143, control +0.02574 | **confirmed to 5 dp** | interpretation rewritten; the analysis tool now prints both arms by default |
| **Both** patches worsen behaviour (+0.00338 / +0.00346); a near-zero difference means *similar*, not *absent* | **confirmed** | "does not affect behaviour" withdrawn |
| The repair flags still used the cached baseline — **8,162 of 35,328 rows** disagree on sign | **confirmed exactly** | `delta` now uses the matched-batch base; cached value kept as `base_score_cached` |
| The claim linter accepts citation-**shaped** text: "See nonexistent_evidence.json" passed | **confirmed** | citations are now resolved against the filesystem and git; two self-tests added for this hole; findings now exit non-zero |
| The sampling guard is optional and all four history-prefix outputs lack the flag | **confirmed** | flag backfilled into all four metas and CSVs, marked as backfilled |
| "Five data leaks" conflates distinct defect types | agreed | reclassified: one leakage, one prevented feature shortcut, one population-definition error, one split error, one artifact mismatch |
| "Label-dependent sampling invalidates AUC" is an incorrect blanket rule, in both directions | agreed | guard keeps a fail-safe default but now takes a written `justification`; docstring states both directions |
| "Metric use is clean because views were fixed" is insufficient | agreed | narrowed to "the metric *set* was recorded in advance" (commit `007ca54`), which is all that was checked |
| The 2026-09-19 review was called an "external **human** audit" without evidence | **confirmed as unsupported** | corrected to "external review"; this was itself a provenance claim asserted from assumption |
| Fingerprints prepare a verification; the comparison is still owed | agreed | unchanged — still blocked on Anvil |

Two of these — the contrast decomposition and the stale baseline — are defects
this audit **missed while auditing for exactly that class of defect**. The audit
recomputed the contrast and confirmed it to five decimals without noticing that
the number it confirmed did not mean what the write-up said. Arithmetic
verification and interpretive verification are different checks, and passing the
first is not evidence for the second.

The review's overall verdict is recorded as stated: the corrections are specific
and do not justify discarding the study.
