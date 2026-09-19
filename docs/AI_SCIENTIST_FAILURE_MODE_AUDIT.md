# Self-audit against Luo, Kasirzadeh & Shah, "The More You Automate, the Less You See"

arXiv:2509.08713. Audited 2026-09-19 against this repository's artifacts, not
from memory. Every claim below names the file or commit that supports it.

The paper's central empirical result is that reviewing the **paper alone**
detects these failures at 51.4 % accuracy, while paper **plus execution logs and
code** reaches 74.0 %. This audit is therefore written against the logs and
code, and it is the right document to hand a reviewer alongside the manuscript.

**The most important finding of this audit, stated first:** the errors that
mattered most in this project were caught by an **external human audit**
(2026-09-19), not by self-audit. That audit found a population mismatch that had
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

**This project's record is good only because leaks were found.** Five, all
documented in `docs/SCORE_DECOMPOSITION_PROGRESS.md`:

| leak | status |
|---|---|
| Forecast first day had itself as its own context (4,000 rows) | found, fixed, now asserted as an exact set equality (V26) |
| Raw integer `user` column would have carried identity into the time-series model | excluded before training |
| `val.jsonl` is a strict subset of `eval.jsonl` — 16,992 rows double-counted | found, dedup added |
| Split keyed on post-merge positives (4) rather than the answer key (5) | found, fixed |
| TWOS delta cache built from `audit_pool_v3` (2,275 rows), not `session_jsonl_v3` (10,354) | caught **by an assertion in the code**, 2026-09-19 |

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

**Clean, and independently checkable.** The token-class views (`full`,
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
number was biggest, and the README should say so.

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
variants are reported in the four-row replication table; the change was driven
by a mechanism visible in the serialization without looking at any loss; it made
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
  came from an external audit.
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
