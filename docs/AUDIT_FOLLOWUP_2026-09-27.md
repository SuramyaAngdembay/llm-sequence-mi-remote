# Audit follow-up, 2026-09-27

Response to `~/Documents/mi-paper-review-2026-09-27/review.md`. Everything
here is **exploratory**. Every receiver population was examined earlier, and
the H6 cohort is development data.

Status: sections 1 and 2 are complete. Section 3 (H6) is **pending**: the job
is queued on Anvil, and its results will be added here and in
`results/h6_2026_09_27/README.md`.

## 1. Corrections to the completed experiments

Each correction was checked against the saved artifacts before it was
written. The corrected reports and the ledger are authoritative. Commit
messages `a260e92`, `a85fe08` and `b955acc` keep superseded wording.

| item | now reads | where |
|---|---|---|
| A, familiarity | No detectable familiarity effect was found under the tested profile substitutions (F4 −0.004 [−0.031, +0.022]; not an equivalence test; the two partners also differ in other attributes). | Pilot 2 README, ledger |
| B, mediation | The tested average session-position activation statistic did not detect a clear selected-versus-control response; mediation remains untested. Signed record means cancel; absolute changes are 2.5–8.2% of the means for 4596, 3673 and 3455, and similar for controls. | Pilot 2 README, ledger |
| C, "shortcut" | Profile dependence, or profile-conditioned prediction, created by fine-tuning. Harm to detection or robustness is not established. | Pilot 2 README, ledger, hub |
| D, H4 | Reported per feature with eligible-user counts. 4596 is associated with lower adapted loss (−0.83, 12/12 eligible users); 3455 goes the other way; the selected-set gain on attack days is inconclusive (+0.20 [−0.71, +1.02]). | Pilot 2 README, ledger |
| E, TWOS Direction A | Inconclusive and deprioritized. The model test used the equality-only projection, not the QP. The projection is smaller. The direct attack-window contrast includes zero. Jaccard 0.78 is one of 8 features replaced. | TWOS README, ledger |
| F, Pilot 4 | The mean-difference condition skipped 44 of 1,396 reference tokens (27 attack, 17 benign). Absolute paired differences replace "ten times". | Pilot 4 README, ledger |

## 2. TWOS applied-edit validation (repaired)

`scripts/twos_edit_validation_gpu.py` now captures the patched hidden state
with a forward pre-hook on the next block, after the patch hook and its bf16
cast. It takes the adapter-off state from an identically batched run,
re-encodes the difference, and compares it with the explicitly recorded
intended code. That code is built from the same cached deltas the scored
shifts used. Details and tables are in the repair section of
`results/twos_feasibility_2026_09_26/README.md`.

- **The implementation passes.** Zero-edit hidden states equal unhooked ones
  exactly. Applied edit norms equal intended norms (median ratio 1.000). There
  were no numerical failures. 27 of 64 receivers are no-ops, and 142 tokens on
  37 receivers are edited.
- **Realization is approximate.** The own-support decoder edit has normalized
  target error median 0.040 (q90 0.103, q99 0.297, max 0.798). Part of the
  intended support is replaced.
- **Common-budget comparison, attack windows, direct paired differences:**

  | comparison | held fixed | decoder minus projection |
  |---|---|---|
  | at the decoder's budget | magnitude, large | +0.0042 [+0.0013, +0.0077] |
  | at the projection's budget | magnitude, small | −0.0010 [−0.0030, +0.0005] |
  | common requested target | target | +0.0021 [−0.0017, +0.0070] |

  On attack windows, neither method's excess over a coherence-preserving
  random edit of matched magnitude is detected; the intervals include zero.
  The evidence comes from eight previously examined users and is
  descriptive only.

## 3. H6: where profile-dependent prediction arises (pending)

Protocol frozen before scoring:
`docs/H6_PROFILE_PATCHING_PROTOCOL_2026-09-27.md` (commit `ab7299f`), with
dated pre-scoring notes. Code changes after the freeze and before any CERT
output are listed in the ledger. They were tested on TWOS as an execution
check only.

Results: *pending.*

## 4. Claims, kept separate

| claim type | what the evidence supports now |
|---|---|
| Detector performance | Not tested in this round. No endpoint here measures detection. |
| Input dependence | Supported: adaptation makes behaviour prediction sensitive to profile replacement (adapted minus base +0.11 to +0.18 nats per token), on attack and benign days alike. No familiarity effect detected. |
| Represented information | Feature-level associations only. 4596's association with lower adapted loss reproduces; 3455 goes the other way. What any feature represents is not identified. |
| Causal influence and mediation | Influence: the tested SAE edits change session predictions more than the tested PCA and mean-difference edits at the same tokens and sizes. Mediation of the profile dependence by the selected features: untested until H6. |

## 5. Reading

**Supported (exploratory).**

- Adaptation increases behaviour prediction's sensitivity to profile
  replacement.
- Feature 4596 has a reproducible association with lower adapted loss.
- Under the existing intervention design, the selected SAE edits influence
  predictions more than the tested PCA and mean-difference alternatives.
- Donor-policy bounds give a useful finite-bank robustness check. CERT
  behaviour conclusions survive any reweighting of the bank. CERT profile
  conclusions flip at 1–2% of donor weight, and TWOS behaviour at 0.24%.

**Inconclusive.**

- The selected-set attack-day adaptation gain (H4).
- Attack-specific differentials for the SAE edit (E6, E8c).
- TWOS Direction A.
- Familiarity, since a null interval is not equivalence.

**Implementation limitations.**

- More than half of the Pilot 1 edit effect comes from union-support writes
  into zero coordinates.
- CERT random controls drew independent directions per token; a coherent
  random control was run only on TWOS.
- The mean-difference baseline was amplified about 13-fold and skipped 44
  tokens.
- TWOS realization is approximate, and the TWOS evidence rests on 8
  previously examined users.

**Hypotheses that need another experiment.**

- Whether the profile dependence harms detection or robustness. That needs a
  detection endpoint under profile substitution.
- Whether the mean-difference differential holds at its own positions with a
  replacement-size edit on a fresh population.
- Where and through which components the dependence arises (H6, pending).

## 6. Recommendation

*To be completed with the H6 result.* Independent of it: stop work on
Direction A, keep the donor-policy LP as a robustness appendix, and do not
expand any pilot on previously examined users.

## 7. Compute for this follow-up

| item | hardware | GPU time | CPU time |
|---|---|---|---|
| TWOS validation repair | 1 RTX 3070 (Aquaman) | 90 s | 264 s user, 20 s system |
| H6 code-path tests on TWOS (two runs) | 1 RTX 3070 (Aquaman) | 69 s + 72 s | not timed separately |
| H6 dry run, re-run under a timer (manifest identical, md5 `fe2bd2cd`) | CPU, Anvil login node | 0 | 12 s user, 2 s system (32 s wall) |
| H6 alignment unit tests | CPU, Anvil login node | 0 | 0.3 s user |
| H6 analysis on the TWOS test output | CPU, laptop | 0 | 0.2 s user |
| H6 CERT job | 1 A100 or H100 (Anvil) | pending, at most 30 min | pending |
| **total so far** | | **0.06 GPU-h of the 1 GPU-h cap** | |
