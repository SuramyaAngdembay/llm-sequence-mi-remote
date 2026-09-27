# Audit follow-up, 2026-09-27

Response to `~/Documents/mi-paper-review-2026-09-27/review.md`. Everything
here is **exploratory**. Every receiver population was examined earlier, and
the H6 cohort is development data.

Status: complete. The H6 detail is in `results/h6_2026_09_27/README.md`.

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

## 3. H6: where profile-dependent prediction arises

- **Protocol.** Frozen before scoring:
  `docs/H6_PROFILE_PATCHING_PROTOCOL_2026-09-27.md` (commit `ab7299f`), with
  dated pre-scoring notes.
- **Run.** One A100 job, 9 min 25 s: 120 receivers from 30 discovery users,
  every condition in one job. No exclusions or alignment failures. The zero
  and self-patch checks are exact.
- **Duplicates.** Two duplicate copies also ran. An A100 copy agrees bit for
  bit. An H100 copy moves every endpoint by 0.003 nats or less.

| question | result (attack days; benign days agree) |
|---|---|
| Size of the dependence | Replacing DAY and PSY raises adapted loss by +0.243 [+0.203, +0.283]. The base model does not respond (−0.001). |
| Which profile line | The DAY line alone gives +0.246. The PSY line alone gives +0.004. A second foreign partner gives +0.209. |
| Where it enters the session positions | Through hidden state 22, profile-position patches restore 98%. By 26, session-position patches restore 84% [78%, 90%]; the matched-source control restores 23%. The move happens in blocks 22 to 25. |
| Components | Attention outputs of blocks 21 and 25 each restore about 5%. Late MLPs at session positions (blocks 25, 29, 33) each restore 29% to 54%. These single-component rescues overlap. |
| Mediation at layer 26 | Restoring the adapter's whole layer-26 delta at session positions restores 83%. Restoring the 5 selected coefficients restores 0.3% (+0.0008 [−0.0006, +0.0031]), no more than control or size-matched random edits. |
| Attack-specific | None detected. Rescue fractions match between day types. Attack days are, if anything, slightly less profile-dependent. |

**Caveat on mediation.** The coefficient edits are applied at their intended
size, but their realization is approximate: normalized target error q90 1.0.
The selected features barely change under the swap. The exception is 7693,
which switches off entirely on 17 receivers, and restoring it does not help
either (post hoc, 5 users). Only these 5 features were tested.

## 4. Claims, kept separate

| claim type | what the evidence supports now |
|---|---|
| Detector performance | Not tested in this round. No endpoint here measures detection. |
| Input dependence | Supported: adaptation makes behaviour prediction sensitive to profile replacement. Pilot 2 gives +0.11 to +0.18 nats per token on confirmation users, and H6 gives +0.24 on discovery users. It happens on attack and benign days alike and comes from the organisation line, not the personality line. No familiarity effect detected. |
| Represented information | Feature-level associations only. 4596's association with lower adapted loss reproduces; 3455 goes the other way. What any feature represents is not identified. |
| Causal influence and mediation | Influence: the tested SAE edits change session predictions more than the tested PCA and mean-difference edits at the same tokens and sizes. Mediation: the profile dependence passes through session positions from blocks 22 to 25 and is carried by the adapter's layer-26 delta. It is not carried by the 5 selected coefficients under the tested restoration. These are activation and component patches, not path patching. |

## 5. Reading

**Supported (exploratory).**

- Adaptation increases behaviour prediction's sensitivity to profile
  replacement.
- Feature 4596 has a reproducible association with lower adapted loss.
- Under the existing intervention design, the selected SAE edits influence
  predictions more than the tested PCA and mean-difference alternatives.
- The profile dependence enters session positions in blocks 22 to 25 and is
  carried by the layer-26 adapter delta. The 5 selected coefficients do not
  carry it under the tested restoration (H6, development users).
- The dependence comes from the organisation line, not the personality line.
- Donor-policy bounds give a useful finite-bank robustness check. CERT
  behaviour conclusions survive any reweighting of the bank. CERT profile
  conclusions flip at 1–2% of donor weight, and TWOS behaviour at 0.24%.

**Inconclusive.**

- The selected-set attack-day adaptation gain (H4).
- Attack-specific differentials for the SAE edit (E6, E8c).
- TWOS Direction A.
- Familiarity, since a null interval is not equivalence.
- Whether attack days are less profile-dependent than benign days. One H6
  contrast excludes zero (−0.037) and the other does not; the direction
  matches Pilot 2.

**Implementation limitations.**

- More than half of the Pilot 1 edit effect comes from union-support writes
  into zero coordinates.
- CERT random controls drew independent directions per token; a coherent
  random control was run only on TWOS.
- The mean-difference baseline was amplified about 13-fold and skipped 44
  tokens.
- TWOS realization is approximate, and the TWOS evidence rests on 8
  previously examined users.
- H6 coefficient restoration is approximately realized (q90 normalized error
  1.0). Only one block in four was patched, and there is no path patching.
- The H6 cohort is development data. Three copies of the job ran because the
  duplicate watcher was offline. The primary copy was declared before any
  output was read.

**Hypotheses that need another experiment.**

- Whether the profile dependence harms detection or robustness. That needs a
  detection endpoint under profile substitution.
- Whether the mean-difference differential holds at its own positions with a
  replacement-size edit on a fresh population.
- Which heads, and which of blocks 22 to 24, move the organisation
  information into session positions. That needs every-block and head-level
  attention patching.
- Which DAY fields drive the dependence: department, role, team or project.
- Whether other layer-26 SAE features carry the dependence. That needs a
  screen of the original-minus-swapped delta on discovery users, with
  confirmation on fresh users.
- Confirmation of the H6 localization on held-out users.

## 6. Recommendation

Further work is warranted only where the paper needs it.

1. **Report H6 as a characterization.** Fine-tuning made session prediction
   depend on organisational context. That information moves into session
   positions in blocks 22 to 25, and the selected features do not carry it.
   This limits the SAE story: the selected features influence predictions
   (Pilots 1 and 4), but they do not mediate the profile dependence.
2. **Test detection before any harm claim.** The one experiment the paper
   needs next is a detection endpoint under organisation-line substitution:
   does the anomaly score or the day-level PR-AUC change? Without it, the
   dependence stays described, not judged.
3. **Treat mechanism detail as thesis work.** If it is wanted, run one
   bounded pilot on held-out users. It would cover every block from 21 to
   26, head-level attention in blocks 22 to 25, DAY field-level
   substitutions, and a feature screen of the layer-26 delta. That pilot
   should be confirmatory, with endpoints fixed from this result.
4. **Stop:**
   - Direction A;
   - further tests of the 5 selected features as mediators of the profile
     dependence;
   - any pilot on previously examined users.

   Keep the donor-policy LP as a robustness appendix.

## 7. Compute for this follow-up

| item | hardware | GPU time | CPU time |
|---|---|---|---|
| TWOS validation repair | 1 RTX 3070 (Aquaman) | 90 s | 264 s user, 20 s system |
| H6 code-path tests on TWOS (two runs) | 1 RTX 3070 (Aquaman) | 69 s + 72 s | not timed separately |
| H6 dry run, re-run under a timer (manifest identical, md5 `fe2bd2cd`) | CPU, Anvil login node | 0 | 12 s user, 2 s system (32 s wall) |
| H6 alignment unit tests | CPU, Anvil login node | 0 | 0.3 s user |
| H6 primary job 20927962 | 1 A100 (Anvil gpu-debug) | 9 min 25 s | 10 min 9 s |
| H6 duplicates 20927963 and 20928056 | 1 A100, then 1 H100 | 9 min 17 s + 5 min 17 s | 8 min 37 s + 4 min 35 s |
| H6 analyses (primary and H100 copy) | CPU, laptop | 0 | 0.6 s user each |
| **total** | | **0.46 GPU-h of the 1 GPU-h cap**, of which 0.24 is duplicate runs | about 28 min, excluding the untimed TWOS tests |
