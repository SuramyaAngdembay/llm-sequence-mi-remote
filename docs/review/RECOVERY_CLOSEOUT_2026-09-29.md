# Recovery execution: evidence, dependencies, and manuscripts

This is completion of the **recovery consolidation and audit pass**, not completion of every historical causal experiment. Worktree: `repair/cert-label-pipeline-20260929`. The original user/Claude checkout is unchanged.

## Completed in this pass

1. Consolidated all 46 claim families and eight checksum-bound evidence sources into `paper/data/recovery_registry.json`. Already-verified score evaluations were reused, not rerun.
2. Audited **actual saved local-autoencoder bundles** on Magnolia. The audit reads each checkpoint's saved fold, training arguments, scaler dimensions, and artifact hash without fitting or inference.
3. Relabeled Pilot 2 and H6 on CPU; audited partner eligibility; retained fixed-cohort and eligible-subset summaries separately. Checked input-population hashes and receiver identities.
4. Audited the input-swap donor manifests against official insiders and corrected receiver labels.
5. Built main, readable-full, and plain-summary from shared corrected evidence. Readable-full includes the main technical material in its appendix. Previous sources are preserved at commit `e0ad1a3` and in the external `manuscripts_before/` archive.
6. Replaced the generators that would have silently restored stale numbers. All three current builds share the registry, tables, figure, and explicit limitations. All 46 families have a current manuscript location.

## New findings

### C46: saved local-autoencoder membership

| Dataset | Bundles / folds | Training includes official insiders | Test includes extra official insiders |
|---|---:|---:|---:|
| r4.2 | 120 / 60 | 94 | 120 |
| r6.2 | 8 / 4 | 8 | 0 |

The 26 r4.2 bundles with clean training membership are **candidates for restricted reuse**. Their test membership and label-dependent downstream selection still need correction. This audit does not clear any historical comparator table as a whole. C45's separate sequence benchmark uses the same faulty fold builder, but saves aggregate results rather than reusable checkpoints or per-example scores. C27 contamination was already independently established: ten r4.2 insiders and PLJ1771 in r6.2 in its training pool.

### C10: original input-swap pairs

| Dataset | Historical receivers | Still positive | Donor from official insider | Same-user donor | Eligible fixed pairs | Missing positive receivers |
|---|---:|---:|---:|---:|---:|---:|
| r4.2 | 1,309 | 1,132 | 475 | 6 | 723 | 230 |
| r6.2 | 70 | 60 | 0 | 0 | 60 | 13 |

These columns overlap and must not be summed. Eligibility requires a corrected positive receiver, a non-insider donor, and a different user. Corrected r4.2 recovery needs new donors for some existing receivers as well as scores for missing positives. The earlier fixed-score reanalysis remains a record of its inputs; this new step audits its population validity.

### C36: historical Pilot 2

- 1,636 rows; 414 receivers; 30 users.
- 129 receiver labels flip. Corrected groups: 234 incident-window and 180 window-negative receivers.
- 86 of 1,222 scored partner uses are ineligible; 86 receivers have an ineligible partner in at least one condition.
- Frozen-cohort other-department swap effect on behavior loss, adapted minus base: **+0.185792**, descriptive user-bootstrap 95% interval **[0.150229, 0.223316]**.
- Corrected within-user attack-minus-window-negative contrast for that endpoint: **−0.038565**, interval **[−0.058851, −0.019395]**, 24 shared users.

The profile-sensitivity observation survives within its frozen inputs. It does **not** establish attack-specific harm or mediation. The 24-user contrast is not a difference between the two marginal group means, which use different user sets. Eligible-partner sensitivities are recorded separately; neither restores historical day matching. Feature-activation F3 is not promoted: selection still depends on the historical dictionary and labels.

### C39: H6 direct-intervention pilot

- 1,320 rows; 120 receivers; 30 users.
- 48 receiver labels flip; corrected groups contain 66 incident-window and 54 window-negative receivers.
- 44 receivers have an official insider among their two nominally benign profile partners.
- Original pooled I2 rescue **+0.00112532** and I1 rescue **−0.00025030** reproduce. Corrected within-user I2 attack-minus-window-negative contrast: **−0.00029893**, 95% bootstrap interval **[−0.00162816, 0.00092851]**, 20 shared users.

Small pooled averages retain their fixed-input meaning. They do not establish equivalence for every feasible lift or every user, and cannot clear corrected mechanism claims. Partner-subset summaries are explicitly post hoc.

## Checks and artifacts

Evidence root: `/Users/suramya/Documents/mi-paper-recovery-closeout-2026-09-29/`.

- `saved_baseline_fold_audit.json`: every bundle hash, saved fold and training arguments.
- `fixed_pilots/fixed_pilot_recovery.json`: complete grouped CPU analyses with source hashes.
- `fixed_pilots/*_partner_audit.csv`, `*_receivers.csv`, `*_effects.csv`, `C10_*_donor_audit.csv`: explicit eligibility/response records.
- `fixed_pilots/independent_checks.json`: **1,680** numbers recomputed with separate aggregation code; maximum error **5.56e-17**. Pooled H6 matches the previous independent audit within **4.12e-18**.
- `paper/data/claim_manuscript_map.json` and `paper/CLAIM_MAP.md`: every family in all three editions. A mapped pending claim is not a supported result.
- `paper/build/recovery_build_manifest.json`: PDF hashes and registry hash.
- `docs/review/TARGETED_RERUNS_2026-09-29.md`: next experiments and their launch gates.

The label pipeline and reported corrected LM results are repaired. **Historical causal, comparator, and broad replication claims remain partially unrepaired.** C35's original global gate remains failed. No GPU experiments, new adapter training, remote publication, or thesis changes were performed in this pass.

Validation: 28 targeted tests passed; three PDF builds and claim/macro mapping passed; no unresolved citations/references or overfull boxes in the inspected logs. PDF visual inspection covered the readable opening page and results figure/tables. Source and build hashes are verified in `verification.json` in the closeout evidence root.
