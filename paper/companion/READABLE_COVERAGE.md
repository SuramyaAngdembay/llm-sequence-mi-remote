# Readable edition: coverage and evidence

Revised September 14, 2026 against `8a4e9d78075a2c23400336290e308932af4d423c` (remote `main` at review time).

The aim is a complete account of the paper's evidence with a linear main narrative. Findings and limitations that change interpretation stay in the body. Exact protocols, quantitative controls, and supporting tables go in the appendix. This is an editorial and artifact-level revision; no models were retrained.

## Coverage of the longer manuscript

| Evidence in `paper/main.tex` | Readable narrative | Supporting detail |
|---|---|---|
| Motivation, one-class detection, ranking versus intended information | §§1–3 | G: relation to earlier work |
| CERT provenance, raw versus matched population | §2 | A.1: counts and comparison cohorts |
| Serialization, omitted fields, repaired serializer | §§2, 6 | A.2: schema and training recipes |
| QLoRA, likelihood score, base/adapted delta | §§3–4 | A.2–A.3: settings, score targets, dictionary fitting |
| SAE configuration, gaps, active controls | §4 | A.3: dimensions, thresholds, reconstruction quality, selection status |
| Repair, ablation, complete-case contrasts | §4 | C.1: exact operations, signs, matching, search |
| Cluster uncertainty and multiple contexts | §§4, 6 | C.3: resampling unit, weights, supplementary procedures |
| Fold-aligned detection and weak AP | §3, Figure 4 | B.1: all five detectors on both CERT releases, base-model scores |
| Exposure imbalance and user-disjoint collapse | §5.2, Figure 4 | B.1: pooled versus fold values, prevalence caveat |
| Within-user ranking | §5.2 | B.2: intervals and user counts |
| r6.2 primary repair and ablation | §5.1, Figure 2 | D.2: per-user concentration and held-out results |
| r4.2 native repair and partial ablation | §5.1, Figure 2 | D.1–D.2: contexts and confirmation |
| Profile/session attribution and enrichment | §4.4, §5.1, Figure 3 | A.3 and D: selection rules and replication scope |
| Direct feature-transfer failure | §5.3 | D.1: transfer estimates and local-autoencoder comparator |
| SAE seeds and matched-layer geometry | §6 | D.1: destandardized similarities and reference medians |
| Held-out users and configuration dependence | §6 | D.2: discovery sizes, overlap, effects, failed confirmations |
| Benign-only dictionaries | §6 | D.2: attribution, ablation success, repair failure |
| Positive-population subsampling | §6 | D.3: repetitions and fixed-dictionary limitation |
| Repaired 3B adapters and training seeds | §6 | A.2, D.3: corpus caps, layers, attribution, interventions |
| Secondary r6.2 behavioral family | §6 | D.3: recurrence, repair, partial ablation |
| Input masks and profile/behavior swaps | §§5.2, 6 | B.2: conditions, populations, intervals, interpretation |
| Dose, median-donor, and norm controls | §4 | C.2: estimates including failed norm-normalized r6.2 repair |
| TWOS | §7 | E: split, training, counts, intervention and selection scope |
| LANL | §8 | E: sampling, folds, host association, token-length check scope |
| Classical probe | §9 | F.1: actual split, all six results, recipes, importance definition |
| Four serialization conditions at 3B and 8B | §10 | A.2, F.2: recipe differences, cohort, bootstrap, individual ranks |
| Full/shuffle attribution | §10 | F.2: exact profile fractions for each factorial adapter |
| Personality novelty | §10 | F.2: all four cases and competing novelty measures |
| Discussion, limits, practical implications | §11 | Caveats also accompany affected results |
| Broader impact and reproducibility | §11 | G and source guide below |
| Literature context | §1 and inline citations | G and shared `paper/references.bib` |

Operational chronology, duplicate claim recaps, historical job-management prose, and an unverified aggregate compute total are not copied. Hardware and execution recipes remain inspectable in the repository. The historical compute total in `main` should not be assumed to cover the later H100 factorial and LANL jobs.

## Corrections in this edition

1. **Populations:** distinguish raw answer-key rows, matched days, validation benigns, the 406-negative factorial cohort, the expanded masking pool, and the classical probe's separate random split.
2. **Definitions:** include ROC ties; score valid next-token targets; use signed edited-minus-original NLL; state complete-case pairing and best-donor/strength search.
3. **Interventions:** retain all-token reconstruction, unequal edit sizes and donor opportunities, dose matching, median-donor and norm-normalization results, including failures. Neither edit proves strict logical sufficiency or necessity.
4. **Attribution:** activation location is not a complete feature meaning or the whole decision. The intervention tests a feature set. The simpler architecture figure follows the four explanation steps.
5. **Replication:** keep transfer failure, secondary r6.2 behavioral structure, dictionary dependence, and distinct SAE/adapter seed claims. Population subsampling isolates selection in a fixed model and dictionary.
6. **LANL:** AP around 0.013 is above prevalence 0.0065 and is not precision at a chosen alert threshold. Host masking improves ROC without improving AP. Host sensitivity does not uniquely identify an identity mechanism. Token-length checks cover the scored full and host-anonymized conditions.
7. **Classical probe:** the old readable statement “the same holds on r4.2” contradicts stored results. Removal lowers IF and SVM AUC on r4.2, and personality trait O enters the SVM top five. Report all six results and the split difference.
8. **Factorial:** show day and user AUC together. At 8B, no-personality and no-profile user AUCs are similar, while day AUCs differ (0.714 versus 0.827). The conditions change adaptation and scoring together; `DAY` removal also removes week. Across model families, learning rates differ as well as model generation and size.
9. **Shuffle:** fixed reassignment preserves a pseudonym. Original ownership does not establish whether the assigned profile appeared during adaptation. Remove the incorrect exposure argument.
10. **Novelty and conclusions:** four-user neighborhood counts support a hypothesis, not an established familiarity rule. A two-factor law and a causal size effect remain unproven.

## Source guide

Paths are relative to the repository root. Use same-user-excluded, user-cluster reports for headline mechanisms; older permissive-donor or receiver-bootstrap outputs are different analyses.

| Evidence | Source |
|---|---|
| Data and training | `scripts/build_session_jsonl.py`, `scripts/build_session_jsonl_fast.py`, `scripts/train_qlora.py`; `configs/qwen3_8b_qlora_session_targeted.yaml`, `configs/qwen3b_qlora_session.yaml` |
| Intervention definitions | `scripts/eval_token_delta_sae_causal.py`, `scripts/eval_token_delta_sae_necessity.py`, `scripts/cluster_bootstrap_token_delta_sae.py` |
| r6.2 repair | `results/qwen3_8b_token_causal/same_user_recovery/l18_m04_k08_top5_control5_active_no_same_user/cluster_bootstrap/CLUSTER_BOOTSTRAP_REPORT.md` |
| r6.2 ablation | `results/qwen3_8b_token_necessity/same_user_recovery/l18_m04_k08_top5_control5_active_necessity_no_same_user/cluster_bootstrap/CLUSTER_BOOTSTRAP_REPORT.md` |
| r4.2 repair | `results/qwen3_8b_r42_token_causal/same_user_recovery/l26_m02_k04_top5_control5_active_no_same_user/cluster_bootstrap/CLUSTER_BOOTSTRAP_REPORT.md` |
| r4.2 ablation | `results/qwen3_8b_r42_token_necessity/same_user_recovery/l26_m02_k04_top5_control5_active_necessity_no_same_user/cluster_bootstrap/CLUSTER_BOOTSTRAP_REPORT.md` |
| Detector folds | `scripts/eval_fold_aligned_detector_metrics.py`; `detector_metrics_fold_aligned/` under each CERT causal result directory |
| Exposure, within-user, masking, swaps | `results/valonly_detector/`, `results/within_user_ranking/`, `results/masking_ablation/`, `results/swap_counterfactuals/` |
| Magnitude and donor controls | `results/matched_controls/`; `results/benign_dictionary/intervention_norms_r62/`, `intervention_norms_r42/`, and `norms_*_necessity_summary.json` |
| Transfer and native search | `results/qwen3_8b_r42_token_causal/stream_uncapped_v2/`, `native_search_v3_bs24/`; `paper/tables/cert_mechanistic_summary.tex` for the local comparator |
| Seed geometry | `results/sae_seed_stability/alignment_destandardized.json`, `scripts/sae_seed_alignment.py` |
| Population and replications | `results/population_subsample_r42/`, `results/benign_dictionary/`, `results/qwen3b_repaired/`, `results/behavioral_louo/` |
| TWOS | `results/twos_replication/RESULTS.md` and accompanying results |
| LANL | `results/lanl_auth_replication/ap_*.json`, `lanl_checks_output.txt`; `scripts/lanl/`; `slurm/anvil_friend/lanl_run.sh` |
| Classical probe | `results/cross_arch_probe/cross_arch_probe.py`, `r62.json`, `r42.json` |
| Factorial and novelty | `results/train_matched_factorial/README.md`, `anvil_run_summaries.txt`; `scripts/factorial/`; `slurm/anvil_friend/run_mode.sh`, `run_8b.sh` |

**Factorial source distinction:** `3b/roc/` and `8b/roc/` hold earlier quick evaluations, not the later fold re-score reported in the paper. Use fold results recorded in the README and run summaries. Four-user intervals condition on those adapters and the fixed cohort; they do not replace training-seed replication.

## Validation

- Compiled with Tectonic and BibTeX, followed by reference-resolution passes.
- Checked citation keys, cross-references, unique labels, and TeX warnings.
- Inspected the rendered PDF, including the architecture, results, and appendix tables.
- Cross-checked added factorial, LANL, classical, and protocol statements against result artifacts and execution code.
- Retained empirical limitations. This edit does not rerun GPU experiments or establish new mechanistic claims.
