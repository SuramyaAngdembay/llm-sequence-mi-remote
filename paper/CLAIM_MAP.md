# Current claim-to-manuscript map

All 46 families are explicitly covered in each edition. Coverage includes limitations and deferred work; it does not mean every claim is supported.

| Claim | Current disposition | Main / readable location |
|---|---|---|
| C01: Dataset populations and positive-day counts | verified_with_scope_limits | `paper/recovery_content/data.tex:3` |
| C02: Headline 8B QLoRA adapters use benign-user training | verified_with_scope_limits | `paper/recovery_content/data.tex:3` |
| C03: Factorial and target-mask adapters use benign-user training | provenance_qualified | `paper/recovery_content/data.tex:3` |
| C04: Repaired serializer 3B adapter-seed training | provenance_qualified | `paper/recovery_content/data.tex:3` |
| C05: r6.2 detector ROC/AP and comparisons | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C06: r4.2 detector ROC/AP and comparisons | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C07: Seen/unseen-user collapse and exposure imbalance | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C08: Within-user discrimination | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C09: Inference-time profile/personality masking improves detection | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C10: Input profile/behavior swaps on attack days | fixed_input_only | `paper/recovery_content/causal_status.tex:3` |
| C11: r6.2 sparse profile-feature patching effect | pending_dependency_repair | `paper/recovery_content/method_readable.tex:3` |
| C12: r6.2 ablation dependence / necessity | pending_dependency_repair | `paper/recovery_content/method_readable.tex:3` |
| C13: r4.2 native sparse behavioral-feature patching effect | pending_dependency_repair | `paper/recovery_content/method_readable.tex:3` |
| C14: r4.2 ablation dependence | pending_dependency_repair | `paper/recovery_content/method_readable.tex:3` |
| C15: Best-candidate repair versus average-donor repair; held-out confirmation | pending_dependency_repair | `paper/recovery_content/method_technical.tex:3` |
| C16: Literal cross-benchmark feature transfer fails | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C17: r6.2 profile / r4.2 behavior attribution dissociation | pending_dependency_repair | `paper/recovery_content/method_readable.tex:3` |
| C18: SAE-seed robustness of attribution and interventions | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C19: Fixed-dictionary alignment / non-identifiability geometry | conditional_only | `paper/recovery_content/method_readable.tex:3` |
| C20: Positive-population size alone does not explain dissociation | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C21: Benign-only dictionaries reproduce dissociation / ablation | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C22: Mechanistic replication across repaired 3B adapters and seeds | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C23: Secondary r6.2 behavioral feature family and LOUO | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C24: Norm, dose, active-control and median-donor robustness | pending_dependency_repair | `paper/recovery_content/method_technical.tex:3` |
| C25: Package 4 selected edits change behavior-token predictions | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C26: Behavior-specific rather than profile-specific or attack-specific mechanism | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C27: Classical one-class detectors establish architecture-specific profile capture | pending_dependency_repair | `paper/recovery_content/discussion.tex:3` |
| C28: Train-matched 8B profile removal restores user detection; small 3B effect | verified_with_scope_limits | `paper/recovery_content/findings.tex:48` |
| C29: Factorial full/shuffle adapters preserve profile attribution | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C30: Personality novelty explains user-level failure ordering | pending_dependency_repair | `paper/recovery_content/discussion.tex:3` |
| C31: r6.2 behavior-only score mitigation and target-mask training contrast | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C32: r4.2 behavior-only score mitigation | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C33: Time-series autoencoder/forecaster comparison | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C34: Same-LM history/profile-copying experiment | verified_with_scope_limits | `paper/recovery_content/evaluation.tex:3` |
| C35: Pilot 1 anomaly specificity | diagnostic_only | `paper/recovery_content/method_readable.tex:3` |
| C36: Pilot 2 adaptation increases profile-conditioned behavior prediction | fixed_input_only | `paper/recovery_content/causal_status.tex:3` |
| C37: Pilot 3/4 feature associations and alternative-direction controls | pending_dependency_repair | `paper/recovery_content/causal_status.tex:3` |
| C38: H6 saved-state feasibility and implementation diagnostics | conditional_only | `paper/recovery_content/method_readable.tex:3` |
| C39: H6 direct ideal-intervention pilot / negligible average rescue | fixed_input_only | `paper/recovery_content/causal_status.tex:3` |
| C40: TWOS behavioral attribution / early token-class replication | outside_label_repair | `paper/recovery_content/discussion.tex:3` |
| C41: Corrected held-out TWOS intervention is inconclusive | outside_label_repair | `paper/recovery_content/discussion.tex:3` |
| C42: LANL generalization and host shortcut | outside_label_repair | `paper/recovery_content/discussion.tex:3` |
| C43: Full-code fibre geometry, feasible lifts, synthetic error-bound results | conditional_only | `paper/recovery_content/method_readable.tex:3` |
| C44: Claim status, abstracts, figures and conclusions are synchronized | synchronized_recovery_editions | `paper/recovery_content/discussion.tex:3` |
| C45: Deep SVDD, GRU AE, LSTM AE and Isolation Forest benchmark rows | pending_dependency_repair | `paper/recovery_content/discussion.tex:3` |
| C46: Local session-autoencoder mechanistic repair comparator | pending_dependency_repair | `paper/recovery_content/discussion.tex:3` |
