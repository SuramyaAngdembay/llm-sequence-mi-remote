Findings

  1. High impact on the detector table, not on training or the causal runs.
     The real code bug was the detector-metrics path reintroducing split=eval through the Slurm template in llm-
     sequence-mi-remote/slurm/eval_example_scores_detector_metrics_cpu.template.sbatch:18. That could silently shrink
     r6.2 detector evaluation back to the tiny 5-user slice. I fixed and pushed this in ee6a75a.
     Impact: old detector artifacts are not trustworthy as headline benchmark rows. The training runs themselves are
     unaffected.
  2. Medium impact on local-vs-remote comparison reports.
     llm-sequence-mi-remote/scripts/compare_remote_token_vs_local_session.py:40 was assuming local top5/control3 columns
     without making that explicit, even when the remote side had moved to control5_active. I patched this and pushed it
     in ee6a75a.
     Impact: existing comparison reports are not necessarily numerically wrong, but they were under-specified. If they
     are used in the paper, they should be regenerated or clearly labeled with the local intervention/control pair used.
  3. Medium scientific risk remains in causal and necessity matching.
     The causal and necessity scripts still allow same-user matches in the benign/control pool:
      - llm-sequence-mi-remote/scripts/eval_token_delta_sae_causal.py:287
      - llm-sequence-mi-remote/scripts/eval_token_delta_sae_necessity.py:49
        I quantified this from committed best-row artifacts:
      - r6.2 causal benign donors: about 13.6% same-user
      - r4.2 causal benign donors: about 20.4% same-user
      - r6.2 necessity benign matches: about 2.9% same-user
      - r4.2 necessity benign matches: about 20.9% same-user
        Impact: this does not prove the causal results are invalid, but it is the main remaining robustness risk. It can
        inflate "repair" by borrowing a user's own benign style.

  What does not look broken

  - I did not find label leakage into remote QLoRA training.
    llm-sequence-mi-remote/scripts/build_session_jsonl.py:68 and llm-sequence-mi-remote/scripts/train_qlora.py:62 still
    implement benign-only training correctly.
  - I did not find a sign bug in the causal or necessity summaries.
  - I did not find evidence that the major 8B mechanistic results are a pure code artifact.

  What has to be rerun

  - Must rerun / replace for the paper
      - The detector benchmark table should use the new fold-aligned remote detector runs, not the old detector_metrics/
        artifacts. The fold-aligned path is the correct fix.
  - Should rerun if these reports are going into the paper
      - Any REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md generated from llm-sequence-mi-remote/scripts/
        compare_remote_token_vs_local_session.py, especially:
          - results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_*
          - results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_*
          - results/qwen3b_pilot/strict_compare_remote70_daylevel_controlfix/*
      - The numbers may stay the same if you still intend local top5/control3, but the regenerated reports will at least
        be explicit.
  - Best robustness reruns
      - r6.2 8B headline causal/active-control and necessity with same-user exclusion
      - r4.2 8B native causal/active-control and necessity with same-user exclusion
        These are the most valuable reruns if you want to harden the mechanistic claim.

  How much damage control is needed

  - Low for the training story.
      - I do not see evidence that the QLoRA training runs themselves are invalid.
  - Medium for the detector-competitiveness story.
      - The old detector table was overstated. This is the part that needed correction most.
  - Low to medium for the mechanistic story.
      - The existence of positive r6.2 and native r4.2 remote mechanisms still looks real.
      - The strongest remaining caveat is same-user matching, not a discovered sign/math bug.

  So the paper is not collapsing. The main damage is:

  - detector table needed correction
  - comparison reports needed clarification
  - causal claims need one more robustness check if you want them maximally defensible

  Why these happened

  - The codebase evolved quickly across:
      - direct-transfer runs
      - native-search runs
      - active-control audits
      - necessity runs
      - detector closeout
  - That created wrapper/template drift:
      - wrapper fixed
      - template still had stale default
  - eval.jsonl also has unusual semantics:
      - it intentionally contains val + eval
      - that makes it easy to accidentally double-filter and think you are scoring "eval" when you are actually
        shrinking the pool again
  - The local-vs-remote comparison script started as a narrow one-off for top5/control3 and was reused after the control
    regime evolved
  - Same-user matching likely happened because the scripts were optimized for availability of matched controls on
    limited positive pools, not because of a blatant coding mistake

  Best next steps

  1. Let the Anvil fold-aligned detector runs finish and use those as the only headline detector table.
  2. Regenerate the local-vs-remote comparison reports you actually plan to cite.
  3. Run same-user-excluded robustness checks for:
      - r6.2 8B
      - r4.2 8B
  4. After that, freeze the CERT story.

  Current fix status

  - Pushed: ee6a75a Fix detector metrics split default and compare assumptions

  If you want, I can next set up the same-user-excluded causal/necessity rerun scripts in GitHub so the Anvil agent can
  launch the exact robustness checks.
