# Example Scores Detector Metrics

- score source: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2/example_scores.parquet`
- split filter: `eval`

| run_name     | score_name    |   n_eval |   n_pos_days |   day_pr_auc |   day_roc_auc |   day_top1pct_recall |   n_eval_users |   n_pos_users |   user_pr_auc |   user_roc_auc |   user_top1pct_recall |   heldout_user_rank_first_positive |
|:-------------|:--------------|---------:|-------------:|-------------:|--------------:|---------------------:|---------------:|--------------:|--------------:|---------------:|----------------------:|-----------------------------------:|
| qwen3_8b_r62 | adapted_nll   |     1361 |           70 |    0.0834787 |      0.680945 |                    0 |              5 |             4 |             1 |              1 |                  0.25 |                                  1 |
| qwen3_8b_r62 | base_nll      |     1361 |           70 |    0.0697788 |      0.638896 |                    0 |              5 |             4 |             1 |              1 |                  0.25 |                                  1 |
| qwen3_8b_r62 | neg_delta_nll |     1361 |           70 |    0.0681129 |      0.61599  |                    0 |              5 |             4 |             1 |              1 |                  0.25 |                                  1 |
