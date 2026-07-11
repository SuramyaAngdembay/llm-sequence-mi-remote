# Example Scores Detector Metrics

- score source: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on/example_scores.parquet`
- split filter: `<all rows in scores parquet>`

| run_name     | score_name    |   n_eval |   n_pos_days |   day_pr_auc |   day_roc_auc |   day_top1pct_recall |   n_eval_users |   n_pos_users |   user_pr_auc |   user_roc_auc |   user_top1pct_recall |   heldout_user_rank_first_positive |
|:-------------|:--------------|---------:|-------------:|-------------:|--------------:|---------------------:|---------------:|--------------:|--------------:|---------------:|----------------------:|-----------------------------------:|
| qwen3_8b_r42 | adapted_nll   |    42468 |         1309 |    0.0686891 |      0.670378 |            0.0381971 |            149 |            60 |      0.54754  |       0.666479 |                     0 |                                  2 |
| qwen3_8b_r42 | base_nll      |    42468 |         1309 |    0.0718236 |      0.688279 |            0.0359053 |            149 |            60 |      0.508732 |       0.636704 |                     0 |                                  3 |
| qwen3_8b_r42 | neg_delta_nll |    42468 |         1309 |    0.0320842 |      0.530171 |            0.0114591 |            149 |            60 |      0.43065  |       0.542135 |                     0 |                                  3 |
