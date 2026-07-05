# Example Scores Detector Metrics

- score source: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_r42_mb22_gc_on/example_scores.parquet`
- split filter: `eval`

| run_name     | score_name    |   n_eval |   n_pos_days |   day_pr_auc |   day_roc_auc |   day_top1pct_recall |   n_eval_users |   n_pos_users |   user_pr_auc |   user_roc_auc |   user_top1pct_recall |   heldout_user_rank_first_positive |
|:-------------|:--------------|---------:|-------------:|-------------:|--------------:|---------------------:|---------------:|--------------:|--------------:|---------------:|----------------------:|-----------------------------------:|
| qwen3_8b_r42 | adapted_nll   |    15442 |         1309 |    0.178786  |      0.675574 |           0.0290298  |             70 |            60 |      0.932708 |       0.821667 |             0.0166667 |                                  1 |
| qwen3_8b_r42 | base_nll      |    15442 |         1309 |    0.152894  |      0.662855 |           0.0282659  |             70 |            60 |      0.992415 |       0.953333 |             0.0166667 |                                  1 |
| qwen3_8b_r42 | neg_delta_nll |    15442 |         1309 |    0.0867122 |      0.531068 |           0.00763942 |             70 |            60 |      0.994409 |       0.966667 |             0.0166667 |                                  1 |
