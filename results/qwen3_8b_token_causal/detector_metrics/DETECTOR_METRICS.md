# Example Scores Detector Metrics

- score source: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/token_delta_cache/qwen3_8b_session_token_deltas_targeted_mb12_gc_on_fresh_v2/example_scores.parquet`
- split filter: `<all rows in scores parquet>`

| run_name     | score_name    |   n_eval |   n_pos_days |   day_pr_auc |   day_roc_auc |   day_top1pct_recall |   n_eval_users |   n_pos_users |   user_pr_auc |   user_roc_auc |   user_top1pct_recall |   heldout_user_rank_first_positive |
|:-------------|:--------------|---------:|-------------:|-------------:|--------------:|---------------------:|---------------:|--------------:|--------------:|---------------:|----------------------:|-----------------------------------:|
| qwen3_8b_r62 | adapted_nll   |   142072 |           70 |  0.00050173  |      0.547214 |                    0 |            410 |             4 |     0.0220714 |       0.716133 |                     0 |                                 37 |
| qwen3_8b_r62 | base_nll      |   142072 |           70 |  0.000372084 |      0.395164 |                    0 |            410 |             4 |     0.0271368 |       0.542488 |                     0 |                                 15 |
| qwen3_8b_r62 | neg_delta_nll |   142072 |           70 |  0.000379994 |      0.404568 |                    0 |            410 |             4 |     0.0106492 |       0.457512 |                     0 |                                155 |
