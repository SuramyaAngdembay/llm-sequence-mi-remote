# Fold-Aligned Remote Detector Metrics

- score source: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r62_fullpop_scores/example_scores.parquet`
- run name: `qwen3_8b_r62_fold_aligned`
- fold seed: `42`
- benign test users per fold: `800`
- protocol: fixed benign-trained remote model, evaluated on the same leave-one-malicious-user-out test-user construction as the local detector baselines

## Summary

| score_name    |   folds |   day_pr_auc_mean |   day_roc_auc_mean |   day_top1_recall_mean |   user_pr_auc_mean |   user_roc_auc_mean |   user_top1pct_recall_mean |   heldout_user_rank_mean |   heldout_user_rank_median |
|:--------------|--------:|------------------:|-------------------:|-----------------------:|-------------------:|--------------------:|---------------------------:|-------------------------:|---------------------------:|
| adapted_nll   |       4 |       0.000754631 |           0.953157 |                      0 |         0.0537037  |            0.97125  |                          0 |                    24    |                       28.5 |
| base_nll      |       4 |       5.65696e-05 |           0.460541 |                      0 |         0.0095202  |            0.532188 |                          0 |                   375.25 |                      385.5 |
| neg_delta_nll |       4 |       4.14663e-05 |           0.20627  |                      0 |         0.00133533 |            0.065    |                          0 |                   749    |                      748   |

## Per-Fold Rows

| run_name                  | score_name    |   fold | heldout_pos_user   |   n_test_rows |   n_test_pos_rows |   day_pr_auc |   day_roc_auc |   day_top1_recall |   n_test_users |   n_test_pos_users |   user_pr_auc |   user_roc_auc |   user_top1pct_recall |   heldout_user_rank |
|:--------------------------|:--------------|-------:|:-------------------|--------------:|------------------:|-------------:|--------------:|------------------:|---------------:|-------------------:|--------------:|---------------:|----------------------:|--------------------:|
| qwen3_8b_r62_fold_aligned | adapted_nll   |      0 | ACM2278            |        283000 |                 5 |  0.000117659 |     0.915116  |                 0 |            801 |                  1 |    0.0333333  |        0.96375 |                     0 |                  30 |
| qwen3_8b_r62_fold_aligned | base_nll      |      0 | ACM2278            |        283000 |                 5 |  1.26042e-05 |     0.204806  |                 0 |            801 |                  1 |    0.00181818 |        0.31375 |                     0 |                 550 |
| qwen3_8b_r62_fold_aligned | neg_delta_nll |      0 | ACM2278            |        283000 |                 5 |  1.14281e-05 |     0.10773   |                 0 |            801 |                  1 |    0.00135501 |        0.07875 |                     0 |                 738 |
| qwen3_8b_r62_fold_aligned | adapted_nll   |      1 | CDE1846            |        280079 |                46 |  0.00197208  |     0.96094   |                 0 |            801 |                  1 |    0.0333333  |        0.96375 |                     0 |                  30 |
| qwen3_8b_r62_fold_aligned | base_nll      |      1 | CDE1846            |        280079 |                46 |  0.000131169 |     0.422953  |                 0 |            801 |                  1 |    0.00143472 |        0.13    |                     0 |                 697 |
| qwen3_8b_r62_fold_aligned | neg_delta_nll |      1 | CDE1846            |        280079 |                46 |  0.000112697 |     0.279745  |                 0 |            801 |                  1 |    0.00132626 |        0.05875 |                     0 |                 754 |
| qwen3_8b_r62_fold_aligned | adapted_nll   |      2 | CMP2946            |        278317 |                18 |  0.000601989 |     0.947489  |                 0 |            801 |                  1 |    0.037037   |        0.9675  |                     0 |                  27 |
| qwen3_8b_r62_fold_aligned | base_nll      |      2 | CMP2946            |        278317 |                18 |  4.79852e-05 |     0.317774  |                 0 |            801 |                  1 |    0.00452489 |        0.725   |                     0 |                 221 |
| qwen3_8b_r62_fold_aligned | neg_delta_nll |      2 | CMP2946            |        278317 |                18 |  3.61154e-05 |     0.0720253 |                 0 |            801 |                  1 |    0.00134771 |        0.07375 |                     0 |                 742 |
| qwen3_8b_r62_fold_aligned | adapted_nll   |      3 | MBG3183            |        280240 |                 1 |  0.000326797 |     0.989084  |                 0 |            801 |                  1 |    0.111111   |        0.99    |                     0 |                   9 |
| qwen3_8b_r62_fold_aligned | base_nll      |      3 | MBG3183            |        280240 |                 1 |  3.45197e-05 |     0.896631  |                 0 |            801 |                  1 |    0.030303   |        0.96    |                     0 |                  33 |
| qwen3_8b_r62_fold_aligned | neg_delta_nll |      3 | MBG3183            |        280240 |                 1 |  5.62461e-06 |     0.365581  |                 0 |            801 |                  1 |    0.00131234 |        0.04875 |                     0 |                 762 |
