# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over matched positive/benign receiver pairs for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |    estimate |       ci_low |    ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|------------:|-------------:|-----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      26 |             2 |   4 | role           | top5     |      1309 | 0.00237023  |  0.00109184  | 0.00363536 |                     0.0092782  |                  0.000696244 |                          0.0124684 |                       0.00151617 |               -0.00858195 |                   -0.0109522  |
|      26 |             2 |   4 | dept_role      | top5     |      1309 | 0.00147522  |  0.00024629  | 0.00266078 |                     0.00929111 |                 -0.00304549  |                          0.0126495 |                      -0.0011623  |               -0.0123366  |                   -0.0138118  |
|      26 |             2 |   4 | team           | top5     |      1309 | 0.000681123 | -0.000757926 | 0.00202781 |                     0.00915748 |                  0.000648516 |                          0.0125763 |                       0.00338618 |               -0.00850896 |                   -0.00919008 |
|      26 |             2 |   4 | dept           | top5     |      1309 | 0.000263401 | -0.00101817  | 0.00145223 |                     0.0094187  |                 -0.00401858  |                          0.0125505 |                      -0.00115017 |               -0.0134373  |                   -0.0137007  |
