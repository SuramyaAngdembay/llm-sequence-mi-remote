# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over positive receivers for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |    estimate |      ci_low |    ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|------------:|------------:|-----------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      26 |             2 |   4 | team           | top5     |          1301 | 0.00141845  | 0.00113942  | 0.00168969 |                  0.00344192  |                      0.00135538 |                       0.00925979 |                          0.006259   |            -0.00208653 |               -0.00300078  |
|      26 |             2 |   4 | role           | top5     |          1119 | 0.00111151  | 0.000825634 | 0.00139119 |                  0.0042316   |                      0.00646608 |                       0.0107808  |                          0.0119029  |             0.00223447 |                0.00112204  |
|      26 |             2 |   4 | dept_role      | top5     |          1039 | 0.00106737  | 0.000824039 | 0.00130497 |                 -5.94575e-05 |                      0.00163607 |                       0.0064481  |                          0.00705722 |             0.00169553 |                0.000609122 |
|      26 |             2 |   4 | dept           | top5     |          1199 | 0.000981886 | 0.000751466 | 0.00121531 |                  0.000828352 |                      0.00104829 |                       0.0071347  |                          0.00648041 |             0.00021994 |               -0.000654287 |
