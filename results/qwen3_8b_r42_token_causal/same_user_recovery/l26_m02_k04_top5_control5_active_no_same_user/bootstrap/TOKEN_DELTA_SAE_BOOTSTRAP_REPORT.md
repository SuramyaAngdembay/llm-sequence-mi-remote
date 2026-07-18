# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over complete positive receiver contrasts for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   n_complete_receivers |    estimate |      ci_low |    ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------------------:|------------:|------------:|-----------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      26 |             2 |   4 | team           | top5     |          1301 |                   1052 | 0.00141845  | 0.00113942  | 0.00168969 |                 -0.000192052 |                      0.00135538 |                       0.00613002 |                          0.006259   |             0.00154743 |                0.000128985 |
|      26 |             2 |   4 | role           | top5     |          1119 |                   1110 | 0.00111151  | 0.000825634 | 0.00139119 |                  0.00521034  |                      0.00646608 |                       0.0117586  |                          0.0119029  |             0.00125574 |                0.000144227 |
|      26 |             2 |   4 | dept_role      | top5     |          1039 |                   1034 | 0.00106737  | 0.000824039 | 0.00130497 |                  0.000444374 |                      0.00163607 |                       0.00693289 |                          0.00705722 |             0.0011917  |                0.00012433  |
|      26 |             2 |   4 | dept           | top5     |          1199 |                   1157 | 0.000981886 | 0.000751466 | 0.00121531 |                 -4.43505e-05 |                      0.00104829 |                       0.00636966 |                          0.00648041 |             0.00109264 |                0.000110757 |
