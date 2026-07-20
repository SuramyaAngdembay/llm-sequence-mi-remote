# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over complete positive receiver contrasts for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   n_complete_receivers |    estimate |       ci_low |     ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------------------:|------------:|-------------:|------------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      26 |             2 |   4 | dept           | top5     |           605 |                    605 | 0.000757697 |  0.000445824 | 0.00108797  |                   0.00259699 |                      0.00323127 |                       0.00535474 |                          0.00523133 |            0.000634284 |               -0.000123413 |
|      26 |             2 |   4 | dept_role      | top5     |           528 |                    528 | 0.000446293 |  5.5859e-05  | 0.000841337 |                   0.00378525 |                      0.00384821 |                       0.00664797 |                          0.00626464 |            6.29653e-05 |               -0.000383327 |
|      26 |             2 |   4 | team           | top5     |           638 |                    516 | 0.000397797 | -9.72623e-05 | 0.000870747 |                  -0.00718824 |                     -0.0089804  |                      -0.00502562 |                         -0.00721558 |           -0.00179216  |               -0.00218996  |
|      26 |             2 |   4 | role           | top5     |           528 |                    528 | 0.000339938 | -4.63937e-05 | 0.000734529 |                   0.00380529 |                      0.0036096  |                       0.00678456 |                          0.00624893 |           -0.000195693 |               -0.000535631 |
