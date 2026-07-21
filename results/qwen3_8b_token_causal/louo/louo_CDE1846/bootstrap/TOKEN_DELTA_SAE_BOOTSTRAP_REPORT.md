# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over complete positive receiver contrasts for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   n_complete_receivers |     estimate |       ci_low |      ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------------------:|-------------:|-------------:|-------------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      18 |             4 |   8 | project_role   | top5     |            46 |                     46 |   0.00378873 |   0.00199305 |   0.00556904 |                  -0.00289377 |                      0.00165309 |                       -0.0199045 |                          -0.0191463 |             0.00454686 |                0.000758127 |
|      18 |             4 |   8 | role           | top5     |            46 |                     46 |   0.0037164  |   0.0019171  |   0.005507   |                  -0.0028979  |                      0.00165309 |                       -0.0199809 |                          -0.0191463 |             0.00455099 |                0.00083459  |
|      18 |             4 |   8 | dept_role      | top5     |            46 |                     46 |   0.00371255 |   0.0019074  |   0.00551201 |                  -0.00289378 |                      0.00165309 |                       -0.0199807 |                          -0.0191463 |             0.00454687 |                0.000834319 |
|      18 |             4 |   8 | team           | top5     |            46 |                      0 | nan          | nan          | nan          |                 nan          |                    nan          |                      nan         |                         nan         |           nan          |              nan           |
