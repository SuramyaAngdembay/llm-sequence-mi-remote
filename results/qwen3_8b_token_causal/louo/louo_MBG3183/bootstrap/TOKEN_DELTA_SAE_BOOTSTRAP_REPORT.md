# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over complete positive receiver contrasts for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   n_complete_receivers |      estimate |        ci_low |       ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------------------:|--------------:|--------------:|--------------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      18 |             4 |   8 | project_role   | top5     |             1 |                      1 |   0.00215876  |   0.00215876  |   0.00215876  |                  -0.00153255 |                     -0.00153255 |                       -0.0190445 |                          -0.0212032 |                      0 |               -0.00215876  |
|      18 |             4 |   8 | dept_role      | top5     |             1 |                      1 |   0.000673532 |   0.000673532 |   0.000673532 |                  -0.00153255 |                     -0.00153255 |                       -0.0227499 |                          -0.0234234 |                      0 |               -0.000673532 |
|      18 |             4 |   8 | role           | top5     |             1 |                      1 |  -0.00392598  |  -0.00392598  |  -0.00392598  |                  -0.00153255 |                     -0.00153255 |                       -0.0277786 |                          -0.0238526 |                      0 |                0.00392598  |
|      18 |             4 |   8 | team           | top5     |             1 |                      0 | nan           | nan           | nan           |                 nan          |                    nan          |                      nan         |                         nan         |                    nan |              nan           |
