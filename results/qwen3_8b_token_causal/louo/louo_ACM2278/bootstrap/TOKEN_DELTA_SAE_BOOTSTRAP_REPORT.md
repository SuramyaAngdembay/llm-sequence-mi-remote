# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over complete positive receiver contrasts for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   n_complete_receivers |      estimate |        ci_low |       ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------------------:|--------------:|--------------:|--------------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      18 |             4 |   8 | dept_role      | top5     |             5 |                      5 |   4.74811e-05 |  -0.000305444 |   0.000510091 |                   0.00606162 |                      0.00609584 |                        0.022199  |                           0.0221857 |             3.4225e-05 |               -1.32561e-05 |
|      18 |             4 |   8 | project_role   | top5     |             5 |                      5 |   2.98917e-05 |  -0.000245231 |   0.000242561 |                   0.00606162 |                      0.00609584 |                        0.0221814 |                           0.0221857 |             3.4225e-05 |                4.33326e-06 |
|      18 |             4 |   8 | role           | top5     |             5 |                      5 |   1.78277e-05 |  -0.000299168 |   0.000376517 |                   0.00606162 |                      0.00609584 |                        0.0221693 |                           0.0221857 |             3.4225e-05 |                1.63972e-05 |
|      18 |             4 |   8 | team           | top5     |             5 |                      0 | nan           | nan           | nan           |                 nan          |                    nan          |                      nan         |                         nan         |           nan          |              nan           |
