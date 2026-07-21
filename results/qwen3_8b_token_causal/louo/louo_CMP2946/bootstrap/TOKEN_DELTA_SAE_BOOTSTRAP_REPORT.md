# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over complete positive receiver contrasts for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   n_complete_receivers |      estimate |       ci_low |      ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------------------:|--------------:|-------------:|-------------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      18 |             4 |   8 | role           | top5     |            18 |                     18 |   8.29117e-05 |  -0.00117109 |   0.00127753 |                   0.00876615 |                      0.00635093 |                        0.0187597 |                           0.0162616 |            -0.00241522 |                -0.00249813 |
|      18 |             4 |   8 | dept_role      | top5     |            18 |                     18 |   6.7585e-06  |  -0.00134363 |   0.00123587 |                   0.00878813 |                      0.00635093 |                        0.0187056 |                           0.0162616 |            -0.0024372  |                -0.00244396 |
|      18 |             4 |   8 | project_role   | top5     |            18 |                     18 |  -9.31438e-05 |  -0.0016028  |   0.00125656 |                   0.00878813 |                      0.00635093 |                        0.0186057 |                           0.0162616 |            -0.0024372  |                -0.00234406 |
|      18 |             4 |   8 | team           | top5     |            18 |                      0 | nan           | nan          | nan          |                 nan          |                    nan          |                      nan         |                         nan         |           nan          |               nan          |
