# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over positive receivers for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   estimate |    ci_low |   ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------:|----------:|----------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      18 |             4 |   4 | project_role   | top5     |            70 |  0.0419751 | 0.033279  | 0.0505823 |                   -0.0863112 |                      -0.0441339 |                       -0.0357208 |                          -0.0355186 |              0.0421773 |                0.000202199 |
|      18 |             4 |   4 | dept_role      | top5     |            70 |  0.0406695 | 0.0324554 | 0.0490837 |                   -0.0846126 |                      -0.0441339 |                       -0.0353277 |                          -0.0355186 |              0.0404787 |               -0.000190854 |
|      18 |             4 |   4 | role           | top5     |            70 |  0.0406546 | 0.0323619 | 0.0491135 |                   -0.0849984 |                      -0.0441339 |                       -0.0357284 |                          -0.0355186 |              0.0408645 |                0.000209807 |
|      18 |             4 |   4 | team           | top5     |            70 |  0.0222186 | 0.0170215 | 0.0274987 |                   -0.0641441 |                      -0.0423935 |                       -0.0355539 |                          -0.0360219 |              0.0217506 |               -0.000468025 |
