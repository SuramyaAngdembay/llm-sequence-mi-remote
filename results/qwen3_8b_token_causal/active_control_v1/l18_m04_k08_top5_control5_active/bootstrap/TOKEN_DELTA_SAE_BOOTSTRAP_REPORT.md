# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over positive receivers for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   estimate |     ci_low |   ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------:|-----------:|----------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      18 |             4 |   8 | role           | top5     |            70 | 0.0186531  | 0.0133957  | 0.0243907 |                   -0.068197  |                      -0.0496998 |                       -0.010863  |                          -0.0110189 |             0.0184972  |               -0.000155952 |
|      18 |             4 |   8 | dept_role      | top5     |            70 | 0.0163783  | 0.0120514  | 0.0210683 |                   -0.065729  |                      -0.0496998 |                       -0.0106698 |                          -0.0110189 |             0.0160292  |               -0.000349134 |
|      18 |             4 |   8 | project_role   | top5     |            70 | 0.015664   | 0.0113144  | 0.0204483 |                   -0.0653378 |                      -0.0496998 |                       -0.010993  |                          -0.0110189 |             0.015638   |               -2.59089e-05 |
|      18 |             4 |   8 | team           | top5     |            70 | 0.00814964 | 0.00558394 | 0.0108177 |                   -0.0516228 |                      -0.0441116 |                       -0.0109281 |                          -0.0115666 |             0.00751119 |               -0.000638458 |
