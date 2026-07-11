# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over matched positive/benign receiver pairs for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   estimate |    ci_low |   ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-----------:|----------:|----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      18 |             4 |   8 | role           | top5     |        70 |  0.0656883 | 0.0530478 | 0.0784803 |                     -0.0582254 |                  -0.00918147 |                         -0.0068746 |                      -0.0235189  |                 0.0490439 |                   -0.0166444  |
|      18 |             4 |   8 | project_role   | top5     |        70 |  0.0648822 | 0.0522872 | 0.0772754 |                     -0.0582254 |                  -0.0041809  |                         -0.0068746 |                      -0.0177123  |                 0.0540445 |                   -0.0108377  |
|      18 |             4 |   8 | dept_role      | top5     |        70 |  0.0568552 | 0.0462556 | 0.0674038 |                     -0.0582254 |                  -0.00212083 |                         -0.0068746 |                      -0.0076252  |                 0.0561046 |                   -0.0007506  |
|      18 |             4 |   8 | team           | top5     |        70 |  0.0465242 | 0.0365314 | 0.0558513 |                     -0.0582254 |                  -0.0147195  |                         -0.0068746 |                      -0.00989288 |                 0.0435059 |                   -0.00301828 |
