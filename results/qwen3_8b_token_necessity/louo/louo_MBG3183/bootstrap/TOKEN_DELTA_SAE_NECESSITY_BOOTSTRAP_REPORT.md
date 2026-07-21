# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |   estimate |     ci_low |    ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|-----------:|-----------:|-----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      18 |             4 |   8 | team           | top5     |         1 |                  1 | -0.0223159 | -0.0223159 | -0.0223159 |                     0.00419432 |                   0.00238666 |                         -0.0174394 |                       0.00306883 |               -0.00180766 |                    0.0205082  |
|      18 |             4 |   8 | role           | top5     |         1 |                  1 | -0.0244445 | -0.0244445 | -0.0244445 |                     0.00419432 |                  -0.00442818 |                         -0.0174394 |                      -0.0016174  |               -0.0086225  |                    0.015822   |
|      18 |             4 |   8 | dept_role      | top5     |         1 |                  1 | -0.0275514 | -0.0275514 | -0.0275514 |                     0.0188718  |                  -0.015945   |                         -0.0074181 |                      -0.0146835  |               -0.0348168  |                   -0.00726545 |
|      18 |             4 |   8 | project_role   | top5     |         1 |                  1 | -0.0279778 | -0.0279778 | -0.0279778 |                     0.00419432 |                  -0.00123164 |                         -0.0174394 |                       0.00511244 |               -0.00542596 |                    0.0225518  |
