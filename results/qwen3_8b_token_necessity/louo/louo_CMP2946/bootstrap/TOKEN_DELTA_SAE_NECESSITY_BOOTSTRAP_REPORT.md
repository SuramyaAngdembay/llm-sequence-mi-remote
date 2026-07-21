# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |   estimate |       ci_low |   ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|-----------:|-------------:|----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      18 |             4 |   8 | team           | top5     |        18 |                 18 |  0.0429351 |  0.0304774   | 0.0563937 |                     0.00545077 |                  -0.0073166  |                          0.0164836 |                      -0.0392188  |               -0.0127674  |                    -0.0557024 |
|      18 |             4 |   8 | dept_role      | top5     |        18 |                 18 |  0.0154282 |  0.000864101 | 0.0315112 |                     0.0056644  |                  -0.00682967 |                          0.0166078 |                      -0.0113144  |               -0.0124941  |                    -0.0279223 |
|      18 |             4 |   8 | project_role   | top5     |        18 |                 18 |  0.0134334 | -0.0048523   | 0.0326411 |                     0.00545077 |                  -0.00137864 |                          0.0164836 |                      -0.00377919 |               -0.00682941 |                    -0.0202628 |
|      18 |             4 |   8 | role           | top5     |        18 |                 18 |  0.0127232 | -0.00826412  | 0.0339214 |                     0.00545077 |                  -0.00407216 |                          0.0164836 |                      -0.00576248 |               -0.00952293 |                    -0.0222461 |
