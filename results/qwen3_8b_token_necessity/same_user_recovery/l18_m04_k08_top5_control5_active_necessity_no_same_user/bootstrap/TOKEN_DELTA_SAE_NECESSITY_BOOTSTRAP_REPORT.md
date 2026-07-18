# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |   estimate |    ci_low |   ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|-----------:|----------:|----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      18 |             4 |   8 | project_role   | top5     |        70 |                 70 |  0.0651885 | 0.0551452 | 0.0750226 |                     -0.0581652 |                  -0.0033418  |                        -0.00716735 |                      -0.0175324  |                 0.0548234 |                   -0.0103651  |
|      18 |             4 |   8 | role           | top5     |        70 |                 70 |  0.0621669 | 0.0502648 | 0.0733641 |                     -0.0592412 |                  -0.00146433 |                        -0.0079581  |                      -0.0123481  |                 0.0577769 |                   -0.00438998 |
|      18 |             4 |   8 | dept_role      | top5     |        70 |                 70 |  0.0566027 | 0.0449963 | 0.0689897 |                     -0.0592412 |                  -0.00442777 |                        -0.0079581  |                      -0.00974737 |                 0.0548134 |                   -0.00178927 |
|      18 |             4 |   8 | team           | top5     |        70 |                 70 |  0.0522338 | 0.0420541 | 0.061397  |                     -0.0592412 |                  -0.0150535  |                        -0.0079581  |                      -0.0160042  |                 0.0441877 |                   -0.00804606 |
