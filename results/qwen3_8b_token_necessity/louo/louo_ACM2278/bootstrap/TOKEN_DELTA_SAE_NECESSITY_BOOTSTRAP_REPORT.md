# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |   estimate |      ci_low |    ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|-----------:|------------:|-----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      18 |             4 |   8 | role           | top5     |         5 |                  5 |  0.0299453 |  0.00844319 | 0.0677494  |                      0.0045701 |                  -0.0193035  |                          0.0220474 |                      -0.0317714  |              -0.0238736   |                   -0.0538188  |
|      18 |             4 |   8 | project_role   | top5     |         5 |                  5 |  0.0268591 |  0.00246629 | 0.0543025  |                      0.0045701 |                   0.00359316 |                          0.0220474 |                      -0.00578855 |              -0.000976944 |                   -0.027836   |
|      18 |             4 |   8 | dept_role      | top5     |         5 |                  5 |  0.0115491 | -0.0193118  | 0.037782   |                      0.0045701 |                  -0.0163029  |                          0.0220474 |                      -0.0103747  |              -0.020873    |                   -0.0324222  |
|      18 |             4 |   8 | team           | top5     |         5 |                  5 | -0.0149208 | -0.0328686  | 0.00302691 |                      0.0045701 |                  -0.00127183 |                          0.0220474 |                       0.0311264  |              -0.00584193  |                    0.00907891 |
