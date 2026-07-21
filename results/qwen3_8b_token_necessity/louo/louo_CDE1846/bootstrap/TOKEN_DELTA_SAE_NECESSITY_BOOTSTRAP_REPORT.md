# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |    estimate |     ci_low |    ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|------------:|-----------:|-----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      18 |             4 |   8 | dept_role      | top5     |        46 |                 46 | -0.00440941 | -0.0166487 | 0.00902658 |                   -0.000592745 |                 -0.000343478 |                         -0.0167751 |                      -0.0121164  |               0.000249267 |                    0.00465867 |
|      18 |             4 |   8 | team           | top5     |        46 |                 46 | -0.00678004 | -0.0156857 | 0.0021857  |                    0.000213769 |                 -0.00429195  |                         -0.017337  |                      -0.0150627  |              -0.00450572  |                    0.00227432 |
|      18 |             4 |   8 | role           | top5     |        46 |                 46 | -0.00738442 | -0.0196945 | 0.00557934 |                   -0.0007332   |                 -0.00106538  |                         -0.0160119 |                      -0.00895962 |              -0.000332182 |                    0.00705224 |
|      18 |             4 |   8 | project_role   | top5     |        46 |                 46 | -0.00796988 | -0.0198236 | 0.00437916 |                   -0.000467652 |                 -0.00308818  |                         -0.0178288 |                      -0.0124794  |              -0.00262053  |                    0.00534935 |
