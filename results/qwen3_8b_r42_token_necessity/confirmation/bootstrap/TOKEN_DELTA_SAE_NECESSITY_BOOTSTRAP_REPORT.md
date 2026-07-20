# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |    estimate |       ci_low |    ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|------------:|-------------:|-----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      26 |             2 |   4 | dept_role      | top5     |       528 |                528 |  0.00205852 |  0.000298645 | 0.003834   |                     0.00959727 |                   0.00253702 |                         0.0108591  |                       0.00174031 |               -0.00706025 |                  -0.00911877  |
|      26 |             2 |   4 | dept           | top5     |       605 |                605 |  0.00145017 |  5.89876e-05 | 0.00291776 |                     0.00962277 |                   0.00258489 |                         0.0107091  |                       0.002221   |               -0.00703789 |                  -0.00848806  |
|      26 |             2 |   4 | role           | top5     |       528 |                528 |  0.00132729 | -0.00119     | 0.00374805 |                     0.00885373 |                   0.00702799 |                         0.0101256  |                       0.00697259 |               -0.00182574 |                  -0.00315303  |
|      26 |             2 |   4 | team           | top5     |       638 |                638 | -0.00133559 | -0.00354588  | 0.00060988 |                     0.00728491 |                   0.00530407 |                         0.00822007 |                       0.00757483 |               -0.00198083 |                  -0.000645239 |
