# Token Delta SAE Necessity Bootstrap Stats

Bootstrap confidence intervals over complete matched positive/benign receiver-pair contrasts for token-level necessity ablation. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |    estimate |       ci_low |    ci_high |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|------------:|-------------:|-----------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|
|      26 |             2 |   4 | dept_role      | top5     |      1039 |               1039 | 0.00292177  |  0.00145958  | 0.00437872 |                     0.00661923 |                 -0.00483305  |                         0.00999216 |                     -0.00438189  |               -0.0114523  |                   -0.014374   |
|      26 |             2 |   4 | role           | top5     |      1119 |               1119 | 0.00207545  |  0.000678829 | 0.00341792 |                     0.0108196  |                 -0.001657    |                         0.0143868  |                     -0.000165307 |               -0.0124766  |                   -0.0145521  |
|      26 |             2 |   4 | dept           | top5     |      1199 |               1199 | 0.0011549   | -0.000242226 | 0.00253638 |                     0.00693704 |                  4.94119e-05 |                         0.00990238 |                      0.00185986  |               -0.00688762 |                   -0.00804252 |
|      26 |             2 |   4 | team           | top5     |      1301 |               1301 | 0.000661806 | -0.00088006  | 0.00223591 |                     0.00954074 |                 -0.00150441  |                         0.012537   |                      0.000830069 |               -0.0110451  |                   -0.011707   |
