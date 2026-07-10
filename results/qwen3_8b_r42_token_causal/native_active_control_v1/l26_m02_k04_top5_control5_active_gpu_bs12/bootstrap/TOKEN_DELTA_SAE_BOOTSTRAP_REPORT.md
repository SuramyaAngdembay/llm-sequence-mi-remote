# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over positive receivers for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |    estimate |      ci_low |    ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|------------:|------------:|-----------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      26 |             2 |   4 | dept           | top5     |          1309 | 0.00102795  | 0.000780066 | 0.00126993 |                   0.00109736 |                      0.00231283 |                       0.00652978 |                          0.0067173  |             0.00121547 |                0.000187517 |
|      26 |             2 |   4 | team           | top5     |          1309 | 0.00101591  | 0.000751592 | 0.00127709 |                   0.00104436 |                      0.00257994 |                       0.00651674 |                          0.00703641 |             0.00153558 |                0.000519669 |
|      26 |             2 |   4 | role           | top5     |          1309 | 0.000748437 | 0.000476443 | 0.00102058 |                   0.00127589 |                      0.00238922 |                       0.00654716 |                          0.00691206 |             0.00111333 |                0.000364895 |
|      26 |             2 |   4 | dept_role      | top5     |          1309 | 0.000678191 | 0.000409812 | 0.00095073 |                   0.00133881 |                      0.00240868 |                       0.00650728 |                          0.00689896 |             0.00106987 |                0.000391683 |
