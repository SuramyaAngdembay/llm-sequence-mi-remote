# Token Delta SAE Bootstrap Stats

Bootstrap confidence intervals over positive receivers for token-level causal patching. Control comparison: `control5_active`. Bootstrap draws: `4000`.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |     estimate |       ci_low |      ci_high |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-------------:|-------------:|-------------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|
|      18 |             4 |   8 | role           | top5     |            70 |   0.0068477  |   0.00336181 |   0.0107899  |                   -0.067023  |                      -0.0583984 |                      -0.010173   |                         -0.0083961  |             0.00862462 |                 0.00177693 |
|      18 |             4 |   8 | dept_role      | top5     |            70 |   0.0068181  |   0.003541   |   0.0103211  |                   -0.0672285 |                      -0.0583984 |                      -0.0104082  |                         -0.0083961  |             0.00883015 |                 0.00201205 |
|      18 |             4 |   8 | project_role   | top5     |            70 |   0.00420063 |   0.00156509 |   0.00697916 |                   -0.0645022 |                      -0.0583857 |                      -0.0103068  |                         -0.00839091 |             0.00611649 |                 0.00191586 |
|      18 |             4 |   8 | team           | top5     |            70 | nan          | nan          | nan          |                   -0.0494458 |                     nan         |                      -0.00987534 |                        nan          |           nan          |               nan          |
