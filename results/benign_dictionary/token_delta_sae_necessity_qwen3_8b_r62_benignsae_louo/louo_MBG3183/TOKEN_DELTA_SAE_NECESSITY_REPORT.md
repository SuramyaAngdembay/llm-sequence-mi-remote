# Token Delta SAE Necessity Eval

Token-level feature ablation on adapter deltas at hidden-state layer `18` with SAE config `latent_mult=4, k=8`.

Intervention protocol:
- receivers = paired positive and matched benign eval examples
- same-user benign matches excluded: `True`
- pairs are matched by requested context mode with fallback to broader benign pools
- feature sets = top sparse sets ablated in token-level delta-SAE space, compared against the control set
- only receiver token positions where the target sparse features are active are modified
- ablation shrinks selected sparse feature activations toward zero by alpha
- summary advantages are paired contrasts over pairs with complete top/control and positive/benign support

Control comparison: `control5_active`

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_pairs |   n_complete_pairs |   top_positive_mean_best_delta |   top_benign_mean_best_delta |   control_positive_mean_best_delta |   control_benign_mean_best_delta |   top_necessity_advantage |   control_necessity_advantage |   top_minus_control_necessity |
|--------:|--------------:|----:|:---------------|:---------|----------:|-------------------:|-------------------------------:|-----------------------------:|-----------------------------------:|---------------------------------:|--------------------------:|------------------------------:|------------------------------:|
|      18 |             4 |   8 | dept_role      | top5     |         1 |                  1 |                     -0.0462323 |                   0.00372714 |                         -0.0293899 |                     -0.0447254   |                 0.0499594 |                    -0.0153354 |                   0.0652949   |
|      18 |             4 |   8 | project_role   | top5     |         1 |                  1 |                     -0.0446733 |                   0.0351305  |                         -0.041666  |                      0.0351093   |                 0.0798038 |                     0.0767753 |                   0.00302854  |
|      18 |             4 |   8 | role           | top5     |         1 |                  1 |                     -0.0446733 |                  -0.00275555 |                         -0.041666  |                      0.000229537 |                 0.0419177 |                     0.0418955 |                   2.22027e-05 |
|      18 |             4 |   8 | team           | top5     |         1 |                  1 |                     -0.0446733 |                   0.0202532  |                         -0.041666  |                      0.0250105   |                 0.0649264 |                     0.0666765 |                  -0.00175002  |

## Selected Feature Sets

|   layer |   latent_mult |   k | feature_set     |   n_features | feature_ids                      |   mean_row_gap |
|--------:|--------------:|----:|:----------------|-------------:|:---------------------------------|---------------:|
|      18 |             4 |   8 | top5            |            5 | [7718, 10473, 7258, 12322, 7913] |    0.103231    |
|      18 |             4 |   8 | control1        |            1 | [12978]                          |    0.000356234 |
|      18 |             4 |   8 | control5_active |            5 | [12514, 6019, 8972, 3112, 6115]  |    1.98879e-05 |

## Example Receiver-Level Best Ablations

|   layer |   latent_mult |   k | context_mode   | feature_set     | receiver_type   |   pair_idx |   receiver_row_idx |   matched_row_idx | receiver_example_id   | matched_example_id   |   alpha |   base_score |   patched_score |        delta |   n_selected_features |   n_active_receiver_tokens | selected_features                | effect   | strong_effect   |
|--------:|--------------:|----:|:---------------|:----------------|:----------------|-----------:|-------------------:|------------------:|:----------------------|:---------------------|--------:|-------------:|----------------:|-------------:|----------------------:|---------------------------:|:---------------------------------|:---------|:----------------|
|      18 |             4 |   8 | dept_role      | control5_active | benign          |          0 |             105649 |             83551 | QKO2718:72            | MBG3183:283          |    0.75 |     0.716492 |        0.671767 | -0.0447254   |                     5 |                          6 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | dept_role      | control5_active | positive        |          0 |              83551 |            105649 | MBG3183:283           | QKO2718:72           |    0.5  |     0.616214 |        0.586824 | -0.0293899   |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | benign          |          0 |             105649 |             83551 | QKO2718:72            | MBG3183:283          |    0.25 |     0.716492 |        0.720219 |  0.00372714  |                     5 |                          0 | [7718, 10473, 7258, 12322, 7913] | False    | False           |
|      18 |             4 |   8 | dept_role      | top5            | positive        |          0 |              83551 |            105649 | MBG3183:283           | QKO2718:72           |    0.5  |     0.616214 |        0.569982 | -0.0462323   |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | project_role   | control5_active | benign          |          0 |              23426 |             83551 | CCB0547:169           | MBG3183:283          |    0.25 |     0.28439  |        0.319499 |  0.0351093   |                     5 |                          4 | [12514, 6019, 8972, 3112, 6115]  | False    | False           |
|      18 |             4 |   8 | project_role   | control5_active | positive        |          0 |              83551 |             23426 | MBG3183:283           | CCB0547:169          |    0.5  |     0.616214 |        0.574548 | -0.041666    |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | benign          |          0 |              23426 |             83551 | CCB0547:169           | MBG3183:283          |    0.75 |     0.28439  |        0.31952  |  0.0351305   |                     5 |                          5 | [7718, 10473, 7258, 12322, 7913] | False    | False           |
|      18 |             4 |   8 | project_role   | top5            | positive        |          0 |              83551 |             23426 | MBG3183:283           | CCB0547:169          |    0.75 |     0.616214 |        0.571541 | -0.0446733   |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | role           | control5_active | benign          |          0 |              11631 |             83551 | AXW3243:511           | MBG3183:283          |    0.5  |     0.395253 |        0.395483 |  0.000229537 |                     5 |                          4 | [12514, 6019, 8972, 3112, 6115]  | False    | False           |
|      18 |             4 |   8 | role           | control5_active | positive        |          0 |              83551 |             11631 | MBG3183:283           | AXW3243:511          |    0.5  |     0.616214 |        0.574548 | -0.041666    |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | role           | top5            | benign          |          0 |              11631 |             83551 | AXW3243:511           | MBG3183:283          |    0.75 |     0.395253 |        0.392498 | -0.00275555  |                     5 |                          5 | [7718, 10473, 7258, 12322, 7913] | True     | False           |
|      18 |             4 |   8 | role           | top5            | positive        |          0 |              83551 |             11631 | MBG3183:283           | AXW3243:511          |    0.75 |     0.616214 |        0.571541 | -0.0446733   |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | team           | control5_active | benign          |          0 |              11489 |             83551 | AXW3243:337           | MBG3183:283          |    0.25 |     0.428826 |        0.453836 |  0.0250105   |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | False    | False           |
|      18 |             4 |   8 | team           | control5_active | positive        |          0 |              83551 |             11489 | MBG3183:283           | AXW3243:337          |    0.5  |     0.616214 |        0.574548 | -0.041666    |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | team           | top5            | benign          |          0 |              11489 |             83551 | AXW3243:337           | MBG3183:283          |    0.25 |     0.428826 |        0.449079 |  0.0202532   |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | False    | False           |
|      18 |             4 |   8 | team           | top5            | positive        |          0 |              83551 |             11489 | MBG3183:283           | AXW3243:337          |    0.75 |     0.616214 |        0.571541 | -0.0446733   |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
