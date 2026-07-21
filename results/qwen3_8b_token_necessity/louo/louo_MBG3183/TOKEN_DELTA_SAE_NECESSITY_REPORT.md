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
|      18 |             4 |   8 | team           | top5     |         1 |                  1 |                     0.00419432 |                   0.00238666 |                         -0.0174394 |                       0.00306883 |               -0.00180766 |                    0.0205082  |                    -0.0223159 |
|      18 |             4 |   8 | role           | top5     |         1 |                  1 |                     0.00419432 |                  -0.00442818 |                         -0.0174394 |                      -0.0016174  |               -0.0086225  |                    0.015822   |                    -0.0244445 |
|      18 |             4 |   8 | dept_role      | top5     |         1 |                  1 |                     0.0188718  |                  -0.015945   |                         -0.0074181 |                      -0.0146835  |               -0.0348168  |                   -0.00726545 |                    -0.0275514 |
|      18 |             4 |   8 | project_role   | top5     |         1 |                  1 |                     0.00419432 |                  -0.00123164 |                         -0.0174394 |                       0.00511244 |               -0.00542596 |                    0.0225518  |                    -0.0279778 |

## Selected Feature Sets

|   layer |   latent_mult |   k | feature_set     |   n_features | feature_ids                        |   mean_row_gap |
|--------:|--------------:|----:|:----------------|-------------:|:-----------------------------------|---------------:|
|      18 |             4 |   8 | top5            |            5 | [14358, 12848, 4196, 13580, 11292] |    0.100151    |
|      18 |             4 |   8 | control1        |            1 | [2044]                             |    0.000110351 |
|      18 |             4 |   8 | control5_active |            5 | [4204, 4087, 1870, 1888, 13633]    |    3.9645e-06  |

## Example Receiver-Level Best Ablations

|   layer |   latent_mult |   k | context_mode   | feature_set     | receiver_type   |   pair_idx |   receiver_row_idx |   matched_row_idx | receiver_example_id   | matched_example_id   |   alpha |   base_score |   patched_score |       delta |   n_selected_features |   n_active_receiver_tokens | selected_features                  | effect   | strong_effect   |
|--------:|--------------:|----:|:---------------|:----------------|:----------------|-----------:|-------------------:|------------------:|:----------------------|:---------------------|--------:|-------------:|----------------:|------------:|----------------------:|---------------------------:|:-----------------------------------|:---------|:----------------|
|      18 |             4 |   8 | dept_role      | control5_active | benign          |          0 |             105649 |             83551 | QKO2718:72            | MBG3183:283          |    0.25 |     0.716492 |        0.701809 | -0.0146835  |                     5 |                          7 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | dept_role      | control5_active | positive        |          0 |              83551 |            105649 | MBG3183:283           | QKO2718:72           |    0.75 |     0.616214 |        0.608796 | -0.0074181  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | False           |
|      18 |             4 |   8 | dept_role      | top5            | benign          |          0 |             105649 |             83551 | QKO2718:72            | MBG3183:283          |    1    |     0.716492 |        0.700547 | -0.015945   |                     5 |                          1 | [14358, 12848, 4196, 13580, 11292] | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | positive        |          0 |              83551 |            105649 | MBG3183:283           | QKO2718:72           |    0.25 |     0.616214 |        0.635086 |  0.0188718  |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | False    | False           |
|      18 |             4 |   8 | project_role   | control5_active | benign          |          0 |              23426 |             83551 | CCB0547:169           | MBG3183:283          |    0.75 |     0.28439  |        0.289502 |  0.00511244 |                     5 |                         20 | [4204, 4087, 1870, 1888, 13633]    | False    | False           |
|      18 |             4 |   8 | project_role   | control5_active | positive        |          0 |              83551 |             23426 | MBG3183:283           | CCB0547:169          |    0.5  |     0.616214 |        0.598775 | -0.0174394  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | benign          |          0 |              23426 |             83551 | CCB0547:169           | MBG3183:283          |    0.25 |     0.28439  |        0.283158 | -0.00123164 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | project_role   | top5            | positive        |          0 |              83551 |             23426 | MBG3183:283           | CCB0547:169          |    0.25 |     0.616214 |        0.620408 |  0.00419432 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | False    | False           |
|      18 |             4 |   8 | role           | control5_active | benign          |          0 |              11631 |             83551 | AXW3243:511           | MBG3183:283          |    0.5  |     0.395253 |        0.393636 | -0.0016174  |                     5 |                         13 | [4204, 4087, 1870, 1888, 13633]    | True     | False           |
|      18 |             4 |   8 | role           | control5_active | positive        |          0 |              83551 |             11631 | MBG3183:283           | AXW3243:511          |    0.5  |     0.616214 |        0.598775 | -0.0174394  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | role           | top5            | benign          |          0 |              11631 |             83551 | AXW3243:511           | MBG3183:283          |    0.25 |     0.395253 |        0.390825 | -0.00442818 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | role           | top5            | positive        |          0 |              83551 |             11631 | MBG3183:283           | AXW3243:511          |    0.25 |     0.616214 |        0.620408 |  0.00419432 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | False    | False           |
|      18 |             4 |   8 | team           | control5_active | benign          |          0 |              11489 |             83551 | AXW3243:337           | MBG3183:283          |    0.25 |     0.428826 |        0.431895 |  0.00306883 |                     5 |                         10 | [4204, 4087, 1870, 1888, 13633]    | False    | False           |
|      18 |             4 |   8 | team           | control5_active | positive        |          0 |              83551 |             11489 | MBG3183:283           | AXW3243:337          |    0.5  |     0.616214 |        0.598775 | -0.0174394  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | team           | top5            | benign          |          0 |              11489 |             83551 | AXW3243:337           | MBG3183:283          |    0.75 |     0.428826 |        0.431212 |  0.00238666 |                     5 |                          1 | [14358, 12848, 4196, 13580, 11292] | False    | False           |
|      18 |             4 |   8 | team           | top5            | positive        |          0 |              83551 |             11489 | MBG3183:283           | AXW3243:337          |    0.25 |     0.616214 |        0.620408 |  0.00419432 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | False    | False           |
