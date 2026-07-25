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
|      18 |             4 |   8 | role           | top5     |         1 |                  1 |                    -0.0222915  |                 -0.000846505 |                        -0.0220736  |                      -0.002278   |                0.021445   |                    0.0197956  |                   0.00164947  |
|      18 |             4 |   8 | project_role   | top5     |         1 |                  1 |                    -0.0222915  |                  0.00570729  |                        -0.0220736  |                       0.00473633 |                0.0279988  |                    0.0268099  |                   0.00118893  |
|      18 |             4 |   8 | dept_role      | top5     |         1 |                  1 |                    -0.00696391 |                 -0.0166031   |                        -0.00846124 |                      -0.0173357  |               -0.00963914 |                   -0.00887448 |                  -0.000764668 |
|      18 |             4 |   8 | team           | top5     |         1 |                  1 |                    -0.0222915  |                  0.00244865  |                        -0.0220736  |                       0.00383076 |                0.0247402  |                    0.0259043  |                  -0.00116414  |

## Selected Feature Sets

|   layer |   latent_mult |   k | feature_set     |   n_features | feature_ids                         |   mean_row_gap |
|--------:|--------------:|----:|:----------------|-------------:|:------------------------------------|---------------:|
|      18 |             4 |   8 | top5            |            5 | [11574, 866, 109, 4593, 10368]      |    0.0277323   |
|      18 |             4 |   8 | control1        |            1 | [5489]                              |   -5.70497e-05 |
|      18 |             4 |   8 | control5_active |            5 | [11379, 11648, 12880, 16012, 15752] |    1.18293e-05 |

## Example Receiver-Level Best Ablations

|   layer |   latent_mult |   k | context_mode   | feature_set     | receiver_type   |   pair_idx |   receiver_row_idx |   matched_row_idx | receiver_example_id   | matched_example_id   |   alpha |   base_score |   patched_score |        delta |   n_selected_features |   n_active_receiver_tokens | selected_features                   | effect   | strong_effect   |
|--------:|--------------:|----:|:---------------|:----------------|:----------------|-----------:|-------------------:|------------------:|:----------------------|:---------------------|--------:|-------------:|----------------:|-------------:|----------------------:|---------------------------:|:------------------------------------|:---------|:----------------|
|      18 |             4 |   8 | dept_role      | control5_active | benign          |          0 |             105649 |             83551 | QKO2718:72            | MBG3183:283          |    0.25 |     0.716492 |        0.699157 | -0.0173357   |                     5 |                          6 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | dept_role      | control5_active | positive        |          0 |              83551 |            105649 | MBG3183:283           | QKO2718:72           |    1    |     0.616214 |        0.607753 | -0.00846124  |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | False           |
|      18 |             4 |   8 | dept_role      | top5            | benign          |          0 |             105649 |             83551 | QKO2718:72            | MBG3183:283          |    0.5  |     0.716492 |        0.699889 | -0.0166031   |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | positive        |          0 |              83551 |            105649 | MBG3183:283           | QKO2718:72           |    0.75 |     0.616214 |        0.60925  | -0.00696391  |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | False           |
|      18 |             4 |   8 | project_role   | control5_active | benign          |          0 |              23426 |             83551 | CCB0547:169           | MBG3183:283          |    0.75 |     0.28439  |        0.289126 |  0.00473633  |                     5 |                          6 | [11379, 11648, 12880, 16012, 15752] | False    | False           |
|      18 |             4 |   8 | project_role   | control5_active | positive        |          0 |              83551 |             23426 | MBG3183:283           | CCB0547:169          |    0.5  |     0.616214 |        0.594141 | -0.0220736   |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | benign          |          0 |              23426 |             83551 | CCB0547:169           | MBG3183:283          |    0.25 |     0.28439  |        0.290097 |  0.00570729  |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | False    | False           |
|      18 |             4 |   8 | project_role   | top5            | positive        |          0 |              83551 |             23426 | MBG3183:283           | CCB0547:169          |    0.5  |     0.616214 |        0.593923 | -0.0222915   |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | role           | control5_active | benign          |          0 |              11631 |             83551 | AXW3243:511           | MBG3183:283          |    0.5  |     0.395253 |        0.392975 | -0.002278    |                     5 |                          7 | [11379, 11648, 12880, 16012, 15752] | True     | False           |
|      18 |             4 |   8 | role           | control5_active | positive        |          0 |              83551 |             11631 | MBG3183:283           | AXW3243:511          |    0.5  |     0.616214 |        0.594141 | -0.0220736   |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | role           | top5            | benign          |          0 |              11631 |             83551 | AXW3243:511           | MBG3183:283          |    0.75 |     0.395253 |        0.394407 | -0.000846505 |                     5 |                          3 | [11574, 866, 109, 4593, 10368]      | True     | False           |
|      18 |             4 |   8 | role           | top5            | positive        |          0 |              83551 |             11631 | MBG3183:283           | AXW3243:511          |    0.5  |     0.616214 |        0.593923 | -0.0222915   |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | team           | control5_active | benign          |          0 |              11489 |             83551 | AXW3243:337           | MBG3183:283          |    0.25 |     0.428826 |        0.432657 |  0.00383076  |                     5 |                          5 | [11379, 11648, 12880, 16012, 15752] | False    | False           |
|      18 |             4 |   8 | team           | control5_active | positive        |          0 |              83551 |             11489 | MBG3183:283           | AXW3243:337          |    0.5  |     0.616214 |        0.594141 | -0.0220736   |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | team           | top5            | benign          |          0 |              11489 |             83551 | AXW3243:337           | MBG3183:283          |    1    |     0.428826 |        0.431274 |  0.00244865  |                     5 |                          5 | [11574, 866, 109, 4593, 10368]      | False    | False           |
|      18 |             4 |   8 | team           | top5            | positive        |          0 |              83551 |             11489 | MBG3183:283           | AXW3243:337          |    0.5  |     0.616214 |        0.593923 | -0.0222915   |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
