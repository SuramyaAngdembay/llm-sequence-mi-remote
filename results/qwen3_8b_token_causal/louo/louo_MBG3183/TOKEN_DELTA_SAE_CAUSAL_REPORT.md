# Token Delta SAE Causal Eval

Token-level model patching on adapter deltas at hidden-state layer `18` with SAE config `latent_mult=4, k=8`.

Intervention protocol:
- receivers = positive eval examples only
- donors = matched benign donors and same-class anomalous donor controls
- same-user donors excluded: `True`
- feature sets = top sparse sets patched in token-level delta-SAE space, compared against the control set
- only receiver token positions where the target sparse features are active are patched
- patched token deltas move toward a donor token-feature prototype rather than a uniform sequence-wide broadcast
- summary advantages are paired receiver-level contrasts over receivers with complete top/control and benign/anomalous donor support

Control comparison: `control5_active`

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_receivers |   n_complete_receivers |   top_benign_mean_best_delta |   top_anomalous_mean_best_delta |   control_benign_mean_best_delta |   control_anomalous_mean_best_delta |   top_repair_advantage |   control_repair_advantage |   top_minus_control_advantage |
|--------:|--------------:|----:|:---------------|:---------|--------------:|-----------------------:|-----------------------------:|--------------------------------:|---------------------------------:|------------------------------------:|-----------------------:|---------------------------:|------------------------------:|
|      18 |             4 |   8 | project_role   | top5     |             1 |                      1 |                  -0.00153255 |                     -0.00153255 |                       -0.0190445 |                          -0.0212032 |                      0 |               -0.00215876  |                   0.00215876  |
|      18 |             4 |   8 | dept_role      | top5     |             1 |                      1 |                  -0.00153255 |                     -0.00153255 |                       -0.0227499 |                          -0.0234234 |                      0 |               -0.000673532 |                   0.000673532 |
|      18 |             4 |   8 | role           | top5     |             1 |                      1 |                  -0.00153255 |                     -0.00153255 |                       -0.0277786 |                          -0.0238526 |                      0 |                0.00392598  |                  -0.00392598  |
|      18 |             4 |   8 | team           | top5     |             1 |                      0 |                 nan          |                    nan          |                      nan         |                         nan         |                    nan |              nan           |                 nan           |

## Selected Feature Sets

|   layer |   latent_mult |   k | feature_set     |   n_features | feature_ids                        |   mean_row_gap |
|--------:|--------------:|----:|:----------------|-------------:|:-----------------------------------|---------------:|
|      18 |             4 |   8 | top5            |            5 | [14358, 12848, 4196, 13580, 11292] |    0.100151    |
|      18 |             4 |   8 | control1        |            1 | [2044]                             |    0.000110351 |
|      18 |             4 |   8 | control5_active |            5 | [4204, 4087, 1870, 1888, 13633]    |    3.9645e-06  |

## Example Receiver-Level Best Repairs

|   layer |   latent_mult |   k | context_mode   | feature_set     | donor_type   |   receiver_row_idx |   donor_row_idx | receiver_example_id   | donor_example_id   |   alpha |   base_score |   patched_score |       delta |   n_selected_features |   n_active_receiver_tokens | selected_features                  | repair   | strong_repair   |
|--------:|--------------:|----:|:---------------|:----------------|:-------------|-------------------:|----------------:|:----------------------|:-------------------|--------:|-------------:|----------------:|------------:|----------------------:|---------------------------:|:-----------------------------------|:---------|:----------------|
|      18 |             4 |   8 | dept_role      | control5_active | anomalous    |              83551 |           25403 | MBG3183:283           | CDE1846:455        |    1    |     0.616214 |        0.592791 | -0.0234234  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | dept_role      | control5_active | benign       |              83551 |          105639 | MBG3183:283           | QKO2718:58         |    1    |     0.616214 |        0.593464 | -0.0227499  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | anomalous    |              83551 |           25380 | MBG3183:283           | CDE1846:422        |    0.25 |     0.616214 |        0.614682 | -0.00153255 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | dept_role      | top5            | benign       |              83551 |            4546 | MBG3183:283           | AFP2709:310        |    0.25 |     0.616214 |        0.614682 | -0.00153255 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | project_role   | control5_active | anomalous    |              83551 |           25382 | MBG3183:283           | CDE1846:424        |    1    |     0.616214 |        0.595011 | -0.0212032  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | project_role   | control5_active | benign       |              83551 |           80176 | MBG3183:283           | LRC3160:140        |    0.75 |     0.616214 |        0.59717  | -0.0190445  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | anomalous    |              83551 |           25377 | MBG3183:283           | CDE1846:417        |    0.25 |     0.616214 |        0.614682 | -0.00153255 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | project_role   | top5            | benign       |              83551 |           11407 | MBG3183:283           | AXW3243:239        |    0.25 |     0.616214 |        0.614682 | -0.00153255 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | role           | control5_active | anomalous    |              83551 |           25394 | MBG3183:283           | CDE1846:442        |    0.75 |     0.616214 |        0.592362 | -0.0238526  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | role           | control5_active | benign       |              83551 |           69094 | MBG3183:283           | KCM2495:297        |    1    |     0.616214 |        0.588436 | -0.0277786  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | role           | top5            | anomalous    |              83551 |           25375 | MBG3183:283           | CDE1846:415        |    0.25 |     0.616214 |        0.614682 | -0.00153255 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | role           | top5            | benign       |              83551 |            4260 | MBG3183:283           | AFP2709:11         |    0.25 |     0.616214 |        0.614682 | -0.00153255 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
|      18 |             4 |   8 | team           | control5_active | benign       |              83551 |          126684 | MBG3183:283           | TRW3172:403        |    1    |     0.616214 |        0.593943 | -0.0222712  |                     5 |                          8 | [4204, 4087, 1870, 1888, 13633]    | True     | True            |
|      18 |             4 |   8 | team           | top5            | benign       |              83551 |           11478 | MBG3183:283           | AXW3243:323        |    0.25 |     0.616214 |        0.614682 | -0.00153255 |                     5 |                          0 | [14358, 12848, 4196, 13580, 11292] | True     | False           |
