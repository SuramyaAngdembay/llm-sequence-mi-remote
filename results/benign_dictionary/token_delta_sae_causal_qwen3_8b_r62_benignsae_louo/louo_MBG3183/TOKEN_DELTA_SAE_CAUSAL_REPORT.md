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
|      18 |             4 |   8 | role           | top5     |             1 |                      1 |                   -0.0564504 |                      -0.0564465 |                       -0.0351064 |                          -0.0354859 |            3.93391e-06 |               -0.000379503 |                   0.000383437 |
|      18 |             4 |   8 | project_role   | top5     |             1 |                      1 |                   -0.0513652 |                      -0.0562546 |                       -0.0357449 |                          -0.0354859 |           -0.00488937  |                0.000258982 |                  -0.00514835  |
|      18 |             4 |   8 | dept_role      | top5     |             1 |                      1 |                   -0.0455716 |                      -0.0572532 |                       -0.0349768 |                          -0.0354859 |           -0.0116816   |               -0.000509083 |                  -0.0111725   |
|      18 |             4 |   8 | team           | top5     |             1 |                      0 |                  nan         |                     nan         |                      nan         |                         nan         |          nan           |              nan           |                 nan           |

## Selected Feature Sets

|   layer |   latent_mult |   k | feature_set     |   n_features | feature_ids                      |   mean_row_gap |
|--------:|--------------:|----:|:----------------|-------------:|:---------------------------------|---------------:|
|      18 |             4 |   8 | top5            |            5 | [7718, 10473, 7258, 12322, 7913] |    0.103231    |
|      18 |             4 |   8 | control1        |            1 | [12978]                          |    0.000356234 |
|      18 |             4 |   8 | control5_active |            5 | [12514, 6019, 8972, 3112, 6115]  |    1.98879e-05 |

## Example Receiver-Level Best Repairs

|   layer |   latent_mult |   k | context_mode   | feature_set     | donor_type   |   receiver_row_idx |   donor_row_idx | receiver_example_id   | donor_example_id   |   alpha |   base_score |   patched_score |      delta |   n_selected_features |   n_active_receiver_tokens | selected_features                | repair   | strong_repair   |
|--------:|--------------:|----:|:---------------|:----------------|:-------------|-------------------:|----------------:|:----------------------|:-------------------|--------:|-------------:|----------------:|-----------:|----------------------:|---------------------------:|:---------------------------------|:---------|:----------------|
|      18 |             4 |   8 | dept_role      | control5_active | anomalous    |              83551 |           25414 | MBG3183:283           | CDE1846:470        |    1    |     0.616214 |        0.580728 | -0.0354859 |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | dept_role      | control5_active | benign       |              83551 |           84984 | MBG3183:283           | MCO3164:296        |    1    |     0.616214 |        0.581237 | -0.0349768 |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | anomalous    |              83551 |           25404 | MBG3183:283           | CDE1846:456        |    1    |     0.616214 |        0.558961 | -0.0572532 |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | benign       |              83551 |           84850 | MBG3183:283           | MCO3164:105        |    0.75 |     0.616214 |        0.570643 | -0.0455716 |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | project_role   | control5_active | anomalous    |              83551 |           25414 | MBG3183:283           | CDE1846:470        |    1    |     0.616214 |        0.580728 | -0.0354859 |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | project_role   | control5_active | benign       |              83551 |          101818 | MBG3183:283           | PAA2738:29         |    0.5  |     0.616214 |        0.580469 | -0.0357449 |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | anomalous    |              83551 |           25401 | MBG3183:283           | CDE1846:451        |    1    |     0.616214 |        0.55996  | -0.0562546 |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | benign       |              83551 |          135633 | MBG3183:283           | WRF3534:107        |    0.75 |     0.616214 |        0.564849 | -0.0513652 |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | role           | control5_active | anomalous    |              83551 |           25414 | MBG3183:283           | CDE1846:470        |    1    |     0.616214 |        0.580728 | -0.0354859 |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | role           | control5_active | benign       |              83551 |           69094 | MBG3183:283           | KCM2495:297        |    0.25 |     0.616214 |        0.581108 | -0.0351064 |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | role           | top5            | anomalous    |              83551 |           25383 | MBG3183:283           | CDE1846:427        |    1    |     0.616214 |        0.559768 | -0.0564465 |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | role           | top5            | benign       |              83551 |           25292 | MBG3183:283           | CDE1846:291        |    1    |     0.616214 |        0.559764 | -0.0564504 |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
|      18 |             4 |   8 | team           | control5_active | benign       |              83551 |          112559 | MBG3183:283           | ROA3171:29         |    0.5  |     0.616214 |        0.581035 | -0.0351788 |                     5 |                          3 | [12514, 6019, 8972, 3112, 6115]  | True     | True            |
|      18 |             4 |   8 | team           | top5            | benign       |              83551 |          112695 | MBG3183:283           | ROA3171:224        |    0.75 |     0.616214 |        0.565129 | -0.0510852 |                     5 |                          2 | [7718, 10473, 7258, 12322, 7913] | True     | True            |
