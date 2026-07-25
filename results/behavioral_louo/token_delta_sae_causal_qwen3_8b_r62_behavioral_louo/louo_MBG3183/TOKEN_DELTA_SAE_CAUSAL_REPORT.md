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
|      18 |             4 |   8 | project_role   | top5     |             1 |                      1 |                   -0.0105523 |                      -0.0101943 |                       -0.0102735 |                          -0.0106354 |            0.000358045 |               -0.000361919 |                   0.000719965 |
|      18 |             4 |   8 | role           | top5     |             1 |                      1 |                   -0.0105394 |                      -0.0105364 |                       -0.0105128 |                          -0.0107291 |            2.92063e-06 |               -0.000216365 |                   0.000219285 |
|      18 |             4 |   8 | dept_role      | top5     |             1 |                      1 |                   -0.0103925 |                      -0.0109906 |                       -0.0107769 |                          -0.0108429 |           -0.000598073 |               -6.59227e-05 |                  -0.00053215  |
|      18 |             4 |   8 | team           | top5     |             1 |                      0 |                  nan         |                     nan         |                      nan         |                         nan         |          nan           |              nan           |                 nan           |

## Selected Feature Sets

|   layer |   latent_mult |   k | feature_set     |   n_features | feature_ids                         |   mean_row_gap |
|--------:|--------------:|----:|:----------------|-------------:|:------------------------------------|---------------:|
|      18 |             4 |   8 | top5            |            5 | [11574, 866, 109, 4593, 10368]      |    0.0277323   |
|      18 |             4 |   8 | control1        |            1 | [5489]                              |   -5.70497e-05 |
|      18 |             4 |   8 | control5_active |            5 | [11379, 11648, 12880, 16012, 15752] |    1.18293e-05 |

## Example Receiver-Level Best Repairs

|   layer |   latent_mult |   k | context_mode   | feature_set     | donor_type   |   receiver_row_idx |   donor_row_idx | receiver_example_id   | donor_example_id   |   alpha |   base_score |   patched_score |      delta |   n_selected_features |   n_active_receiver_tokens | selected_features                   | repair   | strong_repair   |
|--------:|--------------:|----:|:---------------|:----------------|:-------------|-------------------:|----------------:|:----------------------|:-------------------|--------:|-------------:|----------------:|-----------:|----------------------:|---------------------------:|:------------------------------------|:---------|:----------------|
|      18 |             4 |   8 | dept_role      | control5_active | anomalous    |              83551 |           25403 | MBG3183:283           | CDE1846:455        |    1    |     0.616214 |        0.605371 | -0.0108429 |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | dept_role      | control5_active | benign       |              83551 |           13146 | MBG3183:283           | BGH2610:109        |    0.5  |     0.616214 |        0.605437 | -0.0107769 |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | anomalous    |              83551 |           25380 | MBG3183:283           | CDE1846:422        |    1    |     0.616214 |        0.605224 | -0.0109906 |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | dept_role      | top5            | benign       |              83551 |          105916 | MBG3183:283           | QKO2718:458        |    1    |     0.616214 |        0.605822 | -0.0103925 |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | project_role   | control5_active | anomalous    |              83551 |           25396 | MBG3183:283           | CDE1846:444        |    0.25 |     0.616214 |        0.605579 | -0.0106354 |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | project_role   | control5_active | benign       |              83551 |          138443 | MBG3183:283           | YFT1382:114        |    0.75 |     0.616214 |        0.605941 | -0.0102735 |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | anomalous    |              83551 |           25413 | MBG3183:283           | CDE1846:469        |    1    |     0.616214 |        0.60602  | -0.0101943 |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | project_role   | top5            | benign       |              83551 |           23424 | MBG3183:283           | CCB0547:165        |    1    |     0.616214 |        0.605662 | -0.0105523 |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | role           | control5_active | anomalous    |              83551 |           25383 | MBG3183:283           | CDE1846:427        |    0.75 |     0.616214 |        0.605485 | -0.0107291 |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | role           | control5_active | benign       |              83551 |          106226 | MBG3183:283           | QRR1242:394        |    0.75 |     0.616214 |        0.605701 | -0.0105128 |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | role           | top5            | anomalous    |              83551 |           25402 | MBG3183:283           | CDE1846:452        |    0.75 |     0.616214 |        0.605678 | -0.0105364 |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | role           | top5            | benign       |              83551 |          105978 | MBG3183:283           | QRR1242:35         |    0.75 |     0.616214 |        0.605675 | -0.0105394 |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
|      18 |             4 |   8 | team           | control5_active | benign       |              83551 |           78975 | MBG3183:283           | LLJ3179:224        |    1    |     0.616214 |        0.606175 | -0.0100389 |                     5 |                          4 | [11379, 11648, 12880, 16012, 15752] | True     | True            |
|      18 |             4 |   8 | team           | top5            | benign       |              83551 |           77525 | MBG3183:283           | LGR3119:338        |    0.75 |     0.616214 |        0.60521  | -0.0110037 |                     5 |                          1 | [11574, 866, 109, 4593, 10368]      | True     | True            |
