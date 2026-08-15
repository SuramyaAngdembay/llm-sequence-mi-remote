# Token Delta SAE Causal Eval

Token-level model patching on adapter deltas at hidden-state layer `24` with SAE config `latent_mult=4, k=8`.

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

 layer  latent_mult  k context_mode target  n_receivers  n_complete_receivers  top_benign_mean_best_delta  top_anomalous_mean_best_delta  control_benign_mean_best_delta  control_anomalous_mean_best_delta  top_repair_advantage  control_repair_advantage  top_minus_control_advantage
    24            4  8         role   top5           70                    70                   -0.026068                       -0.00668                       -0.013651                          -0.011732              0.019388                   0.00192                     0.017468

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                   feature_ids  mean_row_gap
    24            4  8            top5           5 [668, 1657, 4773, 2894, 6026]      0.073916
    24            4  8        control1           1                         [748]      0.002688
    24            4  8 control5_active           5  [3100, 6390, 6572, 943, 266]     -0.000010

## Example Receiver-Level Best Repairs

 layer  latent_mult  k context_mode     feature_set donor_type  receiver_row_idx  donor_row_idx receiver_example_id donor_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens            selected_features  repair  strong_repair
    24            4  8         role control5_active  anomalous                71           1521         ACM2278:228      CMP2946:408   1.00    0.318027       0.336269  0.018242                    5                         6 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous                72           1524         ACM2278:231      CMP2946:413   0.50    0.286483       0.310622  0.024139                    5                         6 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous                73           1522         ACM2278:232      CMP2946:409   0.25    0.263782       0.284369  0.020587                    5                         5 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous                74           1526         ACM2278:233      CMP2946:415   0.75    0.351030       0.392102  0.041072                    5                         2 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous                75           1516         ACM2278:234      CMP2946:401   0.50    0.342926       0.393423  0.050496                    5                         3 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous              1255           4190         CDE1846:415      MBG3183:283   0.25    0.460968       0.438966 -0.022001                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1256           4190         CDE1846:416      MBG3183:283   0.25    0.473238       0.451216 -0.022022                    5                         5 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1257           4190         CDE1846:417      MBG3183:283   0.75    0.415135       0.392318 -0.022818                    5                         5 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1258           4190         CDE1846:420      MBG3183:283   0.25    0.411237       0.398100 -0.013137                    5                         6 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1259           4190         CDE1846:421      MBG3183:283   1.00    0.413176       0.367993 -0.045183                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1260           4190         CDE1846:422      MBG3183:283   0.75    0.460541       0.424920 -0.035622                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1261           4190         CDE1846:423      MBG3183:283   0.25    0.445186       0.400925 -0.044261                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1262           4190         CDE1846:424      MBG3183:283   0.75    0.456350       0.407174 -0.049175                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1263           4190         CDE1846:427      MBG3183:283   0.75    0.449075       0.427113 -0.021961                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1264           4190         CDE1846:428      MBG3183:283   0.25    0.454062       0.435785 -0.018276                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1265           4190         CDE1846:429      MBG3183:283   0.25    0.468758       0.452256 -0.016502                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1266           4190         CDE1846:430      MBG3183:283   0.75    0.428227       0.407019 -0.021208                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1267           4190         CDE1846:431      MBG3183:283   0.75    0.451290       0.435340 -0.015950                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1268           4190         CDE1846:434      MBG3183:283   0.25    0.404954       0.381453 -0.023501                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1269           4190         CDE1846:435      MBG3183:283   0.25    0.472196       0.443665 -0.028531                    5                         5 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1270           4190         CDE1846:436      MBG3183:283   0.25    0.457468       0.446610 -0.010858                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1271           4190         CDE1846:437      MBG3183:283   0.25    0.465149       0.441472 -0.023678                    5                         6 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1272           4190         CDE1846:438      MBG3183:283   0.75    0.428764       0.405574 -0.023190                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1273           4190         CDE1846:441      MBG3183:283   0.50    0.485904       0.458601 -0.027302                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1274           4190         CDE1846:442      MBG3183:283   0.50    0.414737       0.386828 -0.027909                    5                         5 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1275           4190         CDE1846:443      MBG3183:283   0.50    0.471049       0.440034 -0.031015                    5                         5 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1276           4190         CDE1846:444      MBG3183:283   0.75    0.392520       0.419514  0.026994                    5                         8 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous              1277           4190         CDE1846:445      MBG3183:283   0.50    0.414716       0.388324 -0.026392                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1278           4190         CDE1846:448      MBG3183:283   0.50    0.465886       0.434160 -0.031725                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1279           4190         CDE1846:449      MBG3183:283   0.50    0.499757       0.515431  0.015674                    5                         9 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous              1280           4190         CDE1846:450      MBG3183:283   0.50    0.462346       0.429670 -0.032676                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1281           4190         CDE1846:451      MBG3183:283   1.00    0.457420       0.427046 -0.030374                    5                         6 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1282           4190         CDE1846:452      MBG3183:283   1.00    0.464426       0.431741 -0.032685                    5                         2 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1283           4190         CDE1846:455      MBG3183:283   0.75    0.416762       0.394541 -0.022221                    5                         2 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1284           4190         CDE1846:456      MBG3183:283   1.00    0.383490       0.431062  0.047571                    5                        10 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active  anomalous              1285           4190         CDE1846:457      MBG3183:283   1.00    0.440333       0.425539 -0.014795                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1286           4190         CDE1846:458      MBG3183:283   0.75    0.454909       0.435510 -0.019399                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1287           4190         CDE1846:459      MBG3183:283   0.50    0.481454       0.457132 -0.024322                    5                         2 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1288           4190         CDE1846:462      MBG3183:283   1.00    0.443602       0.411209 -0.032393                    5                         6 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active  anomalous              1289           4190         CDE1846:463      MBG3183:283   0.75    0.453309       0.415600 -0.037709                    5                         5 [3100, 6390, 6572, 943, 266]    True           True
