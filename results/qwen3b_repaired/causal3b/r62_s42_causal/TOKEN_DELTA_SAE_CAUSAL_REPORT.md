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
    24            4  8         role   top5           70                    70                   -0.003612                       -0.00258                       -0.001846                          -0.000947              0.001032                  0.000899                     0.000133

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                    feature_ids  mean_row_gap
    24            4  8            top5           5 [1900, 2642, 2146, 4976, 5627]      0.129575
    24            4  8        control1           1                         [2785]     -0.000983
    24            4  8 control5_active           5  [5846, 1978, 6452, 494, 1613]     -0.000006

## Example Receiver-Level Best Repairs

 layer  latent_mult  k context_mode     feature_set donor_type  receiver_row_idx  donor_row_idx receiver_example_id donor_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens             selected_features  repair  strong_repair
    24            4  8         role control5_active  anomalous                71           1523         ACM2278:228      CMP2946:410   1.00    0.324349       0.305344 -0.019004                    5                         6 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous                72           1523         ACM2278:231      CMP2946:410   1.00    0.303915       0.305943  0.002028                    5                         2 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous                73           1523         ACM2278:232      CMP2946:410   1.00    0.283694       0.282094 -0.001600                    5                         2 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous                74           1516         ACM2278:233      CMP2946:401   1.00    0.377288       0.387775  0.010487                    5                         4 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous                75           1527         ACM2278:234      CMP2946:416   1.00    0.372281       0.377031  0.004750                    5                         4 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1255           4190         CDE1846:415      MBG3183:283   1.00    0.322910       0.310219 -0.012691                    5                        18 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1256           4190         CDE1846:416      MBG3183:283   0.25    0.326972       0.316226 -0.010746                    5                         8 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1257           4190         CDE1846:417      MBG3183:283   0.50    0.283497       0.267135 -0.016362                    5                        20 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1258           4190         CDE1846:420      MBG3183:283   0.75    0.314484       0.333972  0.019489                    5                        12 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1259           4190         CDE1846:421      MBG3183:283   0.25    0.261480       0.288326  0.026846                    5                        11 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1260           4190         CDE1846:422      MBG3183:283   0.50    0.320619       0.331144  0.010524                    5                         4 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1261           4190         CDE1846:423      MBG3183:283   0.50    0.302170       0.329105  0.026936                    5                        14 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1262           4190         CDE1846:424      MBG3183:283   0.50    0.307945       0.336379  0.028434                    5                         7 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1263           4190         CDE1846:427      MBG3183:283   0.75    0.296463       0.283732 -0.012731                    5                        11 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1264           4190         CDE1846:428      MBG3183:283   0.75    0.327899       0.315816 -0.012082                    5                        11 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1265           4190         CDE1846:429      MBG3183:283   0.25    0.323417       0.308411 -0.015006                    5                         5 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1266           4190         CDE1846:430      MBG3183:283   0.75    0.285782       0.274279 -0.011503                    5                        15 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1267           4190         CDE1846:431      MBG3183:283   0.25    0.312516       0.303071 -0.009444                    5                         8 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1268           4190         CDE1846:434      MBG3183:283   0.75    0.266757       0.255954 -0.010803                    5                        12 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1269           4190         CDE1846:435      MBG3183:283   0.75    0.334333       0.323608 -0.010726                    5                         8 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1270           4190         CDE1846:436      MBG3183:283   1.00    0.342055       0.332301 -0.009753                    5                         8 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1271           4190         CDE1846:437      MBG3183:283   1.00    0.324391       0.311287 -0.013103                    5                        11 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1272           4190         CDE1846:438      MBG3183:283   1.00    0.292802       0.277459 -0.015343                    5                         9 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1273           4190         CDE1846:441      MBG3183:283   0.50    0.346164       0.333224 -0.012940                    5                         7 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1274           4190         CDE1846:442      MBG3183:283   0.50    0.271960       0.258569 -0.013391                    5                        13 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1275           4190         CDE1846:443      MBG3183:283   1.00    0.332389       0.319647 -0.012742                    5                         5 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1276           4190         CDE1846:444      MBG3183:283   0.25    0.320913       0.328158  0.007245                    5                        18 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1277           4190         CDE1846:445      MBG3183:283   0.50    0.278518       0.263869 -0.014649                    5                        10 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1278           4190         CDE1846:448      MBG3183:283   0.25    0.325859       0.316203 -0.009656                    5                         8 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1279           4190         CDE1846:449      MBG3183:283   1.00    0.399746       0.421597  0.021851                    5                         8 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1280           4190         CDE1846:450      MBG3183:283   0.75    0.333364       0.324771 -0.008593                    5                        20 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1281           4190         CDE1846:451      MBG3183:283   0.75    0.314781       0.313665 -0.001116                    5                         5 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1282           4190         CDE1846:452      MBG3183:283   1.00    0.326137       0.322816 -0.003321                    5                         5 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1283           4190         CDE1846:455      MBG3183:283   1.00    0.284086       0.273522 -0.010564                    5                        18 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1284           4190         CDE1846:456      MBG3183:283   1.00    0.322613       0.325973  0.003360                    5                        20 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active  anomalous              1285           4190         CDE1846:457      MBG3183:283   1.00    0.302524       0.292337 -0.010186                    5                        13 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1286           4190         CDE1846:458      MBG3183:283   1.00    0.318868       0.308915 -0.009952                    5                         8 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1287           4190         CDE1846:459      MBG3183:283   1.00    0.344559       0.334022 -0.010537                    5                        15 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active  anomalous              1288           4190         CDE1846:462      MBG3183:283   0.50    0.292956       0.287765 -0.005191                    5                        17 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active  anomalous              1289           4190         CDE1846:463      MBG3183:283   0.75    0.309511       0.302811 -0.006700                    5                        11 [5846, 1978, 6452, 494, 1613]    True          False
