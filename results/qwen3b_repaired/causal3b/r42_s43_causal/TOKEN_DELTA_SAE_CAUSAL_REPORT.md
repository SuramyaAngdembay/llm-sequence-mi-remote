# Token Delta SAE Causal Eval

Token-level model patching on adapter deltas at hidden-state layer `24` with SAE config `latent_mult=2, k=4`.

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
    24            2  4         team   top5         1301                  1052                    0.002292                       0.002129                        0.002099                           0.002292             -0.000162                  0.000194                    -0.000356

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                   feature_ids  mean_row_gap
    24            2  4            top5           5 [3800, 2172, 2719, 105, 3333]      0.043802
    24            2  4        control1           1                        [2312]     -0.000303
    24            2  4 control5_active           5 [1807, 841, 3356, 2960, 2216]      0.000005

## Example Receiver-Level Best Repairs

 layer  latent_mult  k context_mode     feature_set donor_type  receiver_row_idx  donor_row_idx receiver_example_id donor_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens             selected_features  repair  strong_repair
    24            2  4         team control5_active  anomalous                14           1593         AAF0535:177      FMG0527:373   0.25    0.388990       0.400647  0.011657                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                15           1593         AAF0535:178      FMG0527:373   0.25    0.404367       0.425075  0.020708                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                16           1593         AAF0535:179      FMG0527:373   0.25    0.426711       0.435144  0.008432                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                17           3362         AAF0535:182      LJR0523:220   0.75    0.505957       0.497093 -0.008864                    5                         6 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active  anomalous                18           1593         AAF0535:183      FMG0527:373   0.75    0.463116       0.446532 -0.016584                    5                         6 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                19           3362         AAF0535:184      LJR0523:220   0.75    0.468176       0.451901 -0.016275                    5                         6 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                20           3362         AAF0535:185      LJR0523:220   0.75    0.463333       0.478567  0.015235                    5                         7 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                21           3359         AAF0535:186      LJR0523:217   0.50    0.506143       0.495033 -0.011110                    5                         6 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                22           1591         AAF0535:190      FMG0527:371   0.25    0.531922       0.510842 -0.021079                    5                         5 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                23           1591         AAF0535:191      FMG0527:371   0.25    0.474848       0.466920 -0.007928                    5                         5 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active  anomalous                24           1591         AAF0535:192      FMG0527:371   0.25    0.463480       0.454881 -0.008599                    5                         6 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active  anomalous                25           3355         AAF0535:193      LJR0523:212   1.00    0.527145       0.497217 -0.029929                    5                         6 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                26           3360         AAF0535:196      LJR0523:218   0.25    0.463414       0.447028 -0.016386                    5                         5 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                27           3360         AAF0535:197      LJR0523:218   0.25    0.410669       0.403288 -0.007381                    5                         4 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active  anomalous                28           1593         AAF0535:198      FMG0527:373   0.75    0.450052       0.438149 -0.011902                    5                         5 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                29           3360         AAF0535:199      LJR0523:218   0.25    0.544827       0.519565 -0.025261                    5                         6 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active  anomalous                30           3360         AAF0535:200      LJR0523:218   0.25    0.524977       0.517624 -0.007353                    5                         7 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active  anomalous                31           3360         AAF0535:203      LJR0523:218   1.00    0.458524       0.473905  0.015380                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                32           3360         AAF0535:204      LJR0523:218   1.00    0.473660       0.484081  0.010421                    5                         7 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                33           1593         AAF0535:205      FMG0527:373   0.75    0.527760       0.564154  0.036394                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                34           3362         AAF0535:206      LJR0523:220   1.00    0.477141       0.506845  0.029704                    5                         7 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                35           3360         AAF0535:207      LJR0523:218   1.00    0.426052       0.458059  0.032007                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                36           3356         AAF0535:210      LJR0523:213   0.75    0.510902       0.530916  0.020014                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                37           3356         AAF0535:211      LJR0523:213   0.75    0.415500       0.447548  0.032048                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                38           3356         AAF0535:212      LJR0523:213   0.75    0.415193       0.446988  0.031796                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                39           3356         AAF0535:213      LJR0523:213   0.75    0.403042       0.438724  0.035683                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                40           3356         AAF0535:214      LJR0523:213   0.75    0.414483       0.448684  0.034201                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                41           3355         AAF0535:217      LJR0523:212   0.75    0.407091       0.414785  0.007694                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                42           3359         AAF0535:218      LJR0523:217   0.50    0.448606       0.488358  0.039752                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                43           3355         AAF0535:219      LJR0523:212   0.75    0.426555       0.433037  0.006482                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                44           3359         AAF0535:220      LJR0523:217   0.50    0.525303       0.534350  0.009046                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                45           3355         AAF0535:221      LJR0523:212   0.75    0.392811       0.401135  0.008323                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                46           3355         AAF0535:224      LJR0523:212   0.50    0.490239       0.532172  0.041933                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                47           3355         AAF0535:225      LJR0523:212   0.50    0.483969       0.524298  0.040329                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                48           3355         AAF0535:226      LJR0523:212   0.50    0.398892       0.407050  0.008159                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                49           3355         AAF0535:227      LJR0523:212   0.50    0.477531       0.507793  0.030262                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                50           3355         AAF0535:228      LJR0523:212   0.50    0.471346       0.492320  0.020974                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                72           1074         AAM0658:294      CQW0652:427   1.00    0.337624       0.360844  0.023219                    5                         2 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                73           1072         AAM0658:295      CQW0652:423   0.25    0.328664       0.352046  0.023382                    5                         3 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active  anomalous                74           1068         AAM0658:296      CQW0652:417   0.75    0.397680       0.437867  0.040187                    5                         3 [1807, 841, 3356, 2960, 2216]   False          False
