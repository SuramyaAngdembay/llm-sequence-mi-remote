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
    24            2  4         team   top5         1301                  1052                   -0.016743                      -0.017167                       -0.018112                          -0.018113             -0.000424                 -0.000001                    -0.000423

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                    feature_ids  mean_row_gap
    24            2  4            top5           5  [819, 2668, 3412, 3286, 3659]      0.043946
    24            2  4        control1           1                         [3522]      0.010515
    24            2  4 control5_active           5 [3221, 1358, 2852, 1584, 1834]     -0.000006

## Example Receiver-Level Best Repairs

 layer  latent_mult  k context_mode     feature_set donor_type  receiver_row_idx  donor_row_idx receiver_example_id donor_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens              selected_features  repair  strong_repair
    24            2  4         team control5_active  anomalous                14           3354         AAF0535:177      LJR0523:211   0.50    0.387739       0.378728 -0.009010                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active  anomalous                15           3358         AAF0535:178      LJR0523:215   0.75    0.403684       0.401254 -0.002431                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active  anomalous                16           3358         AAF0535:179      LJR0523:215   0.75    0.427130       0.430075  0.002945                    5                         5 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active  anomalous                17           3361         AAF0535:182      LJR0523:219   0.50    0.499819       0.461089 -0.038730                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                18           3361         AAF0535:183      LJR0523:219   0.50    0.475722       0.456426 -0.019296                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                19           3361         AAF0535:184      LJR0523:219   0.50    0.500158       0.438953 -0.061205                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                20           3361         AAF0535:185      LJR0523:219   0.50    0.486079       0.457783 -0.028296                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                21           3361         AAF0535:186      LJR0523:219   0.50    0.496222       0.458539 -0.037683                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                22           1591         AAF0535:190      FMG0527:371   0.25    0.523277       0.525093  0.001816                    5                         5 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active  anomalous                23           3357         AAF0535:191      LJR0523:214   0.50    0.469645       0.467346 -0.002299                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active  anomalous                24           3354         AAF0535:192      LJR0523:211   0.50    0.490144       0.461547 -0.028597                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                25           3359         AAF0535:193      LJR0523:217   1.00    0.557493       0.535648 -0.021845                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                26           3355         AAF0535:196      LJR0523:212   0.25    0.455568       0.436612 -0.018956                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                27           1592         AAF0535:197      FMG0527:372   0.75    0.391387       0.386697 -0.004690                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active  anomalous                28           1591         AAF0535:198      FMG0527:371   0.50    0.457297       0.451232 -0.006065                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active  anomalous                29           1592         AAF0535:199      FMG0527:372   0.75    0.537432       0.542718  0.005286                    5                         5 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active  anomalous                30           1592         AAF0535:200      FMG0527:372   0.75    0.552039       0.540882 -0.011158                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                31           3358         AAF0535:203      LJR0523:215   1.00    0.460730       0.449769 -0.010962                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                32           3358         AAF0535:204      LJR0523:215   1.00    0.499942       0.495886 -0.004056                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active  anomalous                33           3362         AAF0535:205      LJR0523:220   0.25    0.544902       0.546540  0.001638                    5                         5 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active  anomalous                34           3358         AAF0535:206      LJR0523:215   1.00    0.525308       0.504456 -0.020853                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                35           1593         AAF0535:207      FMG0527:373   0.75    0.442391       0.446358  0.003967                    5                         5 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active  anomalous                36           3360         AAF0535:210      LJR0523:218   0.25    0.513800       0.485999 -0.027802                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                37           3360         AAF0535:211      LJR0523:218   0.25    0.411087       0.377917 -0.033170                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                38           3360         AAF0535:212      LJR0523:218   0.25    0.415943       0.388120 -0.027823                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                39           3360         AAF0535:213      LJR0523:218   0.25    0.405151       0.379149 -0.026002                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                40           3360         AAF0535:214      LJR0523:218   0.25    0.408635       0.390328 -0.018307                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                41           1594         AAF0535:217      FMG0527:374   0.25    0.404856       0.391278 -0.013578                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                42           3362         AAF0535:218      LJR0523:220   0.25    0.446431       0.427456 -0.018975                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                43           3357         AAF0535:219      LJR0523:214   0.75    0.426653       0.403875 -0.022778                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                44           1594         AAF0535:220      FMG0527:374   0.25    0.514211       0.481508 -0.032703                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                45           3359         AAF0535:221      LJR0523:217   0.25    0.392641       0.372645 -0.019996                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                46           3354         AAF0535:224      LJR0523:211   1.00    0.491810       0.448934 -0.042875                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                47           3354         AAF0535:225      LJR0523:211   1.00    0.505525       0.465261 -0.040264                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                48           1592         AAF0535:226      FMG0527:372   0.50    0.405165       0.376169 -0.028996                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                49           3354         AAF0535:227      LJR0523:211   1.00    0.480142       0.448858 -0.031285                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                50           1593         AAF0535:228      FMG0527:373   0.50    0.478774       0.437367 -0.041406                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                72           1072         AAM0658:294      CQW0652:423   0.25    0.453295       0.406900 -0.046395                    5                         3 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                73           1072         AAM0658:295      CQW0652:423   0.25    0.443275       0.394896 -0.048380                    5                         3 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active  anomalous                74           1065         AAM0658:296      CQW0652:414   0.25    0.518968       0.477687 -0.041281                    5                         3 [3221, 1358, 2852, 1584, 1834]    True           True
