# Token Delta SAE Necessity Eval

Token-level feature ablation on adapter deltas at hidden-state layer `24` with SAE config `latent_mult=4, k=8`.

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

 layer  latent_mult  k context_mode target  n_pairs  n_complete_pairs  top_positive_mean_best_delta  top_benign_mean_best_delta  control_positive_mean_best_delta  control_benign_mean_best_delta  top_necessity_advantage  control_necessity_advantage  top_minus_control_necessity
    24            4  8         role   top5       70                70                      -0.01077                   -0.010159                         -0.011993                       -0.003744                 0.000612                     0.008249                    -0.007637

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                   feature_ids  mean_row_gap
    24            4  8            top5           5 [668, 1657, 4773, 2894, 6026]      0.073916
    24            4  8        control1           1                         [748]      0.002688
    24            4  8 control5_active           5  [3100, 6390, 6572, 943, 266]     -0.000010

## Example Receiver-Level Best Ablations

 layer  latent_mult  k context_mode     feature_set receiver_type  pair_idx  receiver_row_idx  matched_row_idx receiver_example_id matched_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens            selected_features  effect  strong_effect
    24            4  8         role control5_active        benign         0               931               71         BSS2956:283        ACM2278:228   0.75    0.290168       0.284684 -0.005483                    5                         7 [3100, 6390, 6572, 943, 266]    True          False
    24            4  8         role control5_active        benign         1              5070               72         OKH3777:280        ACM2278:231   0.50    0.368105       0.375970  0.007865                    5                         7 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign         2              4571               73         MRS0409:106        ACM2278:232   0.25    0.342225       0.406087  0.063862                    5                         1 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign         3              2605               74         GSH1593:301        ACM2278:233   0.75    0.394336       0.312600 -0.081736                    5                         1 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign         4              2600               75         GSH1593:185        ACM2278:234   0.25    0.414976       0.412499 -0.002477                    5                         0 [3100, 6390, 6572, 943, 266]    True          False
    24            4  8         role control5_active        benign         5              5989             1255         SMM3221:442        CDE1846:415   0.25    0.326389       0.337164  0.010775                    5                         2 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign         6               528             1256         AXW3243:247        CDE1846:416   0.75    0.421396       0.424310  0.002914                    5                         5 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign         7              5304             1257         QKO2718:232        CDE1846:417   0.50    0.479654       0.525660  0.046006                    5                         2 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign         8              1413             1258         CJC1433:116        CDE1846:420   1.00    0.343856       0.343218 -0.000638                    5                         9 [3100, 6390, 6572, 943, 266]    True          False
    24            4  8         role control5_active        benign         9               534             1259         AXW3243:463        CDE1846:421   0.25    0.470930       0.537753  0.066823                    5                         5 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        10              4051             1260         LRC3160:385        CDE1846:422   0.50    0.402204       0.430183  0.027979                    5                         4 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        11              6768             1261         WRF3534:506        CDE1846:423   1.00    0.500897       0.503887  0.002990                    5                         3 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        12              5334             1262         QRR1242:450        CDE1846:424   0.75    0.235421       0.242015  0.006594                    5                         3 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        13              5448             1263         RHO0678:212        CDE1846:427   1.00    0.483396       0.421359 -0.062037                    5                         1 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        14              5320             1264         QRR1242:171        CDE1846:428   0.75    0.237051       0.231970 -0.005081                    5                         2 [3100, 6390, 6572, 943, 266]    True          False
    24            4  8         role control5_active        benign        15              5794             1265         SDM3560:196        CDE1846:429   0.25    0.432014       0.335914 -0.096100                    5                         2 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        16              4041             1266          LRC3160:11        CDE1846:430   0.25    0.380578       0.411323  0.030746                    5                         4 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        17               680             1267          BJW3558:28        CDE1846:431   0.25    0.509047       0.425129 -0.083918                    5                         6 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        18              5975             1268         SLS0834:507        CDE1846:434   0.25    0.538404       0.570019  0.031615                    5                         3 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        19              2983             1269         ILY3152:498        CDE1846:435   0.25    0.402515       0.400769 -0.001746                    5                         9 [3100, 6390, 6572, 943, 266]    True          False
    24            4  8         role control5_active        benign        20              3743             1270         KRD2100:191        CDE1846:436   1.00    0.309098       0.394313  0.085214                    5                         8 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        21              2340             1271         FNW3564:206        CDE1846:437   1.00    0.433956       0.419437 -0.014520                    5                         5 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        22              1154             1272          CCB0547:91        CDE1846:438   0.75    0.288693       0.318639  0.029946                    5                         3 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        23              6325             1273         TRW3172:477        CDE1846:441   0.25    0.389741       0.390167  0.000426                    5                         4 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        24              5790             1274         SDM3560:112        CDE1846:442   0.50    0.424996       0.316951 -0.108045                    5                         3 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        25              5102             1275         PAA2738:233        CDE1846:443   0.50    0.216611       0.198133 -0.018478                    5                        10 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        26              2451             1276         GFB1385:203        CDE1846:444   0.75    0.288270       0.301136  0.012866                    5                         8 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        27              5962             1277          SLS0834:43        CDE1846:445   0.75    0.372957       0.374502  0.001545                    5                         5 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        28              4197             1278         MBG3183:451        CDE1846:448   1.00    0.307573       0.294771 -0.012801                    5                         6 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        29              2977             1279         ILY3152:228        CDE1846:449   0.75    0.335357       0.338635  0.003278                    5                         8 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        30              2983             1280         ILY3152:498        CDE1846:450   0.25    0.402515       0.400769 -0.001746                    5                         9 [3100, 6390, 6572, 943, 266]    True          False
    24            4  8         role control5_active        benign        31              1467             1281          CLE3098:53        CDE1846:451   0.25    0.357511       0.376526  0.019015                    5                         3 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        32               533             1282         AXW3243:457        CDE1846:452   1.00    0.403444       0.444040  0.040595                    5                         8 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        33              4204             1283         MBL1247:199        CDE1846:455   0.25    0.346863       0.380909  0.034047                    5                         6 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        34              6083             1284          SSH1845:50        CDE1846:456   0.50    0.439153       0.483554  0.044401                    5                         5 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        35               329             1285          AKP3573:92        CDE1846:457   0.50    0.471506       0.374816 -0.096690                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        36              5989             1286         SMM3221:442        CDE1846:458   0.25    0.326389       0.337164  0.010775                    5                         2 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        37              5966             1287         SLS0834:151        CDE1846:459   1.00    0.383548       0.394946  0.011397                    5                         5 [3100, 6390, 6572, 943, 266]   False          False
    24            4  8         role control5_active        benign        38              1871             1288         DJP3230:217        CDE1846:462   0.50    0.447443       0.385767 -0.061676                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
    24            4  8         role control5_active        benign        39              4853             1289         NJO1460:444        CDE1846:463   1.00    0.424644       0.409256 -0.015388                    5                         4 [3100, 6390, 6572, 943, 266]    True           True
