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
    24            4  8         role   top5       70                70                     -0.001895                   -0.001036                         -0.000027                       -0.002017                 0.000859                    -0.001989                     0.002848

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                    feature_ids  mean_row_gap
    24            4  8            top5           5 [1900, 2642, 2146, 4976, 5627]      0.129575
    24            4  8        control1           1                         [2785]     -0.000983
    24            4  8 control5_active           5  [5846, 1978, 6452, 494, 1613]     -0.000006

## Example Receiver-Level Best Ablations

 layer  latent_mult  k context_mode     feature_set receiver_type  pair_idx  receiver_row_idx  matched_row_idx receiver_example_id matched_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens             selected_features  effect  strong_effect
    24            4  8         role control5_active        benign         0               931               71         BSS2956:283        ACM2278:228   0.50    0.251041       0.252746  0.001705                    5                        19 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign         1              5070               72         OKH3777:280        ACM2278:231   1.00    0.329284       0.303965 -0.025319                    5                        15 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign         2              4571               73         MRS0409:106        ACM2278:232   1.00    0.423325       0.379611 -0.043715                    5                         4 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign         3              2605               74         GSH1593:301        ACM2278:233   1.00    0.279099       0.268309 -0.010790                    5                         4 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign         4              2600               75         GSH1593:185        ACM2278:234   1.00    0.299640       0.283635 -0.016005                    5                         5 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign         5              5989             1255         SMM3221:442        CDE1846:415   0.25    0.353208       0.362321  0.009114                    5                        25 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign         6               528             1256         AXW3243:247        CDE1846:416   0.25    0.420736       0.393755 -0.026981                    5                        25 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign         7              5304             1257         QKO2718:232        CDE1846:417   0.25    0.506839       0.546578  0.039739                    5                         5 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign         8              1413             1258         CJC1433:116        CDE1846:420   0.25    0.358529       0.369292  0.010763                    5                        22 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign         9               534             1259         AXW3243:463        CDE1846:421   0.50    0.446837       0.432685 -0.014151                    5                         8 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        10              4051             1260         LRC3160:385        CDE1846:422   0.25    0.370043       0.382511  0.012468                    5                         2 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        11              6768             1261         WRF3534:506        CDE1846:423   0.25    0.409231       0.351805 -0.057426                    5                         6 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        12              5334             1262         QRR1242:450        CDE1846:424   1.00    0.344444       0.367730  0.023286                    5                         5 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        13              5448             1263         RHO0678:212        CDE1846:427   0.75    0.325657       0.331196  0.005538                    5                         5 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        14              5320             1264         QRR1242:171        CDE1846:428   0.25    0.355861       0.380263  0.024401                    5                         5 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        15              5794             1265         SDM3560:196        CDE1846:429   1.00    0.309449       0.330808  0.021359                    5                         4 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        16              4041             1266          LRC3160:11        CDE1846:430   0.25    0.363881       0.345925 -0.017956                    5                         3 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        17               680             1267          BJW3558:28        CDE1846:431   0.25    0.442536       0.430576 -0.011960                    5                         6 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        18              5975             1268         SLS0834:507        CDE1846:434   1.00    0.336549       0.405504  0.068954                    5                         7 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        19              2983             1269         ILY3152:498        CDE1846:435   0.50    0.382535       0.352686 -0.029849                    5                         5 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        20              3743             1270         KRD2100:191        CDE1846:436   1.00    0.331842       0.316907 -0.014935                    5                         4 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        21              2340             1271         FNW3564:206        CDE1846:437   1.00    0.452934       0.445615 -0.007319                    5                         5 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active        benign        22              1154             1272          CCB0547:91        CDE1846:438   1.00    0.276438       0.288506  0.012067                    5                        29 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        23              6325             1273         TRW3172:477        CDE1846:441   0.25    0.410134       0.382655 -0.027479                    5                         7 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        24              5790             1274         SDM3560:112        CDE1846:442   0.75    0.338903       0.348501  0.009598                    5                         5 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        25              5102             1275         PAA2738:233        CDE1846:443   0.25    0.246130       0.248845  0.002715                    5                        18 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        26              2451             1276         GFB1385:203        CDE1846:444   1.00    0.339374       0.323776 -0.015597                    5                         4 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        27              5962             1277          SLS0834:43        CDE1846:445   1.00    0.243612       0.269631  0.026019                    5                         7 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        28              4197             1278         MBG3183:451        CDE1846:448   1.00    0.393542       0.391300 -0.002242                    5                         5 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active        benign        29              2977             1279         ILY3152:228        CDE1846:449   0.50    0.314483       0.307963 -0.006519                    5                         3 [5846, 1978, 6452, 494, 1613]    True          False
    24            4  8         role control5_active        benign        30              2983             1280         ILY3152:498        CDE1846:450   0.50    0.382535       0.352686 -0.029849                    5                         5 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        31              1467             1281          CLE3098:53        CDE1846:451   1.00    0.288163       0.301821  0.013658                    5                         5 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        32               533             1282         AXW3243:457        CDE1846:452   0.25    0.387926       0.373844 -0.014082                    5                        13 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        33              4204             1283         MBL1247:199        CDE1846:455   0.25    0.465239       0.423713 -0.041526                    5                        20 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        34              6083             1284          SSH1845:50        CDE1846:456   0.75    0.383961       0.346770 -0.037190                    5                        19 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        35               329             1285          AKP3573:92        CDE1846:457   0.25    0.413707       0.387831 -0.025876                    5                         4 [5846, 1978, 6452, 494, 1613]    True           True
    24            4  8         role control5_active        benign        36              5989             1286         SMM3221:442        CDE1846:458   0.25    0.353208       0.362321  0.009114                    5                        25 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        37              5966             1287         SLS0834:151        CDE1846:459   1.00    0.257410       0.282073  0.024663                    5                        15 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        38              1871             1288         DJP3230:217        CDE1846:462   1.00    0.325321       0.382411  0.057090                    5                         9 [5846, 1978, 6452, 494, 1613]   False          False
    24            4  8         role control5_active        benign        39              4853             1289         NJO1460:444        CDE1846:463   0.75    0.462794       0.487953  0.025159                    5                         7 [5846, 1978, 6452, 494, 1613]   False          False
