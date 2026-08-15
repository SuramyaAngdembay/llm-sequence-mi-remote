# Token Delta SAE Necessity Eval

Token-level feature ablation on adapter deltas at hidden-state layer `24` with SAE config `latent_mult=2, k=4`.

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
    24            2  4         team   top5     1301              1301                     -0.013517                   -0.009405                         -0.014149                       -0.012216                 0.004112                     0.001933                     0.002179

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                    feature_ids  mean_row_gap
    24            2  4            top5           5  [819, 2668, 3412, 3286, 3659]      0.043946
    24            2  4        control1           1                         [3522]      0.010515
    24            2  4 control5_active           5 [3221, 1358, 2852, 1584, 1834]     -0.000006

## Example Receiver-Level Best Ablations

 layer  latent_mult  k context_mode     feature_set receiver_type  pair_idx  receiver_row_idx  matched_row_idx receiver_example_id matched_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens              selected_features  effect  strong_effect
    24            2  4         team control5_active        benign         0               543               14          BQS0525:43        AAF0535:177   1.00    0.526792       0.487599 -0.039193                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign         1              3340               15          LJR0523:85        AAF0535:178   0.25    0.358075       0.349778 -0.008297                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign         2              1589               16         FMG0527:346        AAF0535:179   0.75    0.326804       0.318402 -0.008403                    5                         6 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign         3              1575               17          FMG0527:88        AAF0535:182   0.50    0.328268       0.334186  0.005918                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign         4              1575               18          FMG0527:88        AAF0535:183   0.75    0.328268       0.333551  0.005283                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign         5              3346               19         LJR0523:158        AAF0535:184   0.75    0.378921       0.362874 -0.016047                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign         6               543               20          BQS0525:43        AAF0535:185   1.00    0.526792       0.487599 -0.039193                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign         7              3335               21          LJR0523:10        AAF0535:186   0.25    0.361138       0.345651 -0.015487                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign         8              1163               22          CYA0506:86        AAF0535:190   0.75    0.508416       0.581066  0.072650                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign         9               543               23          BQS0525:43        AAF0535:191   1.00    0.526792       0.487599 -0.039193                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        10              1581               24         FMG0527:225        AAF0535:192   0.25    0.321405       0.343958  0.022553                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        11              3363               25         LJR0523:231        AAF0535:193   0.75    0.378317       0.343601 -0.034717                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        12              3338               26          LJR0523:56        AAF0535:196   0.75    0.367140       0.346602 -0.020539                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        13              3339               27          LJR0523:58        AAF0535:197   0.75    0.379744       0.355689 -0.024054                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        14              3337               28          LJR0523:38        AAF0535:198   0.75    0.360229       0.346186 -0.014044                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        15              3341               29          LJR0523:92        AAF0535:199   1.00    0.372685       0.364709 -0.007976                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign        16              1580               30         FMG0527:204        AAF0535:200   1.00    0.324576       0.338903  0.014327                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        17              1159               31          CYA0506:28        AAF0535:203   0.25    0.515301       0.571383  0.056082                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        18              3344               32         LJR0523:127        AAF0535:204   1.00    0.389879       0.365590 -0.024289                    5                         4 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        19              1576               33          FMG0527:92        AAF0535:205   0.75    0.305782       0.314537  0.008755                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        20              1579               34         FMG0527:186        AAF0535:206   1.00    0.308129       0.327641  0.019512                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        21              1571               35          FMG0527:39        AAF0535:207   0.25    0.328874       0.315591 -0.013282                    5                         7 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        22              1162               36          CYA0506:79        AAF0535:210   0.75    0.508651       0.574844  0.066193                    5                         5 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        23              3350               37         LJR0523:176        AAF0535:211   0.75    0.356314       0.337190 -0.019124                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        24              3341               38          LJR0523:92        AAF0535:212   1.00    0.372685       0.364709 -0.007976                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign        25              1588               39         FMG0527:343        AAF0535:213   0.75    0.332626       0.324348 -0.008278                    5                         6 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign        26              1573               40          FMG0527:77        AAF0535:214   0.75    0.313385       0.324867  0.011482                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        27              3343               41         LJR0523:119        AAF0535:217   0.25    0.370421       0.358790 -0.011631                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        28              1582               42         FMG0527:261        AAF0535:218   0.50    0.335632       0.324478 -0.011154                    5                         7 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        29              1575               43          FMG0527:88        AAF0535:219   0.50    0.328268       0.334186  0.005918                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        30              1576               44          FMG0527:92        AAF0535:220   0.75    0.305782       0.314537  0.008755                    5                         6 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        31              1165               45          CYA0506:99        AAF0535:221   0.50    0.516894       0.553322  0.036428                    5                         4 [3221, 1358, 2852, 1584, 1834]   False          False
    24            2  4         team control5_active        benign        32               543               46          BQS0525:43        AAF0535:224   1.00    0.526792       0.487599 -0.039193                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        33              1582               47         FMG0527:261        AAF0535:225   0.50    0.335632       0.324478 -0.011154                    5                         7 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        34              3347               48         LJR0523:161        AAF0535:226   1.00    0.352832       0.345448 -0.007384                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign        35               542               49          BQS0525:29        AAF0535:227   0.50    0.371631       0.362189 -0.009442                    5                         5 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign        36              3346               50         LJR0523:158        AAF0535:228   1.00    0.378921       0.359614 -0.019307                    5                         5 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        37              3377               72         LNR0656:179        AAM0658:294   0.50    0.445979       0.441934 -0.004046                    5                         4 [3221, 1358, 2852, 1584, 1834]    True          False
    24            2  4         team control5_active        benign        38              1061               73         CQW0652:387        AAM0658:295   0.50    0.440274       0.372122 -0.068152                    5                         4 [3221, 1358, 2852, 1584, 1834]    True           True
    24            2  4         team control5_active        benign        39              3108               74         KLM0639:420        AAM0658:296   0.50    0.411970       0.404039 -0.007931                    5                         3 [3221, 1358, 2852, 1584, 1834]    True          False
