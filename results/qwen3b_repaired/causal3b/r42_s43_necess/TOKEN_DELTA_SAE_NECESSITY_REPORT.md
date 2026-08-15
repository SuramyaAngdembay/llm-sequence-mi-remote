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
    24            2  4         team   top5     1301              1301                      0.001619                   -0.000775                          0.002862                       -0.001681                -0.002395                    -0.004544                     0.002149

## Selected Feature Sets

 layer  latent_mult  k     feature_set  n_features                   feature_ids  mean_row_gap
    24            2  4            top5           5 [3800, 2172, 2719, 105, 3333]      0.043802
    24            2  4        control1           1                        [2312]     -0.000303
    24            2  4 control5_active           5 [1807, 841, 3356, 2960, 2216]      0.000005

## Example Receiver-Level Best Ablations

 layer  latent_mult  k context_mode     feature_set receiver_type  pair_idx  receiver_row_idx  matched_row_idx receiver_example_id matched_example_id  alpha  base_score  patched_score     delta  n_selected_features  n_active_receiver_tokens             selected_features  effect  strong_effect
    24            2  4         team control5_active        benign         0               543               14          BQS0525:43        AAF0535:177   0.25    0.505753       0.529847  0.024094                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign         1              3340               15          LJR0523:85        AAF0535:178   0.75    0.331768       0.355597  0.023829                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign         2              1589               16         FMG0527:346        AAF0535:179   0.75    0.385138       0.394080  0.008942                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign         3              1575               17          FMG0527:88        AAF0535:182   0.75    0.390376       0.375313 -0.015064                    5                         4 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active        benign         4              1575               18          FMG0527:88        AAF0535:183   0.50    0.390376       0.371271 -0.019105                    5                         4 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active        benign         5              3346               19         LJR0523:158        AAF0535:184   0.50    0.346738       0.361171  0.014433                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign         6               543               20          BQS0525:43        AAF0535:185   0.25    0.505753       0.529847  0.024094                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign         7              3335               21          LJR0523:10        AAF0535:186   0.50    0.328065       0.335061  0.006996                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign         8              1163               22          CYA0506:86        AAF0535:190   0.50    0.415239       0.503732  0.088493                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign         9               543               23          BQS0525:43        AAF0535:191   0.25    0.505753       0.529847  0.024094                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        10              1581               24         FMG0527:225        AAF0535:192   0.50    0.377664       0.362312 -0.015353                    5                         4 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active        benign        11              3363               25         LJR0523:231        AAF0535:193   1.00    0.346525       0.367386  0.020861                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        12              3338               26          LJR0523:56        AAF0535:196   0.25    0.333293       0.349962  0.016670                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        13              3339               27          LJR0523:58        AAF0535:197   0.25    0.350110       0.375498  0.025388                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        14              3337               28          LJR0523:38        AAF0535:198   0.75    0.332900       0.349366  0.016466                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        15              3341               29          LJR0523:92        AAF0535:199   1.00    0.340833       0.352953  0.012120                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        16              1580               30         FMG0527:204        AAF0535:200   1.00    0.381247       0.382735  0.001488                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        17              1159               31          CYA0506:28        AAF0535:203   0.25    0.424822       0.505685  0.080863                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        18              3344               32         LJR0523:127        AAF0535:204   0.75    0.364385       0.381097  0.016712                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        19              1576               33          FMG0527:92        AAF0535:205   1.00    0.361038       0.355324 -0.005714                    5                         5 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active        benign        20              1579               34         FMG0527:186        AAF0535:206   0.50    0.374501       0.366460 -0.008042                    5                         5 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active        benign        21              1571               35          FMG0527:39        AAF0535:207   0.50    0.385325       0.398552  0.013227                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        22              1162               36          CYA0506:79        AAF0535:210   0.25    0.416987       0.480964  0.063976                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        23              3350               37         LJR0523:176        AAF0535:211   0.25    0.337219       0.350532  0.013313                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        24              3341               38          LJR0523:92        AAF0535:212   1.00    0.340833       0.352953  0.012120                    5                         5 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        25              1588               39         FMG0527:343        AAF0535:213   0.25    0.389401       0.397272  0.007871                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        26              1573               40          FMG0527:77        AAF0535:214   0.25    0.378323       0.382568  0.004246                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        27              3343               41         LJR0523:119        AAF0535:217   0.25    0.335768       0.356820  0.021052                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        28              1582               42         FMG0527:261        AAF0535:218   1.00    0.383735       0.401169  0.017434                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        29              1575               43          FMG0527:88        AAF0535:219   0.75    0.390376       0.375313 -0.015064                    5                         4 [1807, 841, 3356, 2960, 2216]    True           True
    24            2  4         team control5_active        benign        30              1576               44          FMG0527:92        AAF0535:220   1.00    0.361038       0.355324 -0.005714                    5                         5 [1807, 841, 3356, 2960, 2216]    True          False
    24            2  4         team control5_active        benign        31              1165               45          CYA0506:99        AAF0535:221   0.50    0.417287       0.481054  0.063768                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        32               543               46          BQS0525:43        AAF0535:224   0.25    0.505753       0.529847  0.024094                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        33              1582               47         FMG0527:261        AAF0535:225   1.00    0.383735       0.401169  0.017434                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        34              3347               48         LJR0523:161        AAF0535:226   0.50    0.325821       0.344943  0.019122                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        35               542               49          BQS0525:29        AAF0535:227   0.25    0.338203       0.340723  0.002520                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        36              3346               50         LJR0523:158        AAF0535:228   1.00    0.346738       0.362686  0.015948                    5                         6 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        37              3377               72         LNR0656:179        AAM0658:294   0.75    0.309231       0.406027  0.096796                    5                         2 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        38              1061               73         CQW0652:387        AAM0658:295   0.25    0.472805       0.506684  0.033879                    5                         4 [1807, 841, 3356, 2960, 2216]   False          False
    24            2  4         team control5_active        benign        39              3108               74         KLM0639:420        AAM0658:296   0.75    0.463627       0.486290  0.022663                    5                         2 [1807, 841, 3356, 2960, 2216]   False          False
