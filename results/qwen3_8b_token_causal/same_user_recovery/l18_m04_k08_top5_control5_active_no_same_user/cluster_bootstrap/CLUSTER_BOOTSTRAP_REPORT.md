# User-Level Cluster Bootstrap (causal)

Source rows: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh_no_same_user/l18_m04_k08_top5_control5_active_no_same_user/token_delta_sae_causal_best_rows.csv`
Control set: `control5_active`. Bootstrap draws: `10000` (user-level resampling).

The `pooled_estimate` column must match the receiver/pair-level bootstrap
estimate for the same configuration; the cluster CIs replace the
day-level CIs as the honest uncertainty statement.

NOTE: only 4 malicious users in at least one configuration.
Percentile CIs from so few clusters are unstable; per-user estimates,
sign agreement (`n_users_positive` / `n_users`), and leave-one-user-out
results are the more informative robustness statements.

## Summary

|   layer |   latent_mult |   k | context_mode   | target   |   n_users |   n_units |   pooled_estimate |   user_mean_estimate |   n_users_positive |   cluster_ci_low |   cluster_ci_high |   usermean_ci_low |   usermean_ci_high |   days_per_user_min |   days_per_user_median |   days_per_user_max |
|--------:|--------------:|----:|:---------------|:---------|----------:|----------:|------------------:|---------------------:|-------------------:|-----------------:|------------------:|------------------:|-------------------:|--------------------:|-----------------------:|--------------------:|
|      18 |             4 |   8 | role           | top5     |         4 |        70 |        0.0068477  |           0.00272671 |                  4 |      9.18905e-05 |        0.00999603 |       9.57608e-05 |         0.00768374 |                   1 |                   11.5 |                  46 |
|      18 |             4 |   8 | dept_role      | top5     |         4 |        70 |        0.0068181  |           0.00245765 |                  2 |     -0.00022998  |        0.00995532 |      -0.000457402 |         0.00761247 |                   1 |                   11.5 |                  46 |
|      18 |             4 |   8 | project_role   | top5     |         4 |        70 |        0.00420063 |           0.00118107 |                  3 |     -5.82437e-05 |        0.00589168 |      -0.00176987  |         0.00467204 |                   1 |                   11.5 |                  46 |

## Per-user estimates

|   layer |   latent_mult |   k | context_mode   | target   | user_id   |   n_units |   user_estimate |
|--------:|--------------:|----:|:---------------|:---------|:----------|----------:|----------------:|
|      18 |             4 |   8 | dept_role      | top5     | ACM2278   |         5 |    -0.000117141 |
|      18 |             4 |   8 | dept_role      | top5     | CDE1846   |        46 |     0.010189    |
|      18 |             4 |   8 | dept_role      | top5     | CMP2946   |        18 |     0.000552909 |
|      18 |             4 |   8 | dept_role      | top5     | MBG3183   |         1 |    -0.000794172 |
|      18 |             4 |   8 | project_role   | top5     | ACM2278   |         5 |     0.000457072 |
|      18 |             4 |   8 | project_role   | top5     | CDE1846   |        46 |     0.00607703  |
|      18 |             4 |   8 | project_role   | top5     | CMP2946   |        18 |     0.000825004 |
|      18 |             4 |   8 | project_role   | top5     | MBG3183   |         1 |    -0.00263482  |
|      18 |             4 |   8 | role           | top5     | ACM2278   |         5 |     8.99553e-05 |
|      18 |             4 |   8 | role           | top5     | CDE1846   |        46 |     0.0102111   |
|      18 |             4 |   8 | role           | top5     | CMP2946   |        18 |     0.000504196 |
|      18 |             4 |   8 | role           | top5     | MBG3183   |         1 |     0.000101566 |

## Leave-one-user-out

|   layer |   latent_mult |   k | context_mode   | target   | held_out_user   |   estimate_without_user |   held_out_user_estimate |
|--------:|--------------:|----:|:---------------|:---------|:----------------|------------------------:|-------------------------:|
|      18 |             4 |   8 | dept_role      | top5     | ACM2278         |             0.00735158  |             -0.000117141 |
|      18 |             4 |   8 | dept_role      | top5     | CDE1846         |             0.000357187 |              0.010189    |
|      18 |             4 |   8 | dept_role      | top5     | CMP2946         |             0.00898682  |              0.000552909 |
|      18 |             4 |   8 | dept_role      | top5     | MBG3183         |             0.00692842  |             -0.000794172 |
|      18 |             4 |   8 | project_role   | top5     | ACM2278         |             0.0044886   |              0.000457072 |
|      18 |             4 |   8 | project_role   | top5     | CDE1846         |             0.000604192 |              0.00607703  |
|      18 |             4 |   8 | project_role   | top5     | CMP2946         |             0.00536912  |              0.000825004 |
|      18 |             4 |   8 | project_role   | top5     | MBG3183         |             0.0042997   |             -0.00263482  |
|      18 |             4 |   8 | role           | top5     | ACM2278         |             0.00736752  |              8.99553e-05 |
|      18 |             4 |   8 | role           | top5     | CDE1846         |             0.000401119 |              0.0102111   |
|      18 |             4 |   8 | role           | top5     | CMP2946         |             0.00904352  |              0.000504196 |
|      18 |             4 |   8 | role           | top5     | MBG3183         |             0.00694547  |              0.000101566 |
