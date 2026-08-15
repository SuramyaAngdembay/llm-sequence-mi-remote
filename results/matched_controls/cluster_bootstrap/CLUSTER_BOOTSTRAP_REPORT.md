# User-Level Cluster Bootstrap (causal)

Source rows: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/matched_controls/causal_r62/token_delta_sae_causal_best_rows.csv`
Control set: `control5_matched`. Bootstrap draws: `10000` (user-level resampling).

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
|      18 |             4 |   8 | role           | top5     |         4 |        70 |        0.00594946 |           0.00223891 |                  2 |      -0.00175209 |        0.00960481 |       -0.00112002 |         0.00730234 |                   1 |                   11.5 |                  46 |

## Per-user estimates

|   layer |   latent_mult |   k | context_mode   | target   | user_id   |   n_units |   user_estimate |
|--------:|--------------:|----:|:---------------|:---------|:----------|----------:|----------------:|
|      18 |             4 |   8 | role           | top5     | ACM2278   |         5 |    -0.000151873 |
|      18 |             4 |   8 | role           | top5     | CDE1846   |        46 |     0.00978708  |
|      18 |             4 |   8 | role           | top5     | CMP2946   |        18 |    -0.00190026  |
|      18 |             4 |   8 | role           | top5     | MBG3183   |         1 |     0.0012207   |

## Leave-one-user-out

|   layer |   latent_mult |   k | context_mode   | target   | held_out_user   |   estimate_without_user |   held_out_user_estimate |
|--------:|--------------:|----:|:---------------|:---------|:----------------|------------------------:|-------------------------:|
|      18 |             4 |   8 | role           | top5     | ACM2278         |              0.00641879 |             -0.000151873 |
|      18 |             4 |   8 | role           | top5     | CDE1846         |             -0.00140597 |              0.00978708  |
|      18 |             4 |   8 | role           | top5     | CMP2946         |              0.00866667 |             -0.00190026  |
|      18 |             4 |   8 | role           | top5     | MBG3183         |              0.00601799 |              0.0012207   |
