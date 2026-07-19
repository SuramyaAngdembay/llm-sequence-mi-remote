# User-Level Cluster Bootstrap (necessity)

Source rows: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_necessity_qwen3_8b_no_same_user/l18_m04_k08_top5_control5_active_necessity_no_same_user/token_delta_sae_necessity_best_rows.csv`
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
|      18 |             4 |   8 | project_role   | top5     |         4 |        70 |         0.0651885 |            0.0590115 |                  4 |       0.0260589  |         0.0829195 |        0.0255038  |          0.0925193 |                   1 |                   11.5 |                  46 |
|      18 |             4 |   8 | role           | top5     |         4 |        70 |         0.0621669 |            0.0319928 |                  3 |       0.0164804  |         0.0826906 |        0.0050432  |          0.067931  |                   1 |                   11.5 |                  46 |
|      18 |             4 |   8 | dept_role      | top5     |         4 |        70 |         0.0566027 |            0.0230064 |                  2 |      -0.00185924 |         0.0774472 |       -0.00274296 |          0.0590601 |                   1 |                   11.5 |                  46 |
|      18 |             4 |   8 | team           | top5     |         4 |        70 |         0.0522338 |            0.0159671 |                  2 |      -0.0184677  |         0.0705311 |       -0.0164331  |          0.0509212 |                   1 |                   11.5 |                  46 |

## Per-user estimates

|   layer |   latent_mult |   k | context_mode   | target   | user_id   |   n_units |   user_estimate |
|--------:|--------------:|----:|:---------------|:---------|:----------|----------:|----------------:|
|      18 |             4 |   8 | dept_role      | top5     | ACM2278   |         5 |     -0.00141737 |
|      18 |             4 |   8 | dept_role      | top5     | CDE1846   |        46 |      0.0792192  |
|      18 |             4 |   8 | dept_role      | top5     | CMP2946   |        18 |      0.0182922  |
|      18 |             4 |   8 | dept_role      | top5     | MBG3183   |         1 |     -0.00406855 |
|      18 |             4 |   8 | project_role   | top5     | ACM2278   |         5 |      0.0193967  |
|      18 |             4 |   8 | project_role   | top5     | CDE1846   |        46 |      0.0824928  |
|      18 |             4 |   8 | project_role   | top5     | CMP2946   |        18 |      0.0316108  |
|      18 |             4 |   8 | project_role   | top5     | MBG3183   |         1 |      0.102546   |
|      18 |             4 |   8 | role           | top5     | ACM2278   |         5 |      0.0279175  |
|      18 |             4 |   8 | role           | top5     | CDE1846   |        46 |      0.0845443  |
|      18 |             4 |   8 | role           | top5     | CMP2946   |        18 |      0.0180908  |
|      18 |             4 |   8 | role           | top5     | MBG3183   |         1 |     -0.00258157 |
|      18 |             4 |   8 | team           | top5     | ACM2278   |         5 |     -0.019485   |
|      18 |             4 |   8 | team           | top5     | CDE1846   |        46 |      0.0723553  |
|      18 |             4 |   8 | team           | top5     | CMP2946   |        18 |      0.0243793  |
|      18 |             4 |   8 | team           | top5     | MBG3183   |         1 |     -0.0133812  |

## Leave-one-user-out

|   layer |   latent_mult |   k | context_mode   | target   | held_out_user   |   estimate_without_user |   held_out_user_estimate |
|--------:|--------------:|----:|:---------------|:---------|:----------------|------------------------:|-------------------------:|
|      18 |             4 |   8 | dept_role      | top5     | ACM2278         |               0.0610658 |              -0.00141737 |
|      18 |             4 |   8 | dept_role      | top5     | CDE1846         |               0.0132543 |               0.0792192  |
|      18 |             4 |   8 | dept_role      | top5     | CMP2946         |               0.069864  |               0.0182922  |
|      18 |             4 |   8 | dept_role      | top5     | MBG3183         |               0.057482  |              -0.00406855 |
|      18 |             4 |   8 | project_role   | top5     | ACM2278         |               0.0687109 |               0.0193967  |
|      18 |             4 |   8 | project_role   | top5     | CDE1846         |               0.0320218 |               0.0824928  |
|      18 |             4 |   8 | project_role   | top5     | CMP2946         |               0.0768115 |               0.0316108  |
|      18 |             4 |   8 | project_role   | top5     | MBG3183         |               0.0646471 |               0.102546   |
|      18 |             4 |   8 | role           | top5     | ACM2278         |               0.0648014 |               0.0279175  |
|      18 |             4 |   8 | role           | top5     | CDE1846         |               0.0192767 |               0.0845443  |
|      18 |             4 |   8 | role           | top5     | CMP2946         |               0.0774239 |               0.0180908  |
|      18 |             4 |   8 | role           | top5     | MBG3183         |               0.0631052 |              -0.00258157 |
|      18 |             4 |   8 | team           | top5     | ACM2278         |               0.0577506 |              -0.019485   |
|      18 |             4 |   8 | team           | top5     | CDE1846         |               0.0136676 |               0.0723553  |
|      18 |             4 |   8 | team           | top5     | CMP2946         |               0.0618757 |               0.0243793  |
|      18 |             4 |   8 | team           | top5     | MBG3183         |               0.0531847 |              -0.0133812  |
