# Train-matched serialization counterfactuals (P1 factorial), CERT r6.2

Four adapters per scale, identical recipe (benign-only QLoRA, 300k examples, one adapter seed per scale), differing only in the profile block of the serialization built by `scripts/build_session_jsonl_fast.py --profile-mode {full,no_psy,no_profile,shuffle_profile}` (`shuffle_profile` = seeded derangement over all users; each user keeps one fixed borrowed profile, so per-user constancy is preserved and only the profile<->user correspondence is broken). Scales: Qwen2.5-3B (`slurm/anvil_friend/run_mode.sh`) and Qwen3-8B (4xH100 DDP, `slurm/anvil_friend/run_8b.sh`). Each condition changes the serialization used for BOTH adaptation and scoring.

Scoring: `scripts/eval_fold_aligned_detector_metrics.py` (`slurm/anvil_friend/fold_eval.sbatch`): one held-out malicious user per fold, benign test cohort = min(800, available) = all 406 benign users of the pool (407 test users per fold, identical cohort across folds), day-level ROC and user-level ROC after max-aggregation over days. Uncertainty: `scripts/factorial/fold_boot.py`, user-level cluster bootstrap (10,000 draws over the 4 malicious users), paired across arms. With 4 clusters the intervals are descriptive (conditional on the trained adapters and the fixed cohort); they do not characterize variation across training seeds.

## Fold-aligned results (per held-out malicious user: ACM2278, CDE1846, CMP2946, MBG3183)

| scale | condition | mean user ROC | mean day ROC | per-user user ROC | held-out user rank of 407 |
|---|---|---|---|---|---|
| 3b | full | 0.926 | 0.741 | 0.995, 0.803, 0.966, 0.941 | 3, 81, 15, 25 |
| 3b | no_psy | 0.903 | 0.718 | 0.916, 0.926, 0.924, 0.845 | 35, 31, 32, 64 |
| 3b | no_profile | 0.956 | 0.816 | 0.993, 0.993, 0.993, 0.845 | 4, 4, 4, 64 |
| 3b | shuffle_profile | 0.656 | 0.516 | 0.894, 0.488, 0.520, 0.722 | 44, 209, 196, 114 |
| 8b | full | 0.532 | 0.399 | 0.919, 0.155, 0.527, 0.527 | 34, 344, 193, 193 |
| 8b | no_psy | 0.906 | 0.714 | 0.931, 0.872, 0.911, 0.911 | 29, 53, 37, 37 |
| 8b | no_profile | 0.903 | 0.827 | 0.867, 0.978, 0.951, 0.818 | 55, 10, 21, 75 |
| 8b | shuffle_profile | 0.485 | 0.423 | 0.345, 0.241, 0.574, 0.778 | 267, 309, 174, 91 |

Cluster bootstrap (`anvil_run_summaries.txt`, jobs 20672049 [8B] / 20710009 [3B]): 8B full 0.532 [0.248, 0.821], no_psy 0.906 [0.882, 0.926], no_profile 0.903 [0.842, 0.964], shuffle 0.485 [0.293, 0.676]; removal (no_profile - full) = +0.371 [+0.067, +0.690], one-sided bootstrap tail fraction P(diff<=0) = 0.0051. 3B full 0.926 [0.844, 0.982], no_psy 0.903 [0.865, 0.925], no_profile 0.956 [0.882, 0.993], shuffle 0.656 [0.504, 0.808]; removal = +0.030 [-0.065, +0.142], tail fraction 0.3188.

## Delta-SAE audit of the full and shuffle adapters (top-5 profile mass; `anvil_run_summaries.txt`)

- 3B full (best layer 12): 1.000, 0.853, 0.997, 0.026, 0.992 -> 4/5 profile-bound (job 20564928)
- 3B shuffle (layer 12): 1.000, 0.920, 1.000, 1.000, 0.985 -> 5/5 (job 20564929)
- 8B full (layer 24): 0.053, 0.041, 0.095, 1.000, 1.000 -> 2 hard-profile features (job 20595298)
- 8B shuffle (layer 24): 0.085, 0.095, 0.257, 0.998, 1.000 -> 2 hard-profile features (job 20595301)

## PSY-novelty check (`scripts/factorial/profile_novelty.py`, job 20714955; 3,590 benign training users)

| user | org tuple exact matches | min L1 to a benign PSY | benign users within L1<=10 | 8B full user ROC | 3B full user ROC |
|---|---|---|---|---|---|
| ACM2278 | 29 | 9 | 3 | 0.919 | 0.995 |
| CDE1846 | 13 | 7 | 12 | 0.155 | 0.803 |
| CMP2946 | 22 | 6 | 11 | 0.527 | 0.966 |
| MBG3183 | 4 | 12 | 0 | 0.527 | 0.941 |

Reading: the inverted 8B user (CDE1846) has the least novel PSY profile; the surviving user (ACM2278) one of the most novel; the most novel (MBG3183, 0 neighbours) still sits at chance at user level but is the best detected at day level (0.653). Novelty orders the users only approximately; n=4.

`roc/` holds the quick-eval JSONs (user-disjoint ROC vs val benign) that preceded the fold-aligned re-score; the paper reports the fold-aligned numbers.
