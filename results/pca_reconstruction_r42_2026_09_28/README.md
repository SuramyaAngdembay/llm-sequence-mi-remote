# PCA reconstruction-error detector on CERT r4.2 (thesis Chapter III, 2026-09-28)

**Status: done; CPU only, on Magnolia.** This run supplies the numbers of thesis Section 3.3. It
modernizes the PCA code in `InsiderThreatDetection/r4.2/pca_based_detection.ipynb` on Magnolia,
following the user's request of 2026-09-28. It is a thesis experiment. It is separate from the
earlier 10-component probe in `results/cross_arch_probe/`, which used different features, split
and labels.

## What changed from the notebook

| notebook (cell 8 and the GMM variants) | this run (`scripts/pca_reconstruction_r42.py`) |
|---|---|
| malicious = `insider == 1`, so the scenario-2 and scenario-3 days count as benign | malicious = `insider > 0`: 966 event-days (85, 861, 20) |
| user, day, week, start and end times used as features | identifiers and time indices removed; 502 features |
| random split of days, not user-disjoint | the language model's split: train 851 benign users; held-out 79 benign users; 70 insiders |
| GMM on the errors | no GMM; the score is the reconstruction error $e = \|x - \hat x\|^2$ (thesis eq. 2.7) |
| day-level ROC only | user AUC (user score = largest daily error) and day AUC, 95% intervals resampling users |

## Protocol

- **Data.** `r4.2/ExtractedData/dayr4.2.csv` has 330,452 user-days and 1,000 users.
- **Scaling and PCA.** The scaler and PCA are fit on the 287,961 days of the 851 training users.
- **Components.** q is the smallest number of directions reaching 95% of the variance, following
  sklearn's rule. One full SVD per variant is checked against sklearn's own truncation.
- **Consistency checks (all pass).**
  - the 70 insider users are exactly the eval split;
  - every user has a split;
  - the training days contain no malicious day;
  - the pairwise user AUC equals `roc_auc_score`.
- **Bootstrap.** 10,000 resamples for user AUC and 2,000 user-cluster resamples for day AUC;
  seed 42.

## Finding: columns constant in training

193 of the 502 columns never vary among the training days. StandardScaler gives such a column
scale 1, and no retained principal direction touches it. Any nonzero value on an evaluation day
therefore enters the error in raw units.

32 of these columns are nonzero on evaluation days:
- leak-site visits: 68 days, all malicious;
- hacking-site visits: 10 days, all malicious;
- logons to the supervisor's PC (`pc3`, per `feature_extractor.py:399`): 5 days, 4 malicious.

Together these columns carry 99.7% of the malicious days' squared error. Standardization is
undefined for a zero standard deviation, so the thesis uses the variant without them (p = 309)
and reports the other as a sensitivity check. The business unit (`b_unit`) is one of the 193.

## Results (60 = the language model's insider pool; each against the same 79 held-out benign users)

| variant | p | q (95%) | user AUC, 60 insiders | user AUC, 70 insiders | day AUC, 60 insiders |
|---|---|---|---|---|---|
| **varying in training (thesis)** | 309 | 71 | **0.751 [0.668, 0.828]** | 0.777 [0.701, 0.848] | 0.759 [0.717, 0.801] |
| behavior only, varying in training | 299 | 63 | 0.756 [0.672, 0.834] | 0.791 [0.716, 0.860] | 0.793 [0.753, 0.829] |
| all 502 (constant columns in raw units) | 502 | 71 | 0.804 [0.727, 0.875] | 0.832 [0.764, 0.894] | 0.762 [0.720, 0.802] |
| behavior only, all | 491 | 63 | 0.808 [0.730, 0.877] | 0.835 [0.765, 0.896] | 0.795 [0.758, 0.830] |

**Threshold sensitivity.** User AUC for the thesis variant, 60 insiders:

| variance kept | q | user AUC |
|---|---|---|
| 80% | 36 | 0.814 |
| 90% | 55 | 0.754 |
| 95% | 71 | 0.751 |
| 99% | 94 | 0.783 |

**Errors (thesis variant).** Medians:

| days | median error |
|---|---|
| training days | 5.75 |
| held-out benign days | 6.63 |
| malicious days | 16.98 |
| insiders' other days | 8.04 |

6.4% of malicious days exceed the training 99th percentile, against 0.5% of held-out benign days.

**Labels.** Day AUC uses the extraction's event-day labels, not `labels_daily`. It is therefore
not comparable with day-level language-model numbers. User AUC depends only on which users are
malicious; see `docs/LABEL_ALIGNMENT_2026-09-28.md`.

## Files and jobs

| item | detail |
|---|---|
| `pca_reconstruction_r42.json`, `pca_recon_r42-576348.out` | final run: job 576348, 5 min 3 s, 8.2 GB, 1 node |
| `pca_reconstruction_r42_job576345.json`, `pca_recon_r42-576345.out` | first run (2 variants). The final run reproduces it exactly |
| `pca_recon_r42-576347.out` | failed on a wrong self-check: every direction, rather than only the first q, was required to be zero on the constant columns. Fixed |
| `r42_user_splits.csv`, `sessionr4.2_user_map.csv` | split and user-code inputs; md5 `533048…`, `bbd6ba…`. Kept on Magnolia next to the run (the repo ignores `*.csv`); the split follows `pkg4_share/session_jsonl_r42/example_metadata.parquet` |
| input day table | md5 `9e4f7d99de43126c726e9ebe27019ca2` (on Magnolia, not copied) |

```
sbatch slurm/magnolia_pca_reconstruction_r42.sbatch   # on Magnolia, from r4.2/pca_reconstruction_2026_09_28/
```
