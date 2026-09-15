# LANL 'Comprehensive, Multi-Source Cyber-Security Events' (cyber1) replication

Real enterprise Windows authentication log (Kent 2015; 58 days, 12,425 users, 1.05B events, 749 red-team compromise events over 104 users), no profile of any kind. ETL: `scripts/lanl/lanl_etl.py` (32-event windows per user-day; a window is positive iff it contains a red-team event; all red-team users kept, 1/10 sample of benign users -> 1.62M windows, 432 positive), `lanl_manifest.py`, `lanl_split.py` (md5 5-fold user split, fold 4 unseen; 300k benign training windows, 2k val; eval = all positives + 30k seen + 30k unseen windows). Adapter: Qwen2.5-3B benign-only QLoRA per serialization (`slurm/anvil_friend/lanl_run.sh`, `lanl_extract.sh`), scored by adapted NLL. Four serializations differ only in the identity tokens: full, user anonymized, host anonymized, user deranged (fixed permutation).

| serialization | seen AUC | seen AP | unseen AUC | unseen AP |
|---|---|---|---|---|
| full (user + host) | 0.949 | 0.177 | 0.497 | 0.013 |
| user anonymized | 0.944 | 0.176 | 0.518 | 0.019 |
| host anonymized | 0.877 | 0.069 | 0.627 | 0.013 |
| user deranged | 0.947 | 0.173 | 0.487 | 0.012 |

Prevalence: seen 234/22563 = 0.0104; unseen 198/30410 = 0.0065. Unseen AP sits at the base rate in every condition: the host-anonymization gain is in ranking, not alert precision.

## Measurements behind the paper's wording (`scripts/lanl/lanl_checks.py`, `lanl_checks_output.txt`)

- Hosts are user-associated, not per-user-constant: among the 1,180 training users with >=20 events, the modal source host covers a median 54% of a user's events (q25 0.40, q75 0.61); only 7% of users issue >=80% of events from one host; modal destination host median 16%.
- No truncation: longest full-serialization window 1,503 tokens (p99 1,408; longest positive 1,417) against max_seq_len 2,048; host-anonymized max 1,311.
- Single 3B adapter per condition; red-team credential misuse is the ground truth, so this replicates the mechanism, not a detection benchmark.
