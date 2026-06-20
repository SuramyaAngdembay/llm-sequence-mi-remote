# Data Spec

## Primary Source

Preferred source for the LLM branch:

- `sessionr6.2.csv`
  - path:
    - `/homes/01/srangdembay/InsiderThreatDetection/r6.2/lcdal-r62-full/extract_stage/r6.2/ExtractedData/sessionr6.2.csv`

Use the raw session table rather than the already standardized benchmark parquet because the LLM branch needs:

- per-session ordering
- session metadata
- user/day grouping
- user/host/time structure before standardization

## Current Scale

From the prepared session benchmark:

- rows: `1,949,074`
- active user-day pairs: `1,393,297`
- users: `4,000`
- day indices: `516`
- numeric benchmark feature count after metadata drops: `243`
- matched positive user-day pairs: `70`
- matched positive session rows: `101`

Reference:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_lcdal_session_features_clean/sessionr6.2_prepare_stats.json`

## Required Companion Labels

Ship the label file used for evaluation:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-semantic-daily-ldap/labels_daily.parquet`

This remains the canonical evaluation label source for user/day anomaly scoring.

## Canonical Session Schema Groups

The raw session table contains:

- identity/context:
  - `user`, `day`, `week`, `pc`, `project`, `role`, `b_unit`, `f_unit`, `dept`, `team`, `ITAdmin`
- temporal/session:
  - `starttime`, `endtime`, `duration`, `n_days`, `n_concurrent_sessions`, `start_with`, `end_with`, `ses_start`, `ses_end`, `isworkhour`, `isafterhour`, `isweekend`, `isweekendafterhour`
- psychometrics:
  - `O`, `C`, `E`, `A`, `N`
- channel families:
  - `usb_*`
  - `file_*`
  - `email_*`
  - `http_*`
  - aggregate counts like `n_allact`, `n_logon`, `n_usb`, `n_file`, `n_email`, `n_http`

## Recommended Remote Transfer Units

Do not transfer a single monolithic CSV if avoidable.

Preferred transfer artefacts:

1. `sessionr6.2.csv` split by `user_id` hash bucket or by day range
2. compressed with `zstd` or `gzip`
3. checksum manifest included

Recommended shard naming:

- `sessionr6.2_shard_000.csv.zst`
- `sessionr6.2_shard_001.csv.zst`
- ...

Recommended companion files:

- `labels_daily.parquet`
- `manifest.json`
- `sha256sums.txt`

## Canonical LLM Training Representation

Do not convert sessions into free-form prose.

Preferred serialized format:

- one JSON object per `user_id/day_index`
- ordered list of sessions inside each example

Example sketch:

```json
{
  "user_id": "U1234",
  "day_index": 212,
  "day_of_week": 2,
  "org": {
    "project": "P17",
    "role": "R4",
    "dept": "D9",
    "team": "T2",
    "pc_home_match": true
  },
  "psychometric": {"O": 38, "C": 24, "E": 22, "A": 40, "N": 33},
  "sessions": [
    {
      "start_hour": 7.27,
      "end_hour": 11.75,
      "pc": "PC042",
      "is_workhour": 0,
      "is_afterhour": 0,
      "duration": 269.58,
      "n_usb": 0,
      "n_file": 0,
      "n_email": 11,
      "n_http": 51
    }
  ]
}
```

## Minimal Shipping Plan

For the first pilot, ship only:

- raw `sessionr6.2` shards
- `labels_daily.parquet`
- one small smoke subset:
  - around `5k-20k` user-day examples

Do not ship:

- local model checkpoints
- large activation dumps
- SAE outputs

Those should be produced on the remote cluster.

