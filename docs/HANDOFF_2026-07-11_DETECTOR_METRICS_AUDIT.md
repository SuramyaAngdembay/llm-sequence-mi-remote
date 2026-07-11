# Detector Metrics Audit

Status: detector-metric audit completed on 2026-07-11 after inspecting the
actual `example_scores.parquet` layout on Anvil.

## Bug Summary

The earlier detector-metric artifacts were generated with:

- `--split eval`

But the remote `example_scores.parquet` files already come from the extracted
`eval.jsonl` pool, which itself contains:

- all positive-user days (`split="eval"`)
- plus benign validation-user days (`split="val"`)

So applying `--split eval` a second time incorrectly dropped the benign
validation portion of the extracted evaluation pool.

## Confirmed Anvil Split Layout

For `r6.2`:

- total score rows: `142072`
- total score users: `410`
- split rows inside the parquet:
  - `val = 140711`
  - `eval = 1361`

For `r4.2`:

- total score rows: `42468`
- total score users: `149`
- split rows inside the parquet:
  - `val = 27026`
  - `eval = 15442`

So the earlier committed detector artifacts were scoring only the `eval`-labeled
slice, not the full extracted remote evaluation pool.

## Consequence

The old detector artifacts overstated detector quality, especially on `r6.2`,
because they reduced the user pool to:

- `5` users on `r6.2`
- `70` users on `r4.2`

The corrected detector artifacts now score **all rows in the score parquet**.

## Corrected Headline Read

`r6.2`, `adapted_nll`:

- day PR-AUC `0.0005017303747021707`
- day ROC-AUC `0.547214375250248`
- user PR-AUC `0.022071356156067014`
- first positive user rank `37`

`r4.2`, `adapted_nll`:

- day PR-AUC `0.0686890523564744`
- day ROC-AUC `0.670377882593637`
- user PR-AUC `0.5475400517260249`
- first positive user rank `2`

## Interpretation

This is a real correction, not a rounding issue.

It means:

- the remote CERT mechanistic story remains valid
- but the remote detector story is weaker than the earlier `eval`-only
  detector artifacts suggested

It also exposes a broader benchmark mismatch:

- the remote detector pool is `val + positive-user days`
- the local detector baselines use leave-one-malicious-user-out full test folds

So the current remote-vs-local detector table is **not yet a clean apples-to-
apples benchmark**. A fair detector benchmark will require a fold-aligned remote
scoring path, not just this detector-metrics fix.
