# Required Artifacts

## Must Transfer

- raw `sessionr6.2` shards
- `labels_daily.parquet`
- checksum manifest
- smoke subset

## Must Produce Remotely

- JSONL sequence shards
- tokenized caches
- adapter checkpoints
- delta activation shards
- delta-SAE outputs
- evaluation reports

## Do Not Commit To GitHub

- raw data
- token caches
- activation tensors
- checkpoints
- large parquet outputs

