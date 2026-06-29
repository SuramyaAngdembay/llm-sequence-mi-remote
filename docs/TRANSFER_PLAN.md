# Transfer Plan

This plan assumes the target is the Anvil project space described in:

- [ANVIL_SYSTEM_SPECS.md](./ANVIL_SYSTEM_SPECS.md)

Primary destination:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/`

## What To Transfer First

Only ship the minimum first-pass payload:

1. raw session shards (`sessionr6.2` or `sessionr4.2`)
2. `labels_daily.parquet`
3. `session*_user_map.csv`
4. `sha256sums.txt`
5. `manifest.json`

Do not transfer:

- local checkpoints
- benchmark parquets
- activation dumps
- SAE outputs

## Shard Format

Default local packaging format:

- CSV shards with header in every shard
- `gzip` compression

Expected names, depending on dataset:

- `sessionr6.2_shard_000.csv.gz`
- `sessionr4.2_shard_000.csv.gz`
- ...

Companion files:

- `labels_daily.parquet`
- `session*_user_map.csv`
- `manifest.json`
- `sha256sums.txt`

## Recommended Order

### Option A: Globus

Preferred when available on both sides.

Destination path:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/`

Pros:

- resumable
- checksummed
- stable for large transfers

### Option B: rsync over SSH

Use when source machine can SSH directly to Anvil.

Example:

```bash
rsync -avP --partial \
  /path/to/transfer_package/ \
  x-sangdembay@anvil.rcac.purdue.edu:/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/
```

## On Anvil After Transfer

Verify checksums:

```bash
cd /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/
sha256sum -c sha256sums.txt
```

If unpacking is needed:

```bash
gunzip -t sessionr6.2_shard_000.csv.gz
```

## Local Source Paths

Raw session source:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/lcdal-r62-full/extract_stage/r6.2/ExtractedData/sessionr6.2.csv`
- `/homes/01/srangdembay/insider_threat/r4.2/ExtractedData/sessionr4.2.csv`

Labels:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-semantic-daily-ldap/labels_daily.parquet`
- `/homes/01/srangdembay/InsiderThreatDetection/r4.2/labels_daily.parquet`
