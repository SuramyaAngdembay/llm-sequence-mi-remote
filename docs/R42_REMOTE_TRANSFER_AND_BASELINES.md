# R4.2 Remote Transfer And Baseline Selection

This note does two jobs:

1. fixes the baseline set we will carry forward for the paper
2. gives the concrete `r4.2` remote transfer path for the Anvil branch

## Baseline Set We Are Selecting

We do **not** want a huge model zoo. We want a compact baseline set that tests
the main alternatives a reviewer would reasonably expect.

### Main detector baselines

Carry forward these detector families:

1. **Local session AE**
   - This is the strongest local mechanistic baseline.
   - It stays in the table because it is the best non-LLM branch we found.

2. **Masked reconstruction (`MCM`)**
   - We already have this family in the local benchmark history.
   - It tests whether the gain is really about token/session causal patching, not
     just another reconstruction-style detector.

3. **LSTM autoencoder**
   - This is the most important missing classic sequential baseline.
   - If we add only one new classic baseline, this is it.

4. **GRU autoencoder**
   - Same family as LSTM, cheaper and often used in insider-threat pipelines.
   - Good to include if implementation cost is low after the LSTM path exists.

5. **Deep SVDD**
   - Tests whether a strong one-class hypersphere approach can match the
     reconstruction-based branches.

6. **Isolation Forest**
   - Simple classical non-neural baseline.
   - Low prestige, but reviewers often want one.

7. **Remote token `Qwen3` branch**
   - This is now the main method.
   - For `r6.2`, the strongest audited headline is the `Qwen3-8B` token branch.

### Why we are not selecting more

We are **not** prioritizing:

- more CTMC variants
- more daily-only baselines
- broad transformer zoo expansion
- more local SAE micro-variants

Reason: those questions are already largely answered by the earlier sweeps.

The core comparison now is:

- classic anomaly detectors
- best local mechanistic baseline
- best remote token causal branch

## Which datasets these baselines should run on

### `r6.2`

Use for the full headline table:

- local session AE
- MCM
- LSTM AE
- GRU AE
- Deep SVDD
- Isolation Forest
- remote `Qwen3-8B` token causal branch

### `r4.2`

Use for the transfer / generalization table:

- local session AE
- LSTM AE
- GRU AE
- Deep SVDD
- Isolation Forest
- remote `Qwen3` token branch

`MCM` on `r4.2` is optional. It is useful if easy to wire, but not required for
the first transfer pass.

## Why `r4.2` next

`r4.2` is the cleanest next dataset because:

- it is in the same CERT family
- raw `sessionr4.2.csv` already exists locally
- `labels_daily.parquet` already exists locally
- it tests whether the causal patch result is specific to `r6.2` or survives a
  different CERT version

This is a better next step than jumping straight to another model scale-up.

## Local `r4.2` sources

Raw session source:

- `/homes/01/srangdembay/insider_threat/r4.2/ExtractedData/sessionr4.2.csv`

Daily labels:

- `/homes/01/srangdembay/InsiderThreatDetection/r4.2/labels_daily.parquet`

The transfer prep script regenerates `sessionr4.2_user_map.csv` automatically
from the original CERT extraction code path:

- `/homes/01/srangdembay/insider_threat/r4.2/feature_extraction.py`

Specifically, it uses the same `getuserlist('r4.2')` logic that produced the
numeric user encoding in the extracted session table.

## How to prepare the `r4.2` transfer package

From Magnolia:

```bash
cd /homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote
bash scripts/prepare_r42_transfer_package.sh
```

This writes:

- `artifacts/transfer_package_r42/sessionr4.2_shard_*.csv.gz`
- `artifacts/transfer_package_r42/labels_daily.parquet`
- `artifacts/transfer_package_r42/sessionr4.2_user_map.csv`
- `artifacts/transfer_package_r42/manifest.json`
- `artifacts/transfer_package_r42/sha256sums.txt`

## How to rsync to Anvil

```bash
cd /homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote
bash scripts/rsync_to_anvil.sh artifacts/transfer_package_r42
```

This now defaults to a dedicated dataset subdirectory on Anvil:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2/`

That avoids mixing `r4.2` manifests/checksums with the existing `r6.2` package
already stored at the project root.

On Anvil:

```bash
cd /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2/
sha256sum -c sha256sums.txt
```

## Next remote steps after transfer

Once the `r4.2` package lands on Anvil, the remote branch can follow the same
shape as `r6.2`:

1. build session JSONL
2. train `Qwen3` QLoRA
3. extract token deltas
4. token-SAE frontier
5. token causal patching

The first goal is **not** to sweep many new models. It is to test whether the
current winning remote token intervention path survives dataset transfer.
