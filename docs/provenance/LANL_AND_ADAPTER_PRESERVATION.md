# Preserved LANL data and collaborator adapters (2026-09-24)

Anvil deletes scratch files that have not been read for 30 days, without
warning, and scratch is not backed up (Anvil user guide, as quoted in RCAC
search results; RCAC's general policy page gives 60 days for most other
clusters). The raw LANL files had last been read on 2026-09-07, and several
adapters trained on the collaborator allocation on 2026-09-11/12. These copies
remove that risk. The originals on scratch are left in place.

## Raw LANL data

Source: `/anvil/scratch/x-sangdembay/lanl2015/` (downloaded 2026-09-07; the
download log records HTTP 200 and 7,626,505,158 bytes, equal to the file size).

Copy 1, Anvil project space (not purged):
`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/lanl2015_raw/`.
Every checksum was computed on the source and on the copy, and they match.

| file | bytes | sha256 |
|---|---|---|
| `auth.txt.gz` | 7,626,505,158 | `9c6b0cc261b0edd19324f6fd1839743224938a7f644ed202ca70bd70a89bf672` |
| `redteam.txt.gz` | 4,846 | `606635837c684ad11e464075ecf97bc5df325ff7d7f64614d2d8c8af18051669` |
| `windows_full.jsonl` | 16,145,532,335 | `d87ece7ebbca638a3f672b6a86beaf007285d859d2c6dc8ae4933da0baaa585c` |

`windows_full.jsonl` is the exact input of every historical LANL split. It was
built by the legacy md5 sampling (`--sample-mod 10`), so it carries the
sampling/fold coupling described in the issue ledger (L1); it is kept to
reproduce the historical artifacts, not for new evaluations.

`provenance/` holds the scripts, job files and logs as they were run:
`lanl_etl.py`, `lanl_split.py`, `lanl_manifest.py`, `manifest.csv`, the four
`*.sbatch` files and their `*.out` logs, and `auth_dl.log`. Their checksums are
in `SHA256SUMS.provenance`. The historical commands were:

```
python3 lanl_etl.py --mode build-windows --window-events 32 --sample-mod 10 --emit-ablations --out windows_full.jsonl
python3 lanl_split.py windows_full.jsonl lanl_conditions
```

Copy 2, Aquaman: `/data/suramya/insider_mi/lanl2015_raw/` (`auth.txt.gz`,
`redteam.txt.gz`, `SHA256SUMS`). Both files are verified against the source.

| file | status |
|---|---|
| `redteam.txt.gz` | Verified: sha256 `60663583…51669`, identical to the source. |
| `auth.txt.gz` | Verified 2026-09-24 14:46 CDT: 7,626,505,158 bytes, sha256 `9c6b0cc2…bf672`, identical to the source. The first stream stopped at 3.06 GB when both relay connections dropped at once. The resumed relay checked the 11 chunks already present and sent the other 18, each verified by sha256 on both ends, with no retries needed. |

## Collaborator adapters

Source: `/anvil/scratch/x-bbhusal1/{lanl,p1}/` (collaborator scratch, mode 700,
so the copy was relayed as a tar stream). Destination:
`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/checkpoints/collab_preserved_2026_09_24/`.
Each directory holds the final `adapter/` (not the checkpoints) and its
`train_summary.json`.

| adapter | used by | last read on scratch |
|---|---|---|
| `lanl/adapter_full` | LANL score decomposition; LANL repair option A | 2026-09-21 |
| `lanl/adapter_user_anon` | LANL ablation | 2026-09-11 |
| `lanl/adapter_host_anon` | LANL ablation | 2026-09-11 |
| `lanl/adapter_shuffle` | LANL ablation | 2026-09-12 |
| `p1/adapter_full` | Phase C cells A and B; 3B score decomposition | 2026-09-18 |
| `p1/adapter_targetmask` | Phase C cells C and D | 2026-09-23 |

Verification: 48 files copied (760 MB). The sha256 of every file was computed
on the source and on the copy, and all 48 match. The full list is in the
destination's `SHA256SUMS`. The weight files:

| adapter | `adapter_model.safetensors` sha256 |
|---|---|
| `lanl/adapter_full` | `c688dae5f904fb454c8bdbe404aad6b235e6e0e42f43db4ce329aa61c9c08a90` |
| `lanl/adapter_user_anon` | `9d8618571ca946c5b05ed437383831185caf58e2b984777cf96abf9fcbe03ccf` |
| `lanl/adapter_host_anon` | `53c64adefbc498c03a6a18ea1d3101129728a1c33c6a212abf9cd287493215d2` |
| `lanl/adapter_shuffle` | `794470987aa4b15d3e4ec1d331bb09e9712632e5a665f9fdbfcfeae1b0b8f4a8` |
| `p1/adapter_full` | `8f6648afad39756437566d6402bf0d077c8b823d131b08bc882b11043763107f` |
| `p1/adapter_targetmask` | `6578a196e362e3e0dae3e2793af1a18a675d87bd7eb93b36c0602e338f99a80a` |

## Not copied, deliberately

- LANL delta caches and SAEs: not needed by any planned run, and regenerable.
- The historical LANL split files: regenerable deterministically from `windows_full.jsonl`.
- The staged CERT inputs in `/anvil/scratch/x-sangdembay/pkg4_share` (274 GB): a copy of project-space originals, read by the queued CERT job when it starts. Re-stage them if that job slips past mid-October.
