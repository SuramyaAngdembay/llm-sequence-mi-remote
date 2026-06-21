# Anvil System Specs (Remote Target)

> **Answers to `ANVIL_DISCOVERY_CHECKLIST.md`.** Filled by the remote (Anvil-side) agent on **2026-06-20**.
> Cluster: **Purdue Anvil** (RCAC / NSF ACCESS). Login node observed: `login03.anvil.rcac.purdue.edu`.
> User: `x-sangdembay` · Allocation: `cis230270`. This is the authoritative description of the system
> the data-prep/transfer agent is shipping to. All values verified live (`scontrol`, `sfeatures`,
> `mybalance`, `myquota`) plus RCAC docs.

---

## TL;DR for the data-prep agent
- **Send data here:** `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/`
  (the **projects** filesystem — 5 TB, persistent, NOT purged). Do **not** ship into `$HOME` (25 GB) or `$SCRATCH` (30-day purge).
- **Easiest transfer:** **Globus** if the source machine has (or can run) a Globus endpoint — robust, resumable, checksummed. Otherwise **`rsync -avP` over SSH** from the source to the Anvil login node.
- Ship the **r6.2 session shards + `labels_daily.parquet` + `sessionr6.2_user_map.csv` + `sha256sums.txt`** per `DATA_SPEC.md`. Compress with `zstd`/`gzip`.
- Anvil **compute nodes have no internet**; the **login nodes do**. All transfers/unpacks/model-downloads happen on the login node.

---

## 1. Hardware

### Primary GPU target — H100 (`ai` partition) ✅
| Item | Value |
|---|---|
| GPU model | **NVIDIA H100 SXM, 80 GB** each (NVLink) |
| GPUs per node | **4** (`Gres=gpu:4`) |
| GPU memory / device | **80 GB** |
| CPU / node | 2× Intel Xeon Platinum 8468 → **96 cores** |
| Host RAM / node | **~1 TB** (1,031,000 MB) |
| Nodes | `h000`–`h020` (**21 nodes**), Dell PowerEdge XE9640 |
| Feature tag | `H100` |

This matches the repo's recommended "1× H100 80GB" primary target exactly.

### Alternative GPU — A100 (`gpu` / `gpu-debug` partitions)
| Item | Value |
|---|---|
| GPU model | **NVIDIA A100 SXM, 40 GB** each |
| GPUs per node | **4** |
| CPU / node | **128 cores** · RAM **512 GB** |
| Nodes | `g000`–`g015` (**16 nodes**) |

> Note: Anvil A100s are **40 GB** (not 80 GB). Use `gpu-debug` for cheap smoke tests; H100 (`ai`) for real runs.

### Local scratch / parallel filesystem
- No fast node-local NVMe is exposed for general use; **`$SCRATCH` is the high-performance tier** (GPFS, 10 PB pool, up to ~150 GB/s). Treat `$SCRATCH` as the "scratch" referenced in `ENVIRONMENT.md`.

---

## 2. Scheduler (Slurm)
| Item | Value |
|---|---|
| GPU partitions | **`ai`** (H100), `gpu` / `gpu-debug` (A100), plus CPU partitions |
| Account string (H100) | **`cis230270-ai`** |
| Account string (A100) | **`cis230270-gpu`** |
| Account string (CPU) | **`cis230270`** |
| Partition↔account | **Enforced** — `ai`→`-ai`, `gpu`/`gpu-debug`→`-gpu`, CPU→base account |
| Time limits | Partitions show no hard cap, but **keep jobs ≤ 24–48 h** and checkpoint (see §6) |
| GPU granularity | Nodes are **shared**; request `--gres=gpu:N` (N=1..4) + a slice of cores/RAM |
| Interactive GPU | `sinteractive -A cis230270-ai -p ai -N1 -n24 --gres=gpu:1 -t 1:00:00` |
| Job arrays | Supported (standard Slurm) |
| Node exclusivity | Not required; use `--exclusive` only for whole-node runs |

### Charging (`mybalance`, 2026-06-20)
| Account | Type | SU balance | ≈ |
|---|---|---|---|
| `cis230270-ai` | AI | **933** | ~933 H100-GPU-hours |
| `cis230270-gpu` | GPU | 860 | ~860 A100-GPU-hours |
| `cis230270` | CPU | 22,059 | CPU core-hours |

~**1 SU = 1 GPU-hour**. The 3B QLoRA + delta + SAE pilots (est. ~12–30 GPU-hr total) fit comfortably in the AI balance.

---

## 3. Environment
| Item | Value |
|---|---|
| Preferred setup | **conda** via `module load conda/2026.03` (Anaconda 2026.03). `rclone`, `apptainer` also available as modules |
| Python | 3.10 / 3.11 (the project env will pin 3.11) |
| CUDA | Driver supports CUDA 12.x; build the env against a matching `torch` cu12 wheel |
| `bitsandbytes` | **Works** on A100/H100. A reference env already exists with `peft`+`bitsandbytes`+`transformers` (`/anvil/scratch/x-sangdembay/conda_envs/mcts-vqa-env`) |
| Compute-node internet | **NO egress on compute nodes.** Login nodes have full internet |
| Env location | Build in `/anvil/projects/x-cis230270/x-sangdembay/conda_envs` (`~/.condarc` already redirects envs+pkgs there to spare the home quota) |
| Model download | Pull `Qwen/Qwen2.5-3B` on the **login node** into `$HF_HOME`, then run training with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |

---

## 4. Transfer (how to get data onto Anvil)

**Tools confirmed present on the Anvil login node:** `rsync`, `scp`, `sftp`, `globus` (CLI), `globus-url-copy`, `rclone` (module). Anvil runs **Globus Connect Server v5**.

### Option A — Globus (recommended for the bulk data)
- In the Globus web app (app.globus.org), search collections for **"Anvil"** (RCAC-managed GCSv5 collection covering Anvil home/scratch/projects). Confirm the exact display name in the web UI.
- Destination path inside the Anvil collection: **`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/`**.
- Source side: needs a Globus endpoint. If the source is an institutional cluster it likely already has one; if it's a lab/personal box, install **Globus Connect Personal** (free) and create a collection over the data directory.
- Pros: resumable, parallel, auto-checksum/verify, survives disconnects, fire-and-forget. Best for multi-GB.

### Option B — rsync over SSH (simplest for a one-off)
From the **source machine**, push to Anvil:
```bash
rsync -avP --partial -e ssh \
  /homes/01/srangdembay/.../ExtractedData/sessionr6.2*.zst \
  x-sangdembay@anvil.rcac.purdue.edu:/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/
```
- Anvil access host: **`anvil.rcac.purdue.edu`** (round-robin login nodes). Requires the source machine's SSH key to be authorized for the Anvil account.
- `scp` works too but `rsync -avP --partial` is preferred: resumable, incremental, progress.

### Which is easier?
- **Source already has/can use a Globus endpoint → use Globus** (most robust; matches the `DATA_SPEC` sharded+checksummed plan).
- **Just need a quick one-shot and source can SSH to Anvil → rsync** (zero setup, but babysit it; re-run to resume).
- Either way: ship `zstd`/`gzip` shards + `sha256sums.txt`, then verify on Anvil with `sha256sum -c sha256sums.txt`.

### Login nodes for unpack jobs?
Yes — login nodes are fine for decompressing/checksumming/serializing modest data. For very large/long unpacks, use a short CPU batch job (`-A cis230270 -p shared`).

---

## 5. Storage
| Space | Path / var | Quota | Backed up | Purge | Use for |
|---|---|---|---|---|---|
| home | `/home/x-sangdembay` (`$HOME`) | 25 GB | ✅ yes | never | code, repo, notes |
| scratch | `/anvil/scratch/x-sangdembay` (`$SCRATCH`) | 100 TB / 1M files | ❌ no | **30-day atime purge** | transient run I/O, token caches, delta dumps during a job |
| projects | `/anvil/projects/x-cis230270/x-sangdembay` (`$PROJECT`=`$WORK`) | 5 TB / 1M files | ❌ no | none while alloc active | **raw data shards, adapter checkpoints, SAE outputs, persistent results** |

**Best location per artifact type:**
- **Raw data shards / labels** → `$PROJECT/.../cert-qlora-MI/data/` (persistent).
- **Token caches** → `$SCRATCH` (regenerable) or `$PROJECT` if reused across many jobs.
- **Adapter checkpoints** → `$PROJECT` (persistent).
- **Delta activations** → `$SCRATCH` during the job, copy keepers to `$PROJECT`. ⚠️ Shard into few large files — `$SCRATCH` is near the **1M-file** limit (currently 997K).

---

## 6. Practical constraints
- **Recommended max walltime:** keep individual jobs **≤ 24 h** (48 h ceiling). Long single jobs are allowed but riskier.
- **Checkpoint/resume:** **yes, implement it** — save adapter + optimizer state every N steps so a job can resume after a timeout/preemption.
- **No compute-node internet:** stage everything (data + model weights) on the login node first.
- **Scratch file-count pressure:** avoid millions of tiny files on `$SCRATCH`.
- Use `gpu-debug` (A100) for cheap pipeline smoke tests before spending H100 (AI) SUs.

---

## 7. Filled-in Slurm header (for reference)
```bash
#SBATCH --account=cis230270-ai      # H100 allocation  (use cis230270-gpu for the gpu/gpu-debug A100 partitions)
#SBATCH --partition=ai              # H100 nodes        (or: gpu-debug for smoke tests)
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24          # ~1/4 of the 96-core H100 node per GPU
#SBATCH --mem=240G                  # ~1/4 of ~1 TB
#SBATCH --time=24:00:00
module load conda/2026.03
conda activate /anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora
export HF_HOME=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Sources
- Anvil AI / H100: https://www.rcac.purdue.edu/news/7268 · Partitions: https://www.rcac.purdue.edu/knowledge/anvil/run/partitions?all=true
- Filesystems: https://www.rcac.purdue.edu/knowledge/anvil/storage/filesystems · Scratch purge: https://www.rcac.purdue.edu/policies/scratchpurge
- Accounting/SU: https://www.rcac.purdue.edu/knowledge/anvil/run/accounting · Globus: https://www.rcac.purdue.edu/knowledge/anvil/storage/transfer/globus
