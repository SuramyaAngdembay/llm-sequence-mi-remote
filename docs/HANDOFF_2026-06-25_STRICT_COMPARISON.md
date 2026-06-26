# Handoff 2026-06-25: Strict Token-vs-Session Comparison

This handoff finishes the comparison logic after token-level causal patching became weakly positive.

The main correction is that the earlier local session-AE comparison was on **session rows**, while the remote token QLoRA branch is on **receiver user-days**.
We now collapse the local session repair rows to one value per `(user_id, day_index)` receiver day.

## Current Read

Start here:

- `results/qwen3b_pilot/strict_compare_remote70_daylevel/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

Main numbers on the matched receiver-day unit:

- best local adaptive session-AE day-level advantage: `0.001133`
- best local residual day-level advantage: `0.000654`
- best remote token QLoRA advantage: `0.001405`

So on the correct comparison unit, the current remote token branch is **ahead** of the local session-AE branch.

This is not the final fully-closed branch decision yet because two checks still matter:

1. bootstrap CIs over the `70` positive receivers for the remote token best rows
2. a clean rerun of `m04/k04` because its control set previously collapsed to one feature

## Latest Synced Update

The first of those two checks is now done.

Synced back from Anvil into:

- `artifacts/anvil_token_causal/l18_m02_k08/bootstrap/`
- `artifacts/anvil_token_causal/l18_m02_k08/`
- `artifacts/anvil_token_causal/l18_m04_k04/`

The strongest `m02/k08` bootstrap rows remain positive:

- `team/top1`: estimate `0.001405`, CI `[0.000706, 0.002114]`
- `team/top5`: estimate `0.001382`, CI `[0.000729, 0.002059]`
- `team/top3`: estimate `0.000907`, CI `[0.000248, 0.001541]`

So the updated read is:

- the matched day-level comparison already put remote token ahead of local session-AE
- the best remote token config now also has a clean positive bootstrap interval
- the remaining `m04/k04_controlfix` rerun is now a stability/control-quality check, not the main source of evidence

## Code Already Added

These are already committed on `main`:

- `scripts/compare_remote_token_vs_local_session.py`
- `scripts/bootstrap_token_delta_sae_causal.py`
- `slurm/bootstrap_token_delta_sae_causal.template.sbatch`

Also patched:

- `scripts/sae_core.py`

The control feature selection is now more robust and does not immediately collapse `control3` to one feature when the active-only low-gap pool is too small.

## Local Strict Comparison (Already Done)

This is the command that generated the matched day-level report:

```bash
python scripts/compare_remote_token_vs_local_session.py \
  --session-feature-path /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_lcdal_session_features_clean/sessionr6.2_features.parquet \
  --local-run-root /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_session_lcdal_autoencoder_mech_clean/plain \
  --local-adaptive-best-rows /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_session_lcdal_feature_bundle_repair_adaptive_remote70/adaptive_feature_bundle_repair_best_rows.csv \
  --local-residual-best-rows /homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_session_lcdal_feature_bundle_repair_residual_remote70/residual_feature_bundle_repair_best_rows.csv \
  --remote-summary-csv results/qwen3b_pilot/token_causal/l18_m02_k08/token_delta_sae_causal_summary.csv \
  --out-dir results/qwen3b_pilot/strict_compare_remote70_daylevel
```

## If Anvil CSVs Need To Be Pulled Here

If the full per-receiver token outputs exist only on Anvil, pull them with `rsync`.

Local destination:

```bash
mkdir -p artifacts/anvil_token_causal/l18_m02_k08
mkdir -p artifacts/anvil_token_causal/l18_m04_k04
mkdir -p artifacts/anvil_token_causal/l18_m04_k04_controlfix
```

Pull the current best-row CSVs:

```bash
rsync -avP \
  x-sangdembay@anvil.rcac.purdue.edu:/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m02_k08/token_delta_sae_causal_best_rows.csv \
  artifacts/anvil_token_causal/l18_m02_k08/

rsync -avP \
  x-sangdembay@anvil.rcac.purdue.edu:/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m04_k04/token_delta_sae_causal_best_rows.csv \
  artifacts/anvil_token_causal/l18_m04_k04/
```

If you want the other small summaries too:

```bash
rsync -avP \
  --include='*/' \
  --include='token_delta_sae_causal_best_rows.csv' \
  --include='token_delta_sae_causal_summary.csv' \
  --include='token_delta_sae_causal_selected_sets.csv' \
  --exclude='*' \
  x-sangdembay@anvil.rcac.purdue.edu:/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/ \
  artifacts/anvil_token_causal/
```

## Anvil Pull And Remaining Jobs

First update the repo:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
```

### Anvil update

Current `m02/k08` bootstrap is complete and committed under:

- `results/qwen3b_pilot/token_causal/l18_m02_k08/bootstrap/`

The strongest `m02/k08` rows remain positive under bootstrap:

- `team/top1`: estimate `0.001405`, 95% CI `[0.000706, 0.002114]`
- `team/top5`: estimate `0.001382`, 95% CI `[0.000729, 0.002059]`
- `team/top3`: estimate `0.000907`, 95% CI `[0.000248, 0.001541]`

The original `m04/k04` artifact synced here still shows the expected control defect:

- `control3` collapsed to a single feature `[8043]`

The control-fixed `m04/k04` rerun is queued on Anvil as `18586875`; its dependent CPU bootstrap is `18586876`.

Final update: the control-fixed `m04/k04` rerun completed on Anvil as `18590322`, and its bootstrap completed
as `18590324`. The duplicate A100 attempt was canceled after the H100 run finished.

Control-fix result:

- `control3` now contains three features: `[101, 173, 230]`
- best row: `team/top3`, estimate `0.001446`, 95% CI `[0.000623, 0.002328]`
- next strongest row: `project_role/top5`, estimate `0.000749`, 95% CI `[0.000306, 0.001207]`

The fixed-control rerun remains positive, so the token branch still clears the two requested checks:

1. `m02/k08` bootstrap stays clearly positive for the top team rows.
2. `m04/k04_controlfix` remains positive with a real 3-feature control set.

### 1. Bootstrap the current best `m02/k08`

```bash
BEST_ROWS_CSV=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m02_k08/token_delta_sae_causal_best_rows.csv \
OUT_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m02_k08/bootstrap \
sbatch slurm/bootstrap_token_delta_sae_causal.template.sbatch
```

Expected outputs:

- `token_delta_sae_bootstrap_summary.csv`
- `TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`

### 2. Re-run `m04/k04` with the fixed control pool

```bash
LAYER=18 LATENT_MULT=4 TOPK=4 \
OUTPUT_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m04_k04_controlfix \
sbatch slurm/eval_token_delta_sae_causal.template.sbatch
```

Expected outputs:

- `token_delta_sae_causal_best_rows.csv`
- `token_delta_sae_causal_summary.csv`
- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`

### 3. Bootstrap the control-fixed `m04/k04`

```bash
BEST_ROWS_CSV=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m04_k04_controlfix/token_delta_sae_causal_best_rows.csv \
OUT_DIR=/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3b/l18_m04_k04_controlfix/bootstrap \
sbatch slurm/bootstrap_token_delta_sae_causal.template.sbatch
```

## What To Commit Back Or Sync Back

Small files that should come back to Git or be `rsync`ed here:

- `outputs/token_delta_sae_causal_qwen3b/l18_m02_k08/bootstrap/token_delta_sae_bootstrap_summary.csv`
- `outputs/token_delta_sae_causal_qwen3b/l18_m02_k08/bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`
- `outputs/token_delta_sae_causal_qwen3b/l18_m04_k04_controlfix/token_delta_sae_causal_summary.csv`
- `outputs/token_delta_sae_causal_qwen3b/l18_m04_k04_controlfix/TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `outputs/token_delta_sae_causal_qwen3b/l18_m04_k04_controlfix/bootstrap/token_delta_sae_bootstrap_summary.csv`
- `outputs/token_delta_sae_causal_qwen3b/l18_m04_k04_controlfix/bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`

Best additional CSVs to `rsync` here for local inspection:

- `token_delta_sae_causal_best_rows.csv` from `l18_m02_k08`
- `token_delta_sae_causal_best_rows.csv` from `l18_m04_k04_controlfix`

## Branch Decision Rule

After those two checks:

- current provisional decision:
  - the remote token branch is ahead on the matched receiver-day unit
  - and `m02/k08` now has a positive bootstrap interval
- if `m04/k04_controlfix` is also positive, keep the remote token branch as the main challenger and continue it before graph-first escalation
- if `m04/k04_controlfix` collapses badly, keep `m02/k08` as the best remote evidence but treat the branch as narrower and less stable
