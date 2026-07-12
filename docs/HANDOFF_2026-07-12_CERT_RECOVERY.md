# CERT Recovery Checklist

This handoff captures the minimum recovery plan after the July 11-12 detector-metrics audit.

Use this order only.

## Paper Rules

- Do **not** use the old `results/*/detector_metrics/` artifacts as headline detector rows.
- Use only the fold-aligned remote detector benchmark for the final detector table.
- Keep the current `8B` training/frontier/native-search results; no retraining is required.
- Treat same-user-excluded reruns as robustness audits of the existing `8B` mechanistic headline branches.

## Priority Order

1. Finish fold-aligned remote detector runs.
2. Run same-user-excluded `8B` robustness reruns.
3. Regenerate cited local-vs-remote compare reports.
4. Freeze the CERT paper.

## Step 1: Fold-Aligned Detector Table

These are already the correct replacement path for the old detector artifacts.

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main

bash scripts/submit_qwen3_8b_r62_fold_aligned_detector_anvil.sh
bash scripts/submit_qwen3_8b_r42_fold_aligned_detector_anvil.sh
```

Use only these outputs for the remote detector table:
- `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`
- `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`

## Step 2: Same-User-Excluded 8B Robustness Reruns

These are the only mechanistic reruns required for recovery.

### 2A. r6.2 8B causal active-control, same-user excluded

```bash
bash scripts/submit_qwen3_8b_r62_active_control_no_same_user_anvil.sh
```

### 2B. r6.2 8B necessity, same-user excluded

```bash
bash scripts/submit_qwen3_8b_r62_necessity_no_same_user_anvil.sh
```

### 2C. r4.2 8B native causal active-control, same-user excluded

```bash
bash scripts/submit_qwen3_8b_r42_native_active_control_no_same_user_anvil.sh
```

### 2D. r4.2 8B native necessity, same-user excluded

```bash
bash scripts/submit_qwen3_8b_r42_necessity_no_same_user_anvil.sh
```

## Step 3: Regenerate Cited Compare Reports

The compare script is now explicit about the local target/control assumptions.
Regenerate any cited reports after the remote recovery results land.

Especially regenerate if cited:
- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_*`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_*`
- `results/qwen3b_pilot/strict_compare_remote70_daylevel_controlfix/*`

## What Does Not Need To Be Rerun

- no `3B` retraining
- no `8B` retraining
- no `r4.2` native search rerun
- no SAE frontier rerun
- no LANL work before the CERT story is clean
- no same-hardware Magnolia reruns

## Decision Rule

If the fold-aligned detector rows are reasonable and the same-user-excluded `8B` runs stay broadly positive, freeze the CERT story and write.
