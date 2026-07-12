# Anvil Fold-Aligned Remote Detector Submission

## Context

Magnolia commit `7b61717` added the fold-aligned remote detector path after the detector split audit. The Anvil side reviewed the new path and submitted the required full-population scorer plus dependent fold evaluator for both r6.2 and r4.2.

The earlier eval-only detector read should not be used for the final detector table. These jobs score the full JSONL pool and then reconstruct the local leave-one-malicious-user-out folds.

## Anvil Fixes

Pushed Anvil commits:

- `542c92f` Optimize fold-aligned detector scoring on Anvil
- `ace3f3b` Fix detector scoring progress logging

Main changes:

- score base and adapted NLL through one PEFT backbone by disabling the adapter for the base pass
- chunk fp32 cross-entropy with `LOSS_BATCH_SIZE`
- disable KV cache during scoring
- clear CUDA between base/adapted passes and batches
- route scorer wrappers to `cis230270-gpu` / `gpu` by default instead of scarce `cis230270-ai`
- add GPU polling, `/usr/bin/time -v`, and job metadata
- fix progress logging for batch sizes that do not exactly divide `LOG_EVERY`

This preserves detector-score semantics: base NLL is still the frozen Qwen3-8B backbone, adapted NLL is the QLoRA adapter model, and the fold evaluator is unchanged.

## Debug Probes

All probes used A100-SXM4-40GB on `gpu-debug`.

| job | probe | status | batch | loss batch | rows | mean tokens | peak VRAM MiB | elapsed | MaxRSS |
|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| 19101662 | r6.2 normal head | completed | 32 | 2 | 4096 | 254.10 | 13657 | 00:06:59 | 17715504K |
| 19101663 | r4.2 normal head | completed | 32 | 2 | 4096 | 242.82 | 11017 | 00:06:28 | 17567260K |
| 19102035 | r6.2 long-tail | completed | 80 | 1 | 1280 | 948.70 | 36999 | 00:08:20 | 17543336K |
| 19102188 | r6.2 extreme tail | completed | 56 | 1 | 2177 | 1190.81 | 36199 | 00:15:58 | 17709952K |

`BATCH_SIZE=80` was not selected for the full run because the B80 probe max token length was 1199, while the rare r6.2 extreme tail reaches 1658 tokens. `BATCH_SIZE=56`, `LOSS_BATCH_SIZE=1` was selected because it completed the true extreme-tail file with about 36.2 GiB peak VRAM and stable cleanup.

## Submitted Full Jobs

Shared settings:

- partition/account: `gpu` / `cis230270-gpu`
- walltime: `48:00:00`
- GPU request: `--gres=gpu:1`
- CPU RAM: `180G`
- scorer batch: `BATCH_SIZE=56`
- CE chunk: `LOSS_BATCH_SIZE=1`
- GPU polling: `GPU_POLL_SEC=60`

r4.2:

- score job: `19102500`
- dependent fold-eval job: `19102521`
- score output: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r42_fullpop_scores/example_scores.parquet`
- final report dir: `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`

r6.2:

- score job: `19102519`
- dependent fold-eval job: `19102522`
- score output: `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r62_fullpop_scores/example_scores.parquet`
- final report dir: `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`

As of submission, both GPU scorer jobs were pending with reason `Priority`; dependent CPU jobs were pending with reason `Dependency`.

## Monitoring

Useful commands:

```bash
squeue -j 19102500,19102519,19102521,19102522 -o '%.18i %.9P %.28j %.2t %.10M %.10l %.6D %R'
sacct -j 19102500,19102519 --format=JobID,JobName,Partition,State,Elapsed,ReqMem,MaxRSS,AveRSS,ExitCode -P
tail -n 20 /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r42_fullpop_scores/gpu_poll_19102500.csv
tail -n 20 /anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/detector_score_cache/qwen3_8b_r62_fullpop_scores/gpu_poll_19102519.csv
```

