# Anvil Audit Notes - 2026-07-12

Context: reviewed Magnolia commits `1fc6fa4` and `f86ccba` after pulling from
`71a23d0` to `f86ccba`. This note records the Anvil-side critical review before
staging any additional GPU recovery runs.

## Reviewed Files

- `docs/HANDOFF_2026-07-12_PAPER_CONTEXT.md`
- `docs/HANDOFF_2026-07-12_CERT_RECOVERY.md`
- `scripts/eval_token_delta_sae_causal.py`
- `scripts/eval_token_delta_sae_necessity.py`
- `slurm/eval_token_delta_sae_causal.template.sbatch`
- `slurm/eval_token_delta_sae_necessity.template.sbatch`
- `scripts/submit_qwen3_8b_r62_active_control_no_same_user_anvil.sh`
- `scripts/submit_qwen3_8b_r62_necessity_no_same_user_anvil.sh`
- `scripts/submit_qwen3_8b_r42_native_active_control_no_same_user_anvil.sh`
- `scripts/submit_qwen3_8b_r42_necessity_no_same_user_anvil.sh`

## Literature Cross-Check

Primary sources checked:

- Activation patching best practices:
  https://arxiv.org/abs/2309.16042 and https://arxiv.org/html/2404.15255v1
- Causal tracing / model editing:
  https://rome.baulab.info/
- Causal abstraction / interchange interventions:
  https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf
- Causal scrubbing:
  https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing
- Sparse autoencoders for interpretable LM features:
  https://arxiv.org/abs/2309.08600

The paper-context memo is directionally consistent with these practices:

- It separates exploratory discovery from confirmatory reruns.
- It distinguishes sufficiency-style patching from necessity-style ablation.
- It requires active controls for headline mechanistic rows.
- It treats detector benchmarking separately from mechanistic claims.
- It downgrades transfer claims instead of asserting a universal mechanism.
- It requires same-user exclusion for final mechanistic rows after the audit
  found nontrivial same-user donor/match rates.

The main caution from the literature is that activation patching results depend
strongly on the intervention distribution, metric, and control construction.
Therefore the final paper should frame these as benchmark-specific causal
evidence under a stated protocol, not as a complete circuit proof or a universal
insider-threat mechanism.

## Repair Review

The same-user repair is conceptually valid.

For causal patching, the patch now:

- carries `user_id` into `build_candidate_pairs`
- removes same-row anomalous donors as before
- optionally removes all donors with the receiver's `user_id`
- exposes this through `--exclude-same-user-donors`
- records the flag in the markdown report

For necessity, the patch now:

- carries `user_id` into `build_receiver_pairs`
- optionally removes benign matches with the positive receiver's `user_id`
- exposes this through `--exclude-same-user-matches`
- records the flag in the markdown report

Static checks:

- `bash -n` passed for the four new no-same-user launchers, updated bundle
  launchers, and updated Slurm templates.
- `py_compile` passed under
  `/anvil/projects/x-cis230270/x-sangdembay/conda_envs/cert-qlora-qwen3/bin/python`
  for the causal, necessity, fold-detector, and score scripts.

Smoke test:

- A synthetic user/donor pool confirmed that the new causal and necessity flags
  exclude same-user donor/match rows while preserving valid different-user
  alternatives.

## Current Go/No-Go

Scientific verdict: agree with the recovery plan, with caveats.

The minimum recovery sequence in
`docs/HANDOFF_2026-07-12_CERT_RECOVERY.md` is the right path:

1. Finish fold-aligned remote detector runs.
2. Run only the four same-user-excluded 8B robustness reruns.
3. Regenerate cited local-vs-remote compare reports.
4. Freeze the CERT paper if signs remain stable.

Do not launch extra exploratory scope before those complete.

Operational gate before new GPU submissions:

- `myquota` still reports project file-count usage at 100%.
- This can cause false failures when jobs create output files.
- Reduce project inode pressure before launching the four no-same-user jobs.

## Live Status At Review

Checked at approximately 2026-07-12 02:33 EDT.

- `19102500` r4.2 fold-aligned scoring: running on `gpu`, latest progress
  `159768 / 330295`.
- `19105515` r6.2 fold-aligned scoring: running on `gpu`, latest progress
  `32816 / 1393297`.
- Dependent fold-eval jobs remain pending on `afterok`.

## Next Anvil Actions

1. Keep monitoring the two fold-aligned detector scorers.
2. Before launching same-user robustness jobs, free project file-count quota or
   explicitly verify new file creation works in the target output roots.
3. After detector scoring/eval finishes, commit the fold-aligned detector
   results.
4. Then submit the four no-same-user robustness launchers, preferably after a
   small debug/probe if quota cleanup touched paths or if the queue/hardware
   state changed.
5. Regenerate compare reports only after the replacement mechanistic outputs
   are available.

## Debug Probe Timeout Audit

Checked at approximately 2026-07-12 16:50 EDT.

The first r6.2 same-user recovery probes did not measure useful A100 VRAM:

- `19120013` causal probe timed out at 30 minutes, `MaxRSS=10088220K`.
- `19120014` necessity probe timed out at 30 minutes, `MaxRSS=9133536K`.
- Both GPU poll logs stayed at `0 MiB` for the full probe.

Interpretation: this was not a GPU OOM and not evidence that the run only needs
9-10 GiB. The probes were stuck before model scoring, in CPU-side token-cache
loading. For r6.2, the layer-18 token-delta cache contains 278 chunk files and
about 319 GiB. The old loader scanned every layer chunk twice even when
`MAX_RECEIVERS` / `MAX_PAIRS` made the probe require only a small subset of
examples.

Optimization fix:

- use `extract_summary.json` and `chunk_manifest.csv` as a chunk index when
  `keep_examples` is known
- load only chunk files whose example ranges can contain requested examples
- retain the old full-scan fallback when the manifest is unavailable
- keep the same receiver sample, donor matching, same-user exclusion, feature
  sets, top/control sets, alpha grid, and metric

This is an I/O optimization only. It does not change the causal estimand.

External references checked:

- PyTorch `torch.load(..., mmap=True)` supports memory-mapped loading rather
  than eagerly loading all storages.
- PyTorch data-loading guidance identifies GPU idle time from slow input
  pipelines as a throughput problem.
- Apache Arrow/Parquet row-group statistics are the analogous tabular pattern:
  skip irrelevant chunks instead of decoding every group.
- Zarr uses chunked N-dimensional arrays for selective array access.

## Corrected Probe Submission

Checked at approximately 2026-07-12 17:05 EDT.

Ignore these first post-fix submissions:

- `19122506` causal probe: canceled after 8 minutes 21 seconds.
- `19122507` necessity probe: canceled before running.

Reason: the manual `sbatch --export=...` command encoded comma-containing
values such as `CONTEXT_MODES=team,role,project_role,dept_role` directly in the
export list. Slurm split those commas, and `job_19122506.json` showed the
malformed value `context_modes: "team"`. Those jobs should not be interpreted
as valid probes or resource measurements.

Corrected r6.2 chunk-index probes were resubmitted using shell environment
exports plus `--export=ALL`, preserving comma-containing values:

- `19122902` causal probe, `gpu-debug`, pending at check time.
- `19122903` necessity probe, `gpu-debug`, pending at check time.

Output root:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/same_user_recovery_debug_probes_chunkidx_v2`

Current concurrent recovery job status at the same check:

- `19105515` r6.2 fold-aligned detector scorer is still running on `gpu`
  node `g004`, elapsed 15 hours 32 minutes.
- `19105516` fold detector eval remains pending on dependency.

## Causal Probe Timeout Diagnosis

Checked after `19122902` timed out.

The corrected causal probe preserved the intended context modes, but still
timed out before reaching GPU work:

- job `19122902`
- `MAX_RECEIVERS=16`
- `MAX_CANDIDATE_DONORS=16`
- GPU poll stayed at `0 MiB`
- no causal report/CSV files were written

The reason is probe geometry, not CUDA memory and not malformed arguments.
With same-user exclusion enabled, the causal pairing step produced 1,031 needed
receiver/donor examples spread across `132/278` layer-18 token-cache chunks.
Those selected chunks total about `152.1 GiB`; the current loader can touch the
selected chunks twice before model scoring. That means the 30-minute
`gpu-debug` walltime can be exhausted in pre-GPU token-cache I/O.

The matching necessity probe completed because its `MAX_PAIRS=16` setting
required only 106 examples and selected `51/278` chunks.

Operational fix for debug/profiling:

- run future Python evaluator jobs unbuffered so timeout logs preserve stage
  progress
- make the debug causal probe use a smaller donor fanout by default
  (`MAX_CANDIDATE_DONORS_PROBE=2`)
- keep full recovery run settings separate from debug/profiling settings

## Causal I/O Optimization Follow-Up

Checked after the `bs128` causal micro-probe.

The `bs128` causal probe completed, but peak VRAM stayed low:

- job `19136322`
- `BATCH_SIZE=128`
- `PATCH_CHUNK_SIZE=128`
- peak GPU memory about `11.4 GiB`
- active average GPU memory about `9.2-9.4 GiB`

This means the immediate SU risk is not GPU capacity. The risk is paying A100
time for token-cache extraction and low-utilization setup.

Full r6.2 same-user-excluded causal footprint estimate:

- `MAX_RECEIVERS=0` is equivalent to all 70 positive receivers for this branch
- candidate pairs: `5176`
- needed receiver/donor examples: `4056`
- selected layer-18 token-cache chunks: `139/278`
- selected chunk file size: about `161.1 GiB`

The loader was updated to slice contiguous token rows for requested example IDs
from each selected chunk instead of converting/filtering whole chunk tensors.
This is an I/O/runtime optimization only: receiver sampling, donor matching,
same-user exclusion, feature sets, control sets, alpha grid, and scoring are
unchanged.

## Completed Recovery Status

Checked after `19105515`, `19105516`, `19136579`, and `19138934` completed.

r6.2 fold-aligned detector:

- scoring job `19105515` completed in `1-12:25:52`
- fold eval job `19105516` completed in `00:00:36`
- committed output directory:
  `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`
- adapted detector summary:
  - day PR-AUC: `0.000754631`
  - day ROC-AUC: `0.953157`
  - user PR-AUC: `0.053704`
  - user ROC-AUC: `0.97125`
  - heldout user rank mean: `24.0`

VRAM/runtime probes:

- necessity `bs96` probe `19136579` completed in `00:06:20`
  - output root:
    `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/same_user_recovery_vram_probes/r62_necessity_l18_m04_k08_no_same_user_probe_bs96`
  - peak GPU memory: `15377 MiB`
  - active-average GPU memory: about `10.2-10.7 GiB`
- causal `bs128`, `MAX_CANDIDATE_DONORS=16` row-select probe `19138934`
  completed in `00:18:42`
  - output root:
    `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/same_user_recovery_vram_probes/r62_causal_l18_m04_k08_no_same_user_probe_bs128_donors16_rowselect`
  - selected token rows: `308861`
  - candidate rows: `9608`
  - peak GPU memory: `23477 MiB`
  - active-average GPU memory: about `16.2-16.7 GiB`

Decision implication:

- row-select loading fixed the causal pre-GPU timeout problem for the broader
  donor fanout probe
- VRAM remains below the requested 30 GiB average target, so further GPU
  utilization work should focus on larger scoring batches or evaluator
  batching changes
- the immediate full-run risk is now less about OOM and more about total SU
  cost versus remaining `cis230270-gpu` balance
