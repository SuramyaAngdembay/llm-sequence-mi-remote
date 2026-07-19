# CLAUDE.md

This file is the compact, repo-local research and operations handoff for the
current CERT-focused TMLR paper in `llm-sequence-mi-remote`.

It is written for a coding/research agent that needs enough context to:

- understand the paper's real claim
- know which results are final-safe and which are superseded
- avoid rerunning invalid protocols
- know where Magnolia work ends and Anvil work begins
- build the paper draft without reconstructing the entire project history

This file is intentionally blunt. It favors audited, paper-safe context over
historical optimism.

## 1. Project Summary

This repo contains the remote `Qwen3-8B` / QLoRA branch of a benign-trained
session-language modeling project for CERT insider-threat benchmarks.

The current paper is **CERT-focused**, not a LANL paper.

Current safe scientific framing:

- benign-trained session LLMs can recover **benchmark-specific causal
  structure** on CERT insider-threat benchmarks
- direct token-mechanism transfer can fail
- native rediscovery can succeed

Current unsafe framing:

- strong detector superiority on CERT
- universal transferable sparse insider-threat circuit
- literal feature-level transfer across datasets

The paper is now primarily a **mechanistic interpretability** paper with a
weaker detector side story.

## 2. Current Paper Claim Map

### Claim A: benign-only remote QLoRA training is valid one-class training

Support:

- `scripts/build_session_jsonl.py`
- `scripts/train_qlora.py`

Status:

- valid
- the July 2026 audits did not find label leakage into training

### Claim B: the remote `Qwen3-8B` branch shows strong mechanistic structure on `r6.2`

Final-safe support:

- `results/qwen3_8b_token_causal/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_token_necessity/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m04_k08_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

Headline branch:

- layer `18`
- latent multiplier `4`
- top-k `8`
- `top5`
- `control5_active`

Status:

- strong
- paper-safe
- same-user exclusion applied

### Claim C: direct remote token-mechanism transfer from `r6.2` to `r4.2` fails

Support:

- `results/qwen3_8b_r42_token_causal/stream_uncapped_v2/RESULTS.md`

Status:

- valid supporting claim
- not the final `r4.2` headline result

### Claim D: `r4.2` has its own native remote token mechanism

Final-safe support:

- `results/qwen3_8b_r42_token_causal/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_r42_token_necessity/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l26_m02_k04_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

Headline branch:

- layer `26`
- latent multiplier `2`
- top-k `4`
- `top5`
- `control5_active`

Status:

- valid
- smaller than `r6.2`
- necessity is partial / context-dependent, not uniformly strong

### Claim E: the remote detector is competitive with local detector baselines

This claim is **not** a safe headline claim anymore.

Only valid detector artifacts:

- `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/FOLD_ALIGNED_DETECTOR_REPORT.md`
- `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/FOLD_ALIGNED_DETECTOR_REPORT.md`

Status:

- scientifically usable
- mixed-to-weak on day PR-AUC versus the stronger Magnolia baselines
- should remain background evidence, not the main story

## 3. Final Safe Claim Discipline

Use:

- benchmark-specific causal structure
- direct transfer can fail
- native rediscovery can succeed
- benign-only remote training is valid

Do not use:

- strong detector-superiority language
- universal mechanism language
- old `detector_metrics/` headline rows
- historical permissive-donor / permissive-match mechanistic rows

## 4. Datasets And Evaluation Population

### CERT `r6.2`

Remote matched session-domain positives:

- `70` positive user-days / receivers
- `4` positive users in the remote example domain

Important caveat:

- the raw labels contain more positives than the matched LC-DAL session domain
- the paper should describe the evaluation population as **matched active
  session user-days**, not all answer-key rows

### CERT `r4.2`

Remote matched session-domain positives:

- `1309` positive user-days / receivers
- `60` positive users in the remote example domain

Same caveat:

- remote evaluation is on the matched LC-DAL session domain, not every answer
  key row

## 5. Split Protocol

Defined in:

- `scripts/build_session_jsonl.py`

Rules:

- positive users -> `split="eval"`
- benign users hashed to `val` with `val_frac=0.10`, otherwise `train`

Generated files:

- `all.jsonl` = all rows
- `train.jsonl` = benign train users only
- `val.jsonl` = benign validation users only
- `eval.jsonl` = `val + eval`

Important:

- `eval.jsonl` is intentionally **not** pure positives
- old detector work broke when this file was scored and then filtered again

## 6. Remote Detector Protocol

### Old broken path

The old `results/*/detector_metrics/` artifacts are audit-only.

Why:

- detector scoring was done from an `example_scores.parquet` extracted from
  `eval.jsonl`
- then `--split eval` was applied again
- this silently dropped the benign validation slice and overstated detector
  quality, especially on `r6.2`

### Final-safe detector path

Use only:

- `scripts/score_adapter_examples.py`
- `scripts/eval_fold_aligned_detector_metrics.py`

Protocol:

- score the full population from `all.jsonl`
- evaluate on the same leave-one-malicious-user-out test-user construction as
  the Magnolia local baselines
- seed `42`
- `800` benign test users per fold
- fixed benign-trained remote model
- not per-fold remote retraining

Headline-safe detector outputs:

- `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`
- `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`

Current read:

- strong ROC / rank behavior
- weak day PR-AUC relative to Deep SVDD / GRU AE / LSTM AE on Magnolia

## 7. Remote Mechanistic Estimands

### Causal patching

Script:

- `scripts/eval_token_delta_sae_causal.py`

Meaning:

- patch sparse token-SAE top features into positive receivers
- compare repair advantage against an active control feature set

Final headline control:

- `control5_active`

Old controls:

- `control3` and older control variants are historical only unless a specific
  report says otherwise

### Necessity

Script:

- `scripts/eval_token_delta_sae_necessity.py`

Meaning:

- ablate top features and compare the harm against ablating active-control
  features

### Same-user exclusion

This is required for final headline mechanistic rows.

Reason:

- earlier pipelines allowed same-user donor / benign matches
- this could inflate repair or ablation results by borrowing a user's own
  benign signature

Final headline mechanistic results must come from `same_user_recovery/`
directories only.

### Reporting fix from 2026-07-18

The causal and necessity summary scripts were corrected to report the
**complete paired receiver/pair contrast** instead of a difference of marginal
means.

Affected scripts:

- `scripts/eval_token_delta_sae_causal.py`
- `scripts/eval_token_delta_sae_necessity.py`
- `scripts/bootstrap_token_delta_sae_causal.py`
- `scripts/bootstrap_token_delta_sae_necessity.py`

This was a summary/statistical reporting fix, not a training or GPU-patching
fix.

Paper-safe rule:

- use the repaired summaries whose estimate matches the bootstrap estimate

## 8. Final Headline Experiment Map

### Remote training and mechanistic runs on Anvil

Primary cluster:

- Purdue Anvil

Main remote branch:

- benign-trained `Qwen3-8B` QLoRA session model

Main remote work performed on Anvil:

- QLoRA training
- detector full-population scoring
- fold-aligned detector evaluation
- token delta extraction
- token-SAE causal patching
- necessity ablation
- native `r4.2` token search
- same-user-excluded recovery reruns
- debug probes and runtime fixes

### Local baselines and local mechanistic comparator on Magnolia

Primary local cluster:

- Magnolia

Main local work performed on Magnolia:

- LC-DAL session feature preparation
- local detector baselines
  - Deep SVDD
  - GRU AE
  - LSTM AE
  - Isolation Forest
- local session-AE mechanistic comparator
- local `r4.2` session-AE mechanistic reruns

Relevant local baseline root outside this repo:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/`

## 9. Final-Safe Result Directories

### Detector

- `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`
- `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`

### Remote mechanistic

- `results/qwen3_8b_token_causal/same_user_recovery/`
- `results/qwen3_8b_token_necessity/same_user_recovery/`
- `results/qwen3_8b_r42_token_causal/same_user_recovery/`
- `results/qwen3_8b_r42_token_necessity/same_user_recovery/`

### Final compare reports

- `results/qwen3_8b_token_causal/strict_compare_remote70_daylevel_l18_m04_k08_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l26_m02_k04_no_same_user/REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md`

### Final policy docs

- `docs/FINAL_TABLE_ROW_POLICY.md`
- `docs/HANDOFF_2026-07-12_PAPER_CONTEXT.md`
- `docs/VALIDITY_AUDIT_2026-07-18_CERT_DATA_AND_MECH.md`

## 10. Do Not Use

Do not use as headline evidence:

- any `results/*/detector_metrics/DETECTOR_METRICS.md`
- permissive-donor / permissive-match mechanistic results superseded by
  `same_user_recovery/`
- historical compare reports unless explicitly regenerated after the recovery
  fixes and tied to the final no-same-user summaries

Do not claim:

- detector superiority
- universal transfer
- identical sparse feature identity across datasets

## 11. Final Audited Result Snapshot

### `r6.2`

Detector, fold-aligned:

- `adapted_nll` day PR-AUC `0.000754631`
- mixed / weak detector story

Causal, same-user-excluded:

- `role = 0.006848`
- `dept_role = 0.006818`
- `project_role = 0.004201`
- `team = n/a`

Necessity, same-user-excluded:

- `project_role = 0.065188`
- `role = 0.062167`
- `dept_role = 0.056603`
- `team = 0.052234`

Interpretation:

- strong mechanistic result survives the repair

### `r4.2`

Detector, fold-aligned:

- `adapted_nll` day PR-AUC `0.0134474`
- stronger than Isolation Forest, weaker than the strongest Magnolia
  baselines on day PR-AUC

Causal, same-user-excluded, audited complete-case estimand:

- `team = 0.001418`
- `role = 0.001112`
- `dept_role = 0.001067`
- `dept = 0.000982`

Necessity, same-user-excluded:

- `dept_role = 0.002922`
- `role = 0.002075`
- `dept = 0.001155`, CI crosses zero
- `team = 0.000662`, CI crosses zero

Interpretation:

- direct transfer failed
- native rediscovery succeeded
- smaller and less uniform than `r6.2`

## 12. Important Technical Caveats

### Representation caveat

The session JSONL builder drops several hyphenated columns because of the
`itertuples(...)._asdict()` path. Specifically:

- `file_n-to_usb1`
- `file_n-from_usb1`
- `file_n-file_act3`
- `file_n-disk1`

This does not invalidate existing runs, but the paper should not claim that
those exact fields were in the LLM text unless the representation is changed
and rerun.

### `r4.2` user-map caveat

The historical `r4.2` user-map generation inherits unsorted `os.listdir()`
traversal. The currently transferred / checksummed map is canonical for all
completed runs. Future rebuilds should either reuse it or sort explicitly.

### `r6.2` causal `team`

The final same-user-excluded `team` row is `n/a` because there is no finite
anomalous-control comparison in that context under the stricter donor rules.
Do not force a number there.

## 13. Repo Layout

Important subtrees:

- `scripts/` - main data, scoring, causal, necessity, comparison, and detector
  evaluation scripts
- `slurm/` - Slurm templates used primarily on Anvil
- `results/` - tracked result bundles and reports
- `docs/` - runbooks, audits, handoffs, and final row policy
- `paper/` - first-draft TMLR package

Most important scripts:

- `scripts/build_session_jsonl.py`
- `scripts/train_qlora.py`
- `scripts/extract_adapter_deltas.py`
- `scripts/score_adapter_examples.py`
- `scripts/eval_fold_aligned_detector_metrics.py`
- `scripts/eval_token_delta_sae_causal.py`
- `scripts/bootstrap_token_delta_sae_causal.py`
- `scripts/eval_token_delta_sae_necessity.py`
- `scripts/bootstrap_token_delta_sae_necessity.py`
- `scripts/compare_remote_token_vs_local_session.py`

## 14. Paper Build

The first draft TMLR package lives in:

- `paper/`

Build locally from repo root:

```bash
bash paper/scripts/build_paper.sh
```

Main TeX file:

- `paper/main.tex`

Current draft output:

- `paper/build/main.pdf`

Overleaf:

- compiler: `pdfLaTeX`
- bibliography: `BibTeX`

Generated tables are already tracked in git:

- `paper/tables/cert_detector_comparison.tex`
- `paper/tables/cert_mechanistic_summary.tex`
- `paper/tables/claim_status.tex`

## 15. SSH / Cluster Guide

### Magnolia

Current shell identity:

- host: `magnolia01`
- user: `srangdembay`

Current project root on Magnolia:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2`

Repo path on Magnolia:

- `/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote`

If your environment already resolves the Magnolia hostname directly, the
typical pattern would be:

```bash
ssh srangdembay@magnolia01
```

But this repo does **not** contain an authoritative external Magnolia login
hostname beyond the local machine identity above. Use your existing cluster SSH
alias or institution-provided login name if `magnolia01` is not directly
reachable from your workstation.

Useful Magnolia commands:

```bash
squeue -u srangdembay
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList -P
```

Known Magnolia GPU facts:

- `gpu001`: `2x P100`, `128G` RAM
- `gpu002`: `4x K80`, `128G` RAM

### Anvil

Authoritative login guidance:

- host: `anvil.rcac.purdue.edu`
- example observed login node: `login03.anvil.rcac.purdue.edu`
- user: `x-sangdembay`

Typical login:

```bash
ssh x-sangdembay@anvil.rcac.purdue.edu
```

Persistent project storage:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/`

Typical repo clone location used during this project:

- `~/cert-qlora-MI/llm-sequence-mi-remote`

Useful Anvil commands:

```bash
squeue -u x-sangdembay
myquota
mybalance
```

Main GPU partitions used in this project:

- `ai` / account `cis230270-ai` -> H100 80GB
- `gpu` or `gpu-debug` / account `cis230270-gpu` -> A100 40GB

Important operational fact:

- compute nodes do **not** have internet
- login nodes do

### SSH config and keys

This repo does **not** track SSH private keys.

This shell currently has no populated `~/.ssh/config` entries to reuse.

So for another agent:

- do not expect repo-managed SSH credentials
- do not expect a checked-in SSH config alias
- rely on user-provided SSH keys / `~/.ssh/config` / agent forwarding

If you need both clusters from one workstation, a minimal user-side SSH config
would normally look like:

```sshconfig
Host anvil
    HostName anvil.rcac.purdue.edu
    User x-sangdembay

Host magnolia
    HostName magnolia01
    User srangdembay
```

But the Magnolia `HostName` line above is only a placeholder based on the local
machine identity. Replace it with the real externally reachable Magnolia login
host if your site uses a different ingress name.

## 16. Current Status For A New Agent

If you are picking this project up cold, start here:

1. Read:
   - `docs/FINAL_TABLE_ROW_POLICY.md`
   - `docs/HANDOFF_2026-07-12_PAPER_CONTEXT.md`
   - `docs/VALIDITY_AUDIT_2026-07-18_CERT_DATA_AND_MECH.md`
2. Treat the detector story as background only.
3. Treat the mechanistic story as the paper center.
4. Use only the final-safe result directories listed above.
5. Build the draft with:
   - `bash paper/scripts/build_paper.sh`

## 17. Bottom Line

This paper is no longer a "better detector" paper.

It is now a narrower, cleaner paper showing:

- benign-trained session LLMs can recover benchmark-specific causal structure
  on CERT insider-threat benchmarks
- direct token-mechanism transfer can fail
- native rediscovery can succeed

That is the scientifically correct center of gravity for this repo as of
2026-07-18.
