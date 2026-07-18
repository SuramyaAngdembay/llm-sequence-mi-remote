# CERT Data And Mechanistic Validity Audit - 2026-07-18

This note records the Anvil-side audit of the CERT r6.2/r4.2 data path,
Magnolia transfer context, detector evaluation, and token-SAE mechanistic
summaries after the July 2026 recovery runs.

## Scope

Audited:

- CERT r6.2 and r4.2 README / answer-key context copied from Magnolia.
- Magnolia dataset-creation and local-baseline scripts copied into Anvil
  scratch for review.
- Anvil transfer packages, labels, user maps, JSONL builders, detector metrics,
  causal patching, and necessity scripts.
- Final same-user-excluded Qwen3-8B recovery artifacts.

Scratch audit root:

`/anvil/scratch/x-sangdembay/cert_validity_audit_20260718`

Magnolia context mirror:

`/anvil/scratch/x-sangdembay/cert_validity_audit_20260718/magnolia_context`

## External Context

Primary public sources checked:

- CMU KiltHub CERT Insider Threat Test Dataset:
  https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
- LC-DAL CERT feature extraction repository:
  https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets
- Activation patching best-practices paper:
  https://arxiv.org/abs/2309.16042
- Activation patching interpretation guidance:
  https://arxiv.org/html/2404.15255v1
- Causal scrubbing:
  https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing
- Sparse-autoencoder / dictionary-learning feature extraction:
  https://transformer-circuits.pub/2023/monosemantic-features

The project framing is directionally consistent with current mechanistic
interpretability practice if the paper states the claim narrowly: these are
benchmark-specific activation-patching and ablation results under a declared
intervention/control protocol, not a complete circuit proof or universal
insider-threat mechanism.

## Copied Magnolia Context

Files copied or verified from Magnolia include:

- r6.2 README files:
  - `InsiderThreatDetection/r6.2/README.md`
  - `InsiderThreatDetection/r6.2/SEI_Insider_README.txt`
  - `InsiderThreatDetection/r6.2/extracted/r6.2/readme.txt`
  - `InsiderThreatDetection/r6.2/extracted/answers/readme.txt`
- r4.2 README files:
  - `InsiderThreatDetection/r4.2/readme.txt`
  - `InsiderThreatDetection/r4.2/answers/readme.txt`
- LC-DAL extraction scripts:
  - `InsiderThreatDetection/r6.2/lcdal-r62-full/extract_stage/r6.2/feature_extraction.py`
  - `insider_threat/r4.2/feature_extraction.py`
- Local baseline / preparation scripts and reports:
  - `prepare_lcdal_session_features_r62.py`
  - `ctmc_session_sequence_baselines_r62.py`
  - `ctmc_daily_autoencoder_mech_train_r62.py`
  - `R62_SESSION_LCDAL_SEQUENCE_COMPARE_REPORT.md`
  - `R42_SESSION_LCDAL_SEQUENCE_COMPARE_REPORT.md`
  - `R62_SESSION_LCDAL_AUTOENCODER_MECH_CLEAN_REPORT.md`
  - `R42_SESSION_LCDAL_AUTOENCODER_MECH_CLEAN_REPORT.md`
- Transfer manifests and maps:
  - `artifacts/transfer_package/manifest.json`
  - `artifacts/transfer_package_r42/manifest.json`
  - `artifacts/transfer_package_r42/sessionr4.2_user_map.csv`

## Checks That Passed

Transfer checksums passed on Anvil:

- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/sha256sums.txt`
- `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/data/r4.2/sha256sums.txt`

Magnolia-vs-Anvil byte matches:

- r6.2 `labels_daily.parquet`
- r4.2 `labels_daily.parquet`
- r6.2 `sessionr6.2_user_map.csv`
- r4.2 `sessionr4.2_user_map.csv`

Fresh CPU rebuild from Anvil transfer shards:

- r6.2 rebuild:
  `/anvil/scratch/x-sangdembay/cert_validity_audit_20260718/rebuild_anvil_r62`
- r4.2 rebuild:
  `/anvil/scratch/x-sangdembay/cert_validity_audit_20260718/rebuild_anvil_r42`

Both rebuilds were byte-identical to production for:

- `all.jsonl`
- `train.jsonl`
- `val.jsonl`
- `eval.jsonl`
- `example_metadata.csv`

The build summaries also matched after normalizing the output directory path.

## Dataset Domain Caveat

The remote LLM branch evaluates matched LC-DAL active session user-days, not
every row in the CERT answer key.

Current matched session domain:

- r6.2: `70` positive user-days across `4` users in examples.
- r4.2: `1309` positive user-days across `60` users in examples.

Label files contain more positive answer-key rows than the session-domain
examples:

- r6.2 labels: `99` positive rows across `5` users.
- r4.2 labels: `1883` positive rows across `70` users.

This is not an Anvil transfer bug. Magnolia local session-prep stats report the
same matched-label counts. The paper should state the evaluation population as
matched active session user-days under the LC-DAL session feature domain.

## Split And Detector Validity

Remote JSONL split logic remains valid:

- positive users are assigned to `split="eval"`;
- QLoRA training uses only `train.jsonl`;
- validation uses only `val.jsonl`;
- `eval.jsonl` intentionally contains `val + eval`, not only positives.

The fold-aligned detector replacement path is the final-safe detector path:

- r6.2:
  `results/qwen3_8b_token_causal/detector_metrics_fold_aligned/`
- r4.2:
  `results/qwen3_8b_r42_token_causal/detector_metrics_fold_aligned/`

The fold constructor in `scripts/eval_fold_aligned_detector_metrics.py` matches
the Magnolia local AE test-fold constructor for held-out positive user and
sampled benign test users. The local AE additionally trains a per-fold local
model on the non-test benign users; the fixed remote detector does not require
that training fold.

Old `results/*/detector_metrics/` outputs remain audit-only and should not be
used as headline detector rows.

## Representation Caveats

`build_session_jsonl.py` lists several hyphenated session columns:

- `file_n-to_usb1`
- `file_n-from_usb1`
- `file_n-file_act3`
- `file_n-disk1`

Because the original builder uses `DataFrame.itertuples(...)._asdict()`, pandas
renames invalid Python identifiers and these columns are dropped from the
serialized session dict. The fast builder intentionally reproduces this behavior
for byte-faithful output.

This is not a leakage bug and does not invalidate completed runs, but the paper
should not claim that those four exact file-action fields were included in the
LLM text representation unless the representation is changed and the affected
training/eval artifacts are rerun.

r4.2 user-map generation in `prepare_transfer_package.py` inherits an
unsorted `os.listdir()` traversal of LDAP files. The current transferred map is
checksummed and canonical for all completed Anvil runs. Any future fresh
official CERT rebuild should either reuse the canonical transferred map or sort
the source files and explicitly compare the resulting map before use.

## Mechanistic Validity

Final paper-eligible remote mechanistic rows should use only the
same-user-excluded recovery directories:

- r6.2 causal:
  `results/qwen3_8b_token_causal/same_user_recovery/`
- r6.2 necessity:
  `results/qwen3_8b_token_necessity/same_user_recovery/`
- r4.2 causal:
  `results/qwen3_8b_r42_token_causal/same_user_recovery/`
- r4.2 necessity:
  `results/qwen3_8b_r42_token_necessity/same_user_recovery/`

Candidate-row audit confirmed:

- same-user donor/match rows are zero in the final recovery artifacts;
- `control5_active` controls are not inert in the final recovery artifacts;
- r6.2 causal `team` has no finite same-user-excluded anomalous-control row and
  should remain `n/a`;
- r4.2 causal has incomplete anomalous donor support, especially for `team`.

## Reporting Fix Found During This Audit

The causal and necessity summary scripts previously computed top-vs-control
advantages as differences of marginal means. For r4.2 causal this was
potentially misleading because same-user exclusion left fewer anomalous donor
rows than benign donor rows.

Fix applied on 2026-07-18:

- `scripts/eval_token_delta_sae_causal.py`
- `scripts/eval_token_delta_sae_necessity.py`
- `scripts/bootstrap_token_delta_sae_causal.py`
- `scripts/bootstrap_token_delta_sae_necessity.py`

The reported advantage now uses complete paired receiver/pair contrasts:

- causal: complete top/control and benign/anomalous donor values per receiver;
- necessity: complete top/control and positive/benign values per pair.

Regenerated final recovery summaries now satisfy:

- `token_delta_sae_*_summary.csv` top-vs-control value equals the bootstrap
  `estimate`;
- bootstrap reports expose `n_complete_receivers` or `n_complete_pairs`;
- r4.2 causal `team` is `1301` available receivers, `1052` complete receivers,
  estimate `0.001418`.

This is a reporting/statistical summary fix only. It does not change model
training, SAE fitting, raw best rows, raw candidate rows, or GPU patching
outputs.

## Remaining Limitations

The current audit did not redownload the full official KiltHub CERT archives and
rerun LC-DAL extraction from compressed raw event logs. The public KiltHub page
lists the full download as tens of GiB, and a complete extraction would be a
separate scratch-space job.

Given the checks above, the current high-signal validation is:

- Magnolia labels/maps match Anvil byte-for-byte;
- Anvil transfer shard checksums pass;
- JSONL rebuild from those raw LC-DAL session shards is byte-identical to
  production;
- final same-user recovery artifacts pass exclusion and active-control support
  checks.

If a stricter archival validation is required, the next step is a scratch-only
download of the official r6.2/r4.2/answers archives from KiltHub, rerun LC-DAL
feature extraction, and compare the produced session CSV/user map/labels to the
canonical transferred artifacts before any scientific reruns.

## Verdict

No evidence found that the Qwen3-8B training runs or final same-user-excluded
mechanistic recovery runs are invalid due to data leakage, split leakage, stale
detector scoring, or same-user donor contamination.

The paper should be narrowed and precise:

- detector rows: use fold-aligned full-population scores only;
- mechanistic rows: use same-user-excluded recovery rows only;
- mechanism claim: benchmark-specific causal evidence under the token-SAE
  intervention protocol;
- dataset claim: matched active session user-days, not all CERT answer-key
  incidents;
- representation claim: do not claim inclusion of the four hyphenated
  file-action fields unless rerun with a changed serializer.
