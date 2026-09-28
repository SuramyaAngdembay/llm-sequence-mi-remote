# Honors thesis: dimensionality reduction approaches to insider threat detection

`thesis_mechinterp.tex` is the honors thesis, reframed around dimensionality
reduction, with the language-model and interpretability work added. Its title is
*Dimensionality Reduction Approaches to Insider Threat Detection: From Principal
Components to Interpretable Sparse Features*. Search the source for `NEW`,
`CHANGED` and `MOVED` to see every edit.

| part | status |
|---|---|
| Title, abstract, keywords, abbreviations | changed; CTMC removed from keywords and abbreviations |
| Chapter I, introduction | unchanged placeholder |
| Chapter II, PCA | **unchanged, verbatim** |
| Chapter III, setup | written in full 2026-09-28: the dataset as users, days and events; the insider case (scenarios, labels, base rate, one-class framing); PCA preparation and result, filled from the Magnolia run of 2026-09-28 (one `[To complete]` note remains: the label alignment, pending a decision); language-model data (sessions, serialization, training settings, user-disjoint splits, evaluation pool); five design rules. A visible note flags the label misalignment found the same day |
| Chapter IV, language-model surprisal and its decomposition | new. Section 4.3 explains how the next-token probability is produced (context vector, dot-product scores, softmax, surprisal as log-sum-exp minus the actual token's score). Section 4.4, on why $-\log p$ measures surprise, and the whole-day negative log-likelihood subsection moved here from the removed CTMC chapter. Section 4.5 derives that minimizing surprisal learns the benign frequencies and shows low-rank adaptation as a rank-16 correction |
| Chapter V, interpretable dimensionality reduction of the fine-tuned model | new; the sparse autoencoder is presented as a sparse, overcomplete relative of PCA |
| References | ten entries added: nine from `paper/references.bib`, plus Shannon (1948) |

The removed CTMC chapter and its data-preparation section are preserved verbatim
in `ctmc_chapter_removed.tex`.

Compiles cleanly with TeX Live 2020 (58 pages as of 2026-09-28, two passes, no
errors, no undefined references, no overfull lines). It needs the seven PCA figure files alongside it. The one
overfull line left is in the original PCA chapter. `thesis_mechinterp_preview.pdf`
is built in graphicx draft mode, with placeholder frames in place of those
figures (rebuilt 2026-09-25 after the Chapter V corrections).

Every number in Chapters IV and V was checked against its source file before
it was written:

| number | source |
|---|---|
| user AUC 0.65 / 0.86 / 0.47 / 0.22 and intervals | `results/score_decomposition/r42_headline/batch1_recompute/score_view_summary.csv` |
| +0.21 [+0.16, +0.27]; 51 / 3 / 6 users | `.../batch1_recompute/score_view_contrasts.csv` |
| attribution table, token shares | `results/feature_attribution/feature_attribution_r42/FEATURE_TOKEN_ATTRIBUTION.md` |
| held-out causal 0.00076 [0.00035, 0.00121], 21 of 29 | `results/qwen3_8b_r42_token_causal/confirmation/RESULTS.md` |
| layer 26, 8,192 features, k = 4, 2,000,000 training vectors | r4.2 frontier summary on Anvil |
| feature and control selection rules | `scripts/sae_core.py` (`row_gap`, `_choose_active_low_gap_ids`) |
| worked-example arithmetic, LoRA parameter counts | computed exactly, not by hand |
| counts 80 / 15 / 5 after `n_logon=` in Section 4.5 | illustrative, not data; labelled as such in the text |
| τ = 0.00076 [0.00035, 0.00121], 605 complete days, 21 of 29 users (dept) | `results/qwen3_8b_r42_token_causal/confirmation/RESULTS.md` (cluster interval) |
| donor advantages 0.00063 and −0.00012; benign-only contrast 0.0028 | `.../confirmation/token_delta_sae_causal_summary.csv`, dept row, recomputed 2026-09-24 |
| control activity 0.36% versus selected 0.18% | the staged discovery-split ranking, `pkg4_share/frontier/layer_26/m02_k04/delta_sae_top_features.csv` on Anvil |
| feature 3673's strongest activations on `_dur` | `results/feature_attribution/feature_attribution_r42/FEATURE_TOKEN_ATTRIBUTION.md` |
| example user-day and its token counts (213 tokens; 54 profile, 158 behavior scored) | r4.2 `eval.jsonl` record `AAF0535:7`, tokenized with the adapter's tokenizer and labelled by `scripts/token_class_decomposition.py` |

**Corrections of 2026-09-24** (see `docs/REVIEW_2026-09-24_ISSUE_LEDGER.md`,
entries H1 and H2): the causal-test section now defines the published
difference-in-differences endpoint, with its best-candidate step, and the
interval is attached only to it. Where features activate, what they represent
and which predictions their edit changes are kept apart. The controls are
described as threshold-selected, not activity-matched. The withdrawn TWOS seed
reversal is no longer cited, and the abstract no longer anticipates the pending
per-class result.

**Chapter III sources (checked 2026-09-28).** Section 3.3 (PCA) numbers: `results/pca_reconstruction_r42_2026_09_28/pca_reconstruction_r42.json`, explained in that folder's README.

| number | source |
|---|---|
| 1,000 users, 470,611 sessions, session days 5–505 (2 January 2010 to 17 May 2011) | r4.2 session shards; `results/label_alignment_2026_09_28/label_alignment_r42.json` |
| 70 insiders (30, 30, 10) and the scenario text | CERT `answers.tar.bz2` (`insiders.csv`, `scenarios.txt`), downloaded from the KiltHub record 12841247 |
| 1,883 labelled malicious user-days; 187, 1,676 and 20 by scenario | `labels_daily.parquet` (r4.2) |
| 330,295 user-days; splits 851 / 79 / 70 users; 287,827 / 27,026 / 15,442 days | `pkg4_share/session_jsonl_r42/example_metadata.parquet` |
| sessions-per-day table; `project=na` on every day; longest texts about 1,300 tokens (1,325 among the 200 longest by characters) | the same metadata, `all.jsonl`, and the adapter's tokenizer |
| evaluation pool 40,519 days, 139 users, 1,309 malicious; 29 / 30 / 1 insiders and 139 / 1,168 / 2 days by scenario | the same metadata joined with `insiders.csv` |
| rank 16, α 32, dropout 0.05, seven projections; 1 epoch, 22 × 4 GPUs, learning rate 1.5e-4 cosine, 3% warm-up; 3,271 steps | `pkg4_share/adapter/adapter_config.json`, `training_args.bin`, `checkpoint-3271` |
| the label misalignment note | `docs/LABEL_ALIGNMENT_2026-09-28.md` |
| window labels (1,883 days) vs event-day labels (966; 85 / 861 / 20) | `labels_daily.parquet` vs `insiders.csv` windows; `dayr4.2.csv` `insider` column |
| PCA: n = 330,452, 502 features, 193 constant in training so p = 309 (299 without profile; `b_unit` constant), 287,961 training days, q = 71 (63), 95.1% | `results/pca_reconstruction_r42_2026_09_28/pca_reconstruction_r42.json` (Magnolia job 576348) |
| PCA: medians 6.6 / 17.0, 6.4% vs 0.5% above training q99; user AUC 0.75 [0.67, 0.83], 0.76, 0.78 (70); thresholds 0.81 / 0.75 / 0.78; raw constant columns 99.7% and 0.80 [0.73, 0.87] | the same file |
| Figure 3.1 (user-level ROC of the PCA detector; 30 of 60 insiders, 12 of 79 benign at the marked threshold); Table 3.2 feature counts by group (502 / 309); Table 3.3 | `pca_reconstruction_r42.json` from job 576351 (`user_max_error`, `columns`); the curve's corners are computed in the script that wrote the figure and checked against `user_auc` |
| leak, hacking and supervisor-PC columns; pc code 3 = supervisor's PC | the same file; `InsiderThreatDetection/r4.2/feature_extractor.py:399` on Magnolia |

**Restructure of 2026-09-28 (advisor feedback, Dr. Tian).**

- **DFS and GMM removed everywhere:** the abstract, abbreviations, Chapter III and two references.
  The PCA part now uses PCA, the inverse PCA transformation $\widehat{X} = ZU_q^\top +
  \mathbf{1}\bar{\mathbf{x}}^\top$ and the reconstruction error, with the per-observation error
  and the MSE (Chapter II, Section 2.3).
- **Mini headings removed** in Chapters II to V; only numbered sections remain. Chapter I's
  template headings are untouched placeholders.
- **Step-by-step details condensed:**
  - Chapter II: the seven-step PCA example is now one worked example with a table.
  - Chapter IV: the four-step softmax walk-through, the coin-flip and count examples, and the
    six-step decomposition example.
  - Chapter V: the sparse-coding and attribution toy examples. The causal test is now a numbered
    six-step list.
- **Kept:** every definition, derivation and reported result. The Chapter IV and V finding
  sections are unchanged apart from their subsection headings.
- **Still to complete** (bold `[To complete: ...]` in the text):
  - the PCA feature table, $n$, $p$ and $q$;
  - whether PCA is fit on benign rows only;
  - which design rules the PCA experiment follows;
  - the label-alignment note.
