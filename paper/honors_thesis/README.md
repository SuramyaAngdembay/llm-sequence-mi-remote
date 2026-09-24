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
| Chapter III, setup | CTMC data-preparation section removed; one section added (language-model data) |
| Chapter IV, language-model surprisal and its decomposition | new. Section 4.3 explains how the next-token probability is produced (context vector, dot-product scores, softmax, surprisal as log-sum-exp minus the actual token's score). Section 4.4, on why $-\log p$ measures surprise, and the whole-day negative log-likelihood subsection moved here from the removed CTMC chapter. Section 4.5 derives that minimizing surprisal learns the benign frequencies and shows low-rank adaptation as a rank-16 correction |
| Chapter V, interpretable dimensionality reduction of the fine-tuned model | new; the sparse autoencoder is presented as a sparse, overcomplete relative of PCA |
| References | ten entries added: nine from `paper/references.bib`, plus Shannon (1948) |

The removed CTMC chapter and its data-preparation section are preserved verbatim
in `ctmc_chapter_removed.tex`.

Compiles cleanly with TeX Live 2020 (70 pages, two passes, no errors, no
undefined references). It needs the seven PCA figure files alongside it. The one
overfull line left is in the original PCA chapter. `thesis_mechinterp_preview.pdf`
was built with placeholder grey boxes in place of those figures.

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
