# Honors thesis: mechanistic-interpretability chapters

`thesis_mechinterp.tex` is the honors thesis with the language-model and
interpretability work added. Search the source for `NEW` and `CHANGED` to see
every edit.

| part | status |
|---|---|
| Title, abstract, keywords, abbreviations | changed |
| Chapter II, PCA | **unchanged, verbatim** |
| Chapter III, CTMC | original text unchanged; five sections added, including 3.6 on why $-\log p$ measures surprise |
| Chapter IV, setup | one section added (language-model data) |
| Chapter V, language-model surprisal and its decomposition | new; Section 5.3 explains in four steps how the probability is produced: context vector, dot-product scores, softmax, and surprisal as log-sum-exp minus the actual token's score |
| Chapter VI, mechanistic interpretability | new |
| References | ten entries added: nine from `paper/references.bib`, plus Shannon (1948) |

Compiles cleanly with TeX Live 2020 (two passes, no errors, no undefined
references). It needs the seven PCA figure files alongside it. The one overfull
line left is in the original PCA chapter. `thesis_mechinterp_preview.pdf` was
built with placeholder grey boxes in place of those figures.

Every number in Chapters V and VI was checked against its source file before
it was written:

| number | source |
|---|---|
| user AUC 0.65 / 0.86 / 0.47 / 0.22 and intervals | `results/score_decomposition/r42_headline/batch1_recompute/score_view_summary.csv` |
| +0.21 [+0.16, +0.27]; 51 / 3 / 6 users | `.../batch1_recompute/score_view_contrasts.csv` |
| attribution table, token shares | `results/feature_attribution/feature_attribution_r42/FEATURE_TOKEN_ATTRIBUTION.md` |
| held-out causal 0.00076 [0.00035, 0.00121], 21 of 29 | `results/qwen3_8b_r42_token_causal/confirmation/RESULTS.md` |
| layer 26, 8,192 features, k = 4, 2,000,000 training vectors | r4.2 frontier summary on Anvil |
| feature and control selection rules | `scripts/sae_core.py` (`row_gap`, `_choose_active_low_gap_ids`) |
| worked-example arithmetic | computed exactly, not by hand |
| example user-day and its token counts (213 tokens; 54 profile, 158 behavior scored) | r4.2 `eval.jsonl` record `AAF0535:7`, tokenized with the adapter's tokenizer and labelled by `scripts/token_class_decomposition.py` |
