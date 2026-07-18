# TMLR Draft Package

This directory is a self-contained first-draft TMLR submission package for the
current CERT-focused `llm-sequence-mi-remote` paper.

## What it is

- anonymized submission-style LaTeX draft
- official `jmlr2e.sty` vendored locally
- generated tables sourced from the current paper-safe CERT artifacts
- simple TikZ figures so the draft is not text-only

## What it is not

- not yet a polished final submission
- not yet a camera-ready package
- not yet a complete artifact appendix

## Current scientific scope

The draft follows the current paper-safe claim discipline:

- benign-trained session LLMs recover benchmark-specific causal structure on
  CERT insider-threat benchmarks
- direct token-mechanism transfer can fail
- native rediscovery can succeed

It does **not** claim strong detector superiority on CERT.

## Build

From the repo root:

```bash
bash paper/scripts/build_paper.sh
```

This will:

1. regenerate the draft tables
2. compile `paper/main.tex`
3. place outputs in `paper/build/`

The build script prefers `latexmk`. If `latexmk` is unavailable, it falls back
to repeated `pdflatex`/`bibtex`.

## Main output

- PDF: `paper/build/main.pdf`

## Important draft note

The `r4.2` same-user-excluded causal bundle currently contains a reporting
inconsistency between the top-level `RESULTS.md` narrative and the underlying
`token_delta_sae_causal_summary.csv`. This draft avoids locking a single
bootstrap headline scalar from that inconsistent pair and instead phrases the
`r4.2` causal result conservatively as positive across contexts, with the
regenerated no-same-user compare report used for the explicit remote-vs-local
comparison row.

Before submission, reconcile:

- `results/qwen3_8b_r42_token_causal/same_user_recovery/RESULTS.md`
- `results/qwen3_8b_r42_token_causal/same_user_recovery/l26_m02_k04_top5_control5_active_no_same_user/token_delta_sae_causal_summary.csv`

