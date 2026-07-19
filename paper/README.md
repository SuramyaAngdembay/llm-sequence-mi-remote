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

The repaired same-user-excluded causal rows should be interpreted using the
complete receiver-level top-vs-control contrast that matches the bootstrap
estimand. The paper generator follows that repaired bundle convention.

## Overleaf

The package compiles on Overleaf with:

- compiler: `pdfLaTeX`
- bibliography: `BibTeX`
- main document: `main.tex`

Two ways to get it there:

1. **Zip upload (recommended):** run `make overleaf` from `paper/`, then in
   Overleaf use *New Project -> Upload Project* with `build/overleaf.zip`.
   The zip contains only what Overleaf needs (`main.tex`, `jmlr2e.sty`,
   `references.bib`, `figures/`, `tables/`).
2. **GitHub import:** import the repo and set `paper/main.tex` as the main
   document (Menu -> Settings -> Main document).

Data figures (`figures/*.pdf`) are pre-generated vector PDFs produced by
`scripts/generate_figures.py` (matplotlib) from the tracked result bundles;
Overleaf does not run Python, so regenerate them locally (`make figures`)
and re-upload if the numbers change. The pipeline diagram stays TikZ and
compiles natively on Overleaf.
