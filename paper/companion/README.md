# Companion editions

## Which version to read

- **[readable_full.tex](readable_full.tex)** is the complete reading edition: a linear explanation, findings and caveats, then a technical appendix and references. [Open the PDF](build/readable_full.pdf).
- **[plain_summary.tex](plain_summary.tex)** is the shorter mathematical introduction. It omits much of the validation detail. [Open the PDF](build/plain_summary.pdf).
- **[../main.tex](../main.tex)** is the longer research manuscript and historical source for the companions.

The readable edition was revised against repository commit `8a4e9d7` on September 14, 2026. Its narrative retains the problem → data → detector → internal audit → findings → robustness → real data → input-removal experiment sequence. Supporting detail is in `readable_appendix.tex`.

[READABLE_COVERAGE.md](READABLE_COVERAGE.md) maps the longer manuscript's evidence to this edition and lists corrections and result sources. This revision updates the readable edition; the other PDFs have not been regenerated as part of this edit.

## Build

Run from `paper/companion`. Keep the repository layout: the readable edition uses shared `../references.bib`, local `tmlr.sty`/`tmlr.bst`, the appendix, and `figures/`. The older local `references.bib` is not used by this build.

With Tectonic installed:

```sh
mkdir -p build
tectonic --keep-logs --keep-intermediates --reruns 2 --outdir build readable_full.tex
```

Tectonic runs BibTeX and resolves references automatically. Alternatively, with a standard TeX installation:

```sh
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build readable_full.tex
bibtex build/readable_full
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build readable_full.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build readable_full.tex
```

The plain summary currently needs two `pdflatex` passes and no BibTeX:

```sh
pdflatex -output-directory=build plain_summary.tex
pdflatex -output-directory=build plain_summary.tex
```

Compiling this edition does not rerun experiments or regenerate the other PDFs.
