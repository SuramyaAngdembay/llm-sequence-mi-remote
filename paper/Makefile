SHELL := /bin/bash
PY := $(shell command -v python3 || command -v python)

.PHONY: all tables figures pdf overleaf clean

all: pdf

tables:
	$(PY) scripts/generate_tables.py

figures:
	$(PY) scripts/generate_figures.py

pdf: tables figures
	bash scripts/build_paper.sh

# Self-contained zip for direct upload to Overleaf (New Project -> Upload Project).
# Compiler: pdfLaTeX, bibliography: BibTeX, main document: main.tex.
overleaf: tables figures
	mkdir -p build
	rm -f build/overleaf.zip
	zip -q build/overleaf.zip main.tex tmlr.sty tmlr.bst fancyhdr.sty references.bib \
		figures/*.tex figures/*.pdf tables/*.tex
	@echo "Overleaf package: build/overleaf.zip"

clean:
	rm -rf build
	rm -f tables/*.tex
	rm -f figures/*.pdf figures/*_preview.png
