SHELL := /bin/bash
PYTHON ?= python3
.PHONY: all tables figures pdf overleaf clean
all: pdf
tables:
	$(PYTHON) scripts/generate_tables.py
figures:
	$(PYTHON) scripts/generate_figures.py
pdf:
	$(PYTHON) scripts/build_editions.py
overleaf: tables figures
	mkdir -p build
	$(PYTHON) scripts/package_recovery.py
clean:
	rm -rf build companion/build
