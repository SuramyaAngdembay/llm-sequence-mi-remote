SHELL := /bin/bash

.PHONY: all tables pdf clean

all: pdf

tables:
	python scripts/generate_tables.py

pdf: tables
	bash scripts/build_paper.sh

clean:
	rm -rf build
	rm -f tables/*.tex

