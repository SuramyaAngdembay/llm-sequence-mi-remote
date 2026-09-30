# Paper: corrected evidence working editions

The calendar-label recovery is reflected in **all three current editions**:

- `main.tex` → `build/main.pdf`: research edition, technical specification in the body.
- `companion/readable_full.tex` → `companion/build/readable_full.pdf`: the same evidence and technical content, with a linear explanation and technical appendix.
- `companion/plain_summary.tex` → `companion/build/plain_summary.pdf`: short summary and current limits.

The main labels and reported LM score results are repaired. Historical causal and comparison-model claims are **not all repaired**. These are recovery working editions, not submission-ready replacements claiming every historical result survived. The original texts remain in Git at `e0ad1a3`; external source snapshots also have a hash manifest.

## Source of truth

`data/recovery_registry.json` binds 46 claim families to exact evidence and source hashes. `CLAIM_MAP.md` and `data/claim_manuscript_map.json` locate every family in every edition. Shared prose is in `recovery_content/`. Generated current tables start with `recovery_`; the current figure is `recovery_score_mitigation.pdf`. Other old tables/figures and other manuscript variants are historical and are not included by these builds.

See `../docs/review/RECOVERY_CLOSEOUT_2026-09-29.md` for the audit and `../docs/review/TARGETED_RERUNS_2026-09-29.md` for remaining experiments. The thesis is outside this work.

## Build all three

Use Python with matplotlib and Tectonic (or pdfLaTeX/BibTeX):

```sh
PYTHON=/path/to/python TECTONIC=/path/to/tectonic bash scripts/build_paper.sh
```

Alternatively `make pdf PYTHON=/path/to/python` finds an installed TeX engine. Each edition has its own build log. `build/recovery_build_manifest.json` records the PDF and result-registry hashes. `make overleaf` packages only the current main manuscript inputs.

After changing the registry, regenerate tables/figures and run `scripts/review/map_recovery_claims.py` from the repository root. Do not edit generated values or copy stale historical figures into the current sources.
