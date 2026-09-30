# Companion editions

Both current PDFs are synchronized to the corrected-label evidence registry on the repair branch.

- **readable_full.tex / build/readable_full.pdf**: complete reading edition, preserving problem → data → scoring → internal method → findings → limits. The appendix contains the technical specification and provenance also present in main.
- **plain_summary.tex / build/plain_summary.pdf**: shorter explanation of the supported results and unresolved work.

These are recovery working editions. Main adapters and corrected score results are reusable; not every historical causal/comparator claim is repaired. The original versions are preserved in Git at `e0ad1a3`.

Build all editions from `paper` with `PYTHON=/path/to/python TECTONIC=/path/to/tectonic bash scripts/build_paper.sh`. The companions share `../data/recovery_registry.json`, generated tables, the corrected figure, and `../recovery_content/`; retain the repository layout. See `../CLAIM_MAP.md` for scope and the root paper README for provenance. The older `READABLE_COVERAGE.md` describes the pre-recovery edition and is historical.
