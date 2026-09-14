# Companion editions of the paper

Two re-tellings of `paper/main.tex`, regenerated 2026-09-14 to include the
train-matched factorial (8B causal / 3B capacity contrast), the LANL real
auth-log replication, the capture-vs-utility dissociation, and Fig. 2.

- `plain_summary.tex` — for a mathematician reader (Dr. Tian): one linear
  narrative; every abbreviation and symbol defined at first use; ideas stated
  as mathematics (detector as likelihood, delta as vector difference, sparse
  coding, AUC as a ranking probability, the factorial as a controlled
  experiment); em dashes used sparingly; validation/robustness machinery
  cut; abstract + result figures only; TMLR layout via `[accepted,preprint]`
  (no "under review" header); title without the "plain-language" subtitle.
- `readable_full.tex` — the same linear pedagogy but fully evidenced: every
  validity check, seed control, bootstrap, and robustness result kept, plus
  the delta-SAE architecture figure.

Build (no bibtex needed; the editions cite sources in prose):
    pdflatex -output-directory=build plain_summary.tex   (x2)
    pdflatex -output-directory=build readable_full.tex   (x2)
