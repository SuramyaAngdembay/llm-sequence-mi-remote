# October 7 meeting notes: thesis revision

Applied on October 14, 2026. Meeting-note source:
`/Users/suramya/research-hub/interp-insider-threat/notes/2026-10-07-thesis-meeting-notes.md`.
The note records the user's wording; it does not name attendees or establish the meeting date.

## Structural changes

| Meeting request (old numbering) | Applied change |
|---|---|
| Make the abstract brief, high-level, conclusion-led; remove LoRA | 151-word abstract. Removed LoRA detail and the low-rank keyword. Technical training explanation stays in the body. |
| Combine 2.1 and 2.2 | New 2.1 combines motivation and mathematical formulation. |
| Consolidate 2.4 and 2.5 | New 2.3 combines the numerical reconstruction example and geometry. |
| Combine 2.6 and 2.7; possibly put this in the summary | New 2.4 combines the high-dimensional and synthetic anomaly examples. New 2.7 provides the short synthesis. All seven original figures remain. |
| Possibly fold 3.2 into 3.1 | Dataset and insider cases form one section, now 2.5. |
| Possibly distribute setup across the method chapters | PCA preparation/results move to 2.6; LM preparation moves to 3.2; evaluation details accompany each experiment. The separate setup chapter is removed. |
| Consolidate Chapter IV similarly to Chapter II | The language-model chapter, now Chapter III, has six sections: motivation; data/text; probability/surprisal; training; scoring/decomposition; evaluation/results. |

The notes contain a cut-off second “remove” instruction. No additional removal was inferred
from that fragment. The tentative structure requests were implemented as reversible editorial
choices. Chapter I and personal/front-matter placeholders remain for the author's completion.

## Evidence corrections made with the restructuring

A structural edit alone would have retained obsolete label-dependent results. This revision
uses the completed **r4.2** evidence already available; it launches no new experiments.

- **Population:** corrected calendar-window labels, 1,362 positive observed days, 70 insiders,
  79 held-out non-insiders, 42,468 evaluation days. Window-negative days are not called
  verified benign days. Event-day labels used by PCA remain a separate target.
- **Score decomposition:** user AUC 0.589 (full) to 0.825 (behavior targets), paired gain
  0.236 [0.185, 0.289]. Fold day AUC/AP are explicitly distinguished from user AUC and pooled
  prevalence. No claim of a statistically established reversed profile-only ranking.
- **PCA:** the already-saved all-70-insider rows replace the restricted 60-insider display.
  The ROC is regenerated from the original saved user scores; its area equals the saved
  0.7773960217 AUC and the direct pairwise calculation. No refit. PCA and LM use different
  features and observed-day domains; this is not an architecture leaderboard.
- **SAE:** corrected discovery-selected feature IDs and attribution replace the old table.
  The completed residual-preserving ablation replaces the old donor-search causal result.
  Selected removal raises behavior loss, but the incident-minus-window-negative contrast is
  negative. No attack-specific mechanism, mitigation-from-ablation, or independent confirmation
  is claimed. The exploratory control-policy provenance is retained.
- **Equations:** SAE encoding now includes the actual ReLU, encoder bias and saved scaling;
  decoding and physical subtraction use the correct coordinate system. The AUC equation gives
  explicit half credit to ties. The low-rank bound applies to each matrix, not the entire
  nonlinear network. Activation enrichment describes mass, not firing frequency.
- **SAE sampling:** the reconstruction objective has no label term, but the two-million-row
  sample preserves 171,917 discovery-positive token rows. The thesis now explicitly distinguishes
  that label-aware sample from an unsupervised objective and from discovery-only feature ranking.
  This follows `build_stage_b_labels.py`, `audit_presampled_population.py` and the saved
  independent sample audit, not an assumption that all dictionary fitting is label-free.

`revision_evidence.json` stores the copied result values and source hashes. Eleven source
artifacts were checked against the current recovery registry before reuse. This is a document
revision and source-consistency check, not a fresh independent revalidation of the underlying
model experiments. Original recovery reviews retain their own scopes and qualifications.

## Files, build, and remaining work

- Editable source: `thesis_mechinterp.tex`.
- Current PDF on GitHub: `thesis_mechinterp.pdf`, identical to the local compiled
  `build/thesis_mechinterp.pdf`.
- Seven original PNG figures: `figures/`, extracted from the September 28 full PDF at their
  embedded resolutions. The source PDF hash, image hashes and PDF object IDs are in the evidence
  file. No replacement illustrations were invented.
- The old `thesis_mechinterp_preview.pdf` is historical and is not the updated draft.
- The prior source, prior README, revision scripts and verification report are preserved in
  `/Users/suramya/Documents/thesis-meeting-update-2026-10-14/`.

Still to finish: write Chapter I; complete the title-page month/year and approval-page details;
write or omit the optional dedication and acknowledgments; confirm the final institutional
format. These are distinct from the meeting-note restructuring. The source still visibly
contains those original placeholders. Broader r6.2 recovery is outside this r4.2 thesis revision.
