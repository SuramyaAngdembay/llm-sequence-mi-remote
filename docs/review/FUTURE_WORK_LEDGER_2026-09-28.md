# Future-work ledger: preserved directions, not recovery work

These are kept so they are not lost. None is a paper claim, none is started here, and no novelty
search was run.

| direction | what exists | assumptions | missing evidence | source |
|---|---|---|---|---|
| Full-code fibres and feasible lifts of SAE edits (C43) | Proofs: a full-code target's margin-qualified fibre is one polyhedron; over-k targets are empty; Hoffman and exact-distance decompositions; validity of the widened test | TopK encoder, a known margin, a valid regional Lipschitz bound | A tight, tractable regional bound for the Qwen suffix. The widened bound abstained 68.5% on the synthetic benchmark | `research/imperfect_interventions_2026-09-27/METHOD_STATUS.md` |
| Second-order bounds along the actual correction direction | Route A in METHOD_STATUS | Twice-differentiable suffix on the segment; certified curvature bound M | Whether certified error bounds for attribution patching already exist; any certified M for Qwen | same |
| Certified bounds over restricted lift families | Route B in METHOD_STATUS | A justified low-dimensional family; tractable bound propagation with NF4 weights | Whether certified intervention families exist in the literature; width versus empirical range | same |
| Fixed-geometry H6 diagnostics (C38) | Verified capture of real layer-26 states; over-k target frequency (46.9% of edited tokens); bf16 execution error of edits (median 4.6%, q90 54%) | Fixed states and adapter; these do not depend on labels | Class splits must use corrected labels (48 of 120 H6 labels change); no attack-specific reading | `research/imperfect_interventions_2026-09-27/h6_states/README.md` |
| Fixed-dictionary alignment and non-identifiability (C19) | Seed-retrained dictionaries (r4.2 seeds 43/44) | A fixed dictionary as a coordinate system | Historical fits were unseeded (review finding), so this seed study does not separate the sampling seed from initialisation noise | Anvil `token_delta_sae_frontier_r42_seed4{3,4}` |
| Batch-shape numerics as an intervention confound | Measured: batch-size-only NLL changes of 0.010–0.013 per example; batch 1 is bit-reproducible | Same adapter, text and GPU type | Whether same-batch paired designs remove it fully for every historical estimand | review F2 |
| Behavior-only scoring as a detector design | Cached reanalysis: large across-user gains, much smaller within-user gains | Profiles remain in the input | Complete r4.2 coverage (Stage B1); seeds; an internal account of why profile tokens hurt | recovery report; review F5 |
