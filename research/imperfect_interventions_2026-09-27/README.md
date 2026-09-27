# When does a near-zero effect of an imperfect feature intervention mean anything?

**Exploratory research, 2026-09-27. Kept separate from the paper's claims.** CPU only, no model
inference. The review notes that prompted it are in `~/Documents/mi-paper-review-2026-09-27/`
(`mathematical-directions-after-h6.md` and `h6-followup-review.md`).

## Answer

**An observed near-zero effect supports a negligible *ideal* effect only in three cases:**

- the declared ideal state exists;
- the ideal is executed directly, verified by re-encoding, and passes an equivalence test at a
  pre-declared margin;
- it passes a Lipschitz-widened test, which in practice decides only when realization is
  essentially exact.

A generic implementation-error bound is valid but almost never decisive. In the frozen
evaluation it was decisive in 1 of 14 conditions with inexact realization. Executing the ideal
directly was right in 98% of datasets. The proposed method therefore fails its own success
criteria. This is a negative result for the method.

Two further limits apply even with exact execution:

- **The answer is specific to the lift.** States with the same code can behave differently, and
  two reasonable lifts disagree in several worlds.
- **The answer is specific to restoration.** A restoration null can coexist with necessity and a
  large interaction.

**For H6** (Proposition 1, corollary), 173 of the 369 edited session tokens asked for a TopK code
with five nonzeros from an SAE that keeps exactly four. Those tokens carry 95% of the requested
change. The declared "restore these five coefficients, hold the rest" state does not exist for
them. H6's 0.3% therefore reports the decoder edit that was applied. It is not evidence about
restoring those coefficients.

## Documents

- `ESTIMAND_AND_DERIVATIONS.md` covers:
  - the problem statement and estimand;
  - the ideal-selection rules and the distinction between distance to a set and distance to a
    declared ideal;
  - Propositions 1 to 5 with proof status;
  - counterexamples A to F;
  - the separation of uncertainty sources;
  - the declared margins.
- `PRIOR_ART.md` compares the six papers and five direct leads, lists the sections inspected,
  and assesses novelty.
- `FROZEN_PROTOCOL.md` holds the decision rules and success criteria, committed as `8aa4fcd`
  before any evaluation world existed.

## Synthetic known-answer study

`synth.py` builds worlds with:

- a TopK SAE with controlled crosstalk;
- non-uniform standardization scales;
- directions the encoder cannot see;
- a downstream loss with known sensitivity.

Ground truth comes from executing each ideal exactly, by an exact projection onto the code-τ
polyhedron that is verified by re-encoding.

Procedures compared:

- **P1** tests the applied effect for equivalence.
- **P2** gates P1 on realization error, with a threshold tuned on development worlds.
- **P3** directly executes the decoder-anchored ideal.
- **P4a** widens P1 by K × distance, with a certified regional K.
- **P4b** is P4a with a sampled K. That K is not a bound.
- **P5** runs P3 for two lifts.

Development used three worlds per condition. Evaluation used six fresh worlds per condition, 50
datasets each, with 30 clusters of 3 units. The margin τ is 5% of each world's full-restoration
rescue.

**Cases and ground truth** (evaluation worlds; ranges over the six worlds):

| case | condition | τ (mean) | ideal rescue I2 | ideal rescue I1 | necessity | interaction | realization error |
|---|---|---|---|---|---|---|---|
| 1 | irrelevant, exact realization | 0.008 | -0.000 to +0.000 | -0.018 to +0.040 | -0.000 to +0.000 | +0.000 to +0.000 | 0.00 |
| 1 | irrelevant, crosstalk 0.15 | 0.008 | -0.001 to +0.004 | -0.036 to +0.062 | -0.001 to +0.001 | +0.000 to +0.000 | 0.07 |
| 1 | irrelevant, crosstalk 0.4 | 0.009 | -0.006 to +0.010 | -0.032 to +0.062 | -0.002 to +0.004 | +0.000 to +0.000 | 0.18 |
| 1 | irrelevant, execution noise 5% | 0.008 | -0.000 to +0.000 | -0.018 to +0.040 | -0.000 to +0.000 | +0.000 to +0.000 | 0.03 |
| 1 | irrelevant, edit applied at 10% | 0.008 | -0.000 to +0.000 | -0.018 to +0.040 | -0.000 to +0.000 | +0.000 to +0.000 | 0.90 |
| 1 | irrelevant, MLP readout (loose K) | 0.013 | -0.017 to +0.009 | -0.113 to +0.124 | -0.005 to +0.003 | +0.000 to +0.000 | 0.07 |
| 1 | irrelevant, protected row at cosine 0.97 | 0.025 | -1.233 to +0.361 | -1.341 to +0.380 | -0.075 to +0.122 | +0.000 to +0.000 | 0.97 |
| 1 | irrelevant, uniform scales | 0.008 | -0.000 to +0.000 | -0.000 to +0.000 | -0.000 to +0.000 | +0.000 to +0.000 | 0.00 |
| 2 | relevant, exact realization | 0.026 | +0.336 to +0.361 | +0.334 to +0.400 | +0.150 to +0.163 | +0.000 to +0.000 | 0.00 |
| 2 | relevant, crosstalk 0.15 | 0.027 | +0.334 to +0.395 | +0.333 to +0.416 | +0.150 to +0.172 | +0.000 to +0.000 | 0.07 |
| 2 | relevant, MLP readout (loose K) | 0.035 | +0.372 to +0.503 | +0.361 to +0.533 | +0.161 to +0.198 | +0.000 to +0.000 | 0.07 |
| 2 | relevant, protected row at cosine 0.97 | 0.072 | -0.285 to +1.302 | -0.392 to +1.322 | -0.010 to +0.649 | +0.000 to +0.000 | 0.97 |
| 3 | relevant, edit applied at 2% | 0.026 | +0.336 to +0.361 | +0.334 to +0.400 | +0.150 to +0.163 | +0.000 to +0.000 | 0.98 |
| 3 | weakly relevant, edit applied at 5% | 0.013 | +0.099 to +0.107 | +0.087 to +0.147 | +0.041 to +0.045 | +0.000 to +0.000 | 0.95 |
| 3 | relevant, execution noise 30% | 0.026 | +0.336 to +0.361 | +0.334 to +0.400 | +0.150 to +0.163 | +0.000 to +0.000 | 0.16 |
| 4 | decoder writes an encoder-invisible direction | 0.057 | +0.903 to +0.958 | -0.067 to +0.161 | +0.430 to +0.462 | +0.000 to +0.000 | 0.00 |
| 5 | AND with an invisible partner | 0.065 | -0.000 to +0.000 | -0.862 to +0.101 | +0.204 to +0.221 | +0.443 to +0.476 | 0.00 |
| 6 | insertion into a full TopK code (over-k) | 0.050 | undefined | undefined | n/a | n/a | 0.57 |
| 6 | deletion requiring a support change | 0.050 | +0.282 to +0.617 | +0.294 to +0.627 | n/a | +0.000 to +0.000 | 0.55 |
| 2/3 | relevant, steep MLP, execution noise 30% | 0.025 | -0.022 to +0.531 | -0.041 to +0.638 | +0.004 to +0.213 | +0.000 to +0.000 | 0.17 |

**Correct-decision rate by procedure** (300 datasets per condition). FN is the false-negligible
rate and FP the false-non-negligible rate. "neg/non-neg" means the six worlds straddle τ.

| condition | truth (I2) | P1 applied | P2 gated | P3 direct | P4a bound | P4b sampled K | P5 two lifts |
|---|---|---|---|---|---|---|---|
| C1a_irrelevant_exact | neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |
| C1b_irrelevant_crosstalk | neg | 0.51, **FP 0.35** | 0.28 | 1.00 | 0.00 | 0.00 | 1.00 |
| C1c_irrelevant_crosstalk_hi | neg/non-neg | 0.17, **FP 0.60** | 0.00 | 0.85 | 0.00 | 0.00 | 0.83 |
| C1d_irrelevant_exec_noise | neg | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.96 |
| C1e_irrelevant_under | neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.96 |
| C1f_irrelevant_mlp | neg/non-neg | 0.83, **FP 0.20** | 0.28 | 0.92 | 0.00 | 0.00 | 0.72 |
| C1g_irrelevant_coherent | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C1h_irrelevant_uniform_sigma | neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C2a_relevant_exact | non-neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C2b_relevant_crosstalk | non-neg | 1.00 | 0.28 | 1.00 | 0.98 | 1.00 | 1.00 |
| C2c_relevant_mlp | non-neg | 1.00 | 0.28 | 1.00 | 0.00 | 1.00 | 1.00 |
| C2d_relevant_coherent | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C3a_relevant_under | non-neg | 0.00, **FN 1.00** | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C3b_small_relevant_under | non-neg | 0.00, **FN 1.00** | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C3c_relevant_exec_noise | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.88 | 1.00 |
| C4_many_to_one | non-neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| C5_interaction | neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| C6a_over_k | ill-posed | 0.00 | 0.00 | ill-posed | ill-posed | ill-posed | ill-posed |
| C6b_deletion | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C7_steep_exec_noise | non-neg | 0.91 | 0.00 | 0.85 | 0.00 | 0.00 | 0.85 |

**Frozen criteria for P4a** (`results/eval/summary.json`):

| criterion | result |
|---|---|
| 1. Validity: false-negligible rate ≤ 0.05 in every condition | **passes** (0.00 everywhere) |
| 2. Informativeness: correct and decisive in ≥ 50% of datasets in at least half of the 14 inexact conditions | **fails** (1 of 14) |
| 3. Pooled wrong-or-inconclusive rate below P2's | **fails** |
| 4. Value beyond direct execution | not demonstrable here; direct execution was always available when the bound was |

Pooled wrong-or-inconclusive rates:

| procedure | rate |
|---|---|
| P4a, certified bound | 0.69 |
| P2, realization gate | 0.62 |
| P1, applied effect | 0.19 |
| P3, direct execution | 0.02 |

Why the bound fails (Proposition 5): it can decide only if the applied state lies within about
τ/K of the ideal. That is 0.3% to 0.9% of the edit's length, against measured distances of 5%
under crosstalk and 30% under execution noise.

## What each simple check gets wrong

- **Applied-effect equivalence (P1)** falsely declares negligible in 100% of datasets when the
  edit is under-applied (C3a, C3b). It falsely declares non-negligible in 20% to 60% of datasets
  when crosstalk moves a protected feature that the loss reads (C1b, C1c, C1f). The P0 rule, which
  calls an effect negligible whenever its interval contains zero, does worse.
- **Realization error alone (P2)** was never wrong on the evaluation worlds at the tuned
  threshold of 0.05. Any larger threshold was wrong on development worlds. At 0.05 it is
  inconclusive in 62% of evaluation datasets. It cannot tell which errors the loss reads
  (counterexample D).
- **Direct execution (P3)** decides correctly almost everywhere. It correctly reports the over-k
  target as ill-posed. It is still specific to the lift (see P5 and counterexamples B and E) and to
  restoration (counterexample C).

## Go / no-go for a small-model pilot

**No-go.** The frozen gate in the specification requires all of the following:

- a coherent estimand: **yes**, with a declared lift and a full-code target;
- a correct derivation: **yes**, elementary;
- informative bounds on nontrivial known-answer cases: **no**;
- better decisions than simpler checks: **no**;
- a plausible contribution in the prior art: **only as a protocol**.

No TWOS or transformer run was made.

If the H6 feature question itself matters, the right experiment is a direct execution of a
*feasible* declared intervention, not a bound. Three changes are needed:

- choose a target that exists:
  - "swap-in", where the weakest swapped winner may drop, which is a different and declared
    estimand;
  - or restrict the test to tokens whose target is not over-k;
- execute the decoder-anchored exact projection on captured states, verify it by re-encoding
  with identical arithmetic, and test at τ = 0.012 nats per token;
- add a second lift, plus a noising patch for necessity.

That is an application of established tools. It would be one short GPU job and needs your
decision.

## What could become a contribution

- **Not a method.** The Lipschitz-widened interval is correct and nearly always inconclusive.
  Hoffman constants add nothing over the exact projection. Certified regional sensitivity for
  transformers is an open verification problem, far beyond a bounded project.
- **Possibly a methods note or application section.** It would present a feasibility-first
  checklist for SAE feature interventions:
  1. certify that the declared target code exists, with an over-k count;
  2. execute declared lifts directly and verify them by re-encoding;
  3. compare two lifts to check identification;
  4. test equivalence at a pre-declared margin;
  5. keep restoration, necessity and interaction as separate estimands.

  The H6 over-k finding and counterexamples B to F would be its worked examples. Each step is
  established, and the value would be in the protocol and its demonstrations.
- **The donor-policy LP** remains the existing paper's robustness contribution, unchanged.

## Compute

This investigation used 0 GPU-hours, with no model inference. CPU time was measured with
`/usr/bin/time` or `time.process_time`.

| item | CPU (user) |
|---|---|
| synthetic development runs: v1, v2, v3 percentile, v3 t, v3 t with 30 clusters | 218 + 14 + 16 + 15 + 17 s |
| synthetic evaluation (frozen) | 35 s |
| tests, tables, bound-scale check, H6 local check | about 5 s |
| Anvil CPU: layer-26 SAE and cached-delta checks | about 155 s |
| **total** | **about 8 minutes** |

GPU use across the project since 2026-09-25, before any further model run:

| period | GPU-hours |
|---|---|
| exploration round | 0.39 |
| audit follow-up | 0.46 |
| this investigation | 0 |
| **total** | **0.85** |

Unused authorization is 0.54 GPU-h of the follow-up cap and 7.61 of the round's 8.

## Reproduce

```
python3 test_synth.py
python3 synth.py dev  --config config_dev.json    --out results/dev_percentile
python3 tune_p2.py results/dev_t30/rows.json       # after the t30 development run
python3 synth.py eval --config config_frozen.json --out results/eval
python3 summarize_eval.py results/eval/evaluation.json results/eval/worlds.json results/eval/summary.json
python3 make_tables.py > results/eval/tables.md
python3 h6_realizability_cpu.py local --codes ../../results/h6_2026_09_27/out/h6_token_codes.npz --out results/h6_realizability_local.json
python  h6_realizability_cpu.py anvil --share /anvil/scratch/x-sangdembay/pkg4_share --out results/h6_realizability_anvil.json   # on Anvil
```
