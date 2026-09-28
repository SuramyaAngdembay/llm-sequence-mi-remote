# H6 intervention-validity check (2026-09-27/28)

**Exploratory.** The H6 users are development data. The protocol was frozen in
`PRECHECK_PROTOCOL.md` (commit `da775ec`) before any CERT state or behavioural outcome of these
interventions existed.

**Important caveat, found afterwards.** H6's "attack days" come from `labels_daily`, whose day
index is five days behind the session index (`docs/LABEL_ALIGNMENT_2026-09-28.md`). Every
receiver here is a real H6 receiver, and every result is a within-receiver comparison. Pooled
over both day types, the conclusions below do not depend on which days are truly malicious. The
attack-versus-benign split does, and should not be read as a statement about malicious activity.

## Part 2: the actual states (verified)

**Capture.** A100 job 20931848 ran for 2 min 9 s.
- **Reproduction.** The path-A codes equal H6's saved codes on all 120 receivers (max abs 0.0),
  and the capture finds the same 369 edited tokens.
- **Zero edits.** The identity hook and adding zero give bit-identical states.
- **Identities.** The SAE sha256 is `189840ce…` and the adapter `e0bf5112…`.

**Definitions** (`cert_capture/h6_capture_meta.json`):
- The site is the output of block index 25 (0-based), which is hidden state 26, at the aligned
  session positions.
- Adapted states have the adapter on; base states are the same input with the adapter disabled.
  Both are bf16.
- Path A subtracts in bf16 and then casts to float32. Path B casts to float32 and then subtracts.
- Standardization is (δ − x_mean)/x_std, followed by the encoder, ReLU and TopK with k = 4, in
  float32.

**Full-code target support**, built directly from the full TopK codes. The rows below are the
full-code target, which holds every other coefficient fixed.

| quantity | value |
|---|---|
| swapped-code active count on edited tokens | 4 on all 369 tokens (both paths) |
| targets with 5 nonzeros, hence unrealizable | 173 tokens, 46.9% |
| where those tokens sit | 23 receivers, 10 users; 82 attack-day and 91 benign-day tokens |
| their share of requested squared change | 95.3% |
| of them involving feature 7693 | 166 |
| targets with 3 nonzeros (deletions) | 5 |

A selected-only target (other coefficients free) is never over-k. That is a different
intervention.

**Arithmetic paths on the same states:**

| comparison | result |
|---|---|
| swapped session tokens whose TopK support differs between paths A and B | 238 of 23,267 (1.0%); relative code difference q99 1.8%, max 27% |
| the same, edited tokens only | 4 of 369 |
| over-k classifications that differ between paths | 0 |

**Vector execution error of H6's edit.** This is ‖applied − intended‖ on the 369 edited tokens,
not a norm ratio:

| statistic | value |
|---|---|
| relative error, median | 4.6% |
| relative error, q90 | 54% |
| relative error, max | 99% |
| cosine between applied and intended, min | 0.14 |
| applied state equal to the bf16 sum of state and edit | 100% of tokens |

The error comes entirely from bf16 rounding of state plus edit. H6's norm ratio of 1.001 hid it.

**H6's realization numbers reproduce:** path B normalized target error median 0.103, q90 1.0,
max 2.195. The mass-weighted error is 0.238. On 40 of the 173 over-k tokens, every newly
requested feature stayed inactive.

## Part 3a: precheck (passed every frozen criterion; `cert_projection/precheck.json`)

**Declared replacement.** For the 173 over-k tokens (95.3% of requested mass), the
restoration target drops the weakest non-selected R winner (swap-in). For the 5 deletion tokens,
the next feature enters with a free value (refill). This changes the intervention; it is not
H6's intended full-code restoration, which does not exist for those tokens.

| intervention | tokens verified (bf16 cast, path-B re-encode) | margin used | correction ‖x* − anchor‖ / edit | move / full-restoration distance |
|---|---|---|---|---|
| I2 restoration, decoder-anchored | 369 / 369 (100% of mass) | 0.01 (367), 0.02 (2) | median 0.11, mass-weighted 0.07 | median 0.21 |
| I1 restoration, minimum change | 369 / 369 | mostly 0.01 | — | median 0.06 |
| I2 control-feature restoration | 411 / 411 | 0.01 (408) | median 0.22 | median 0.14 |
| I2 noising in O (necessity) | 369 / 369 | 0.01 (364) | median 0.08 | median 0.21 |

- Every accepted projection has a KKT certificate.
- CPU and GPU encoders agree on all 775 tokens (max value difference 7e-5).

## Part 3b: behavioural pilot (A100 job 20937290, 3 min 35 s; `cert_pilot/`)

**Checks.**
- The SAE-layer states equal the capture bit for bit (775 tokens).
- The zero hooks are exact.
- The `dec` condition reproduces H6's `sel_sess` losses exactly, and R and O reproduce H6.
- On the GPU, all 369 / 369 / 411 / 369 patched tokens realize their declared targets.

**Results.** Behaviour-only loss in nats per token; mean of 30 user means per day type;
90% cluster-t interval (the decision interval) and 95% user bootstrap; margin τ = 0.012.

| estimand | attack-labelled days | benign days | decision |
|---|---|---|---|
| rescue, H6 decoder edit (reproduced) | +0.0008 [−0.0009, +0.0026] | +0.0014 [−0.0003, +0.0031] | negligible |
| rescue, I2 decoder-anchored feasible restoration | +0.0009 [−0.0010, +0.0027] | +0.0014 [−0.0001, +0.0029] | negligible |
| rescue, I1 minimum-change feasible restoration | −0.0004 [−0.0010, +0.0001] | −0.0001 [−0.0004, +0.0002] | negligible |
| rescue, control-feature restoration | +0.0002 [−0.0003, +0.0006] | −0.0001 [−0.0005, +0.0003] | negligible |
| rescue, size-matched random edit | −0.0000 [−0.0004, +0.0003] | +0.0009 [+0.0003, +0.0016] | negligible |
| necessity: noising the 5 coefficients in O | −0.0004 [−0.0010, +0.0002] | +0.0001 [−0.0001, +0.0003] | negligible |
| necessity, size-matched random edit | −0.0004 [−0.0007, +0.0000] | +0.0000 [−0.0002, +0.0002] | negligible |
| I2 minus control; I2 minus random; I2 minus I1; noising minus random | all within ±0.0036 | all within ±0.0034 | negligible |
| context: input effect R − O | +0.243 [+0.209, +0.278] | +0.280 [+0.244, +0.316] | non-negligible |

The intervals above are 90% cluster-t. The 95% bootstrap intervals are in
`cert_pilot/pilot_analysis.json`.

**Lift identification.** I1 and I2 agree (both negligible) on both day types.

**Reading, within this population, context and intervention family.**

- **Restoration.** Restoring the five selected coefficients has a negligible effect under both
  declared feasible lifts. For 95.3% of the requested change, this means restoring them at the
  expense of the weakest other active feature. On attack-labelled days the I2 estimate is 0.4% of
  the 0.24 profile-swap effect, and the interval's upper end is 1.1%.
- **Necessity.** Removing the same coefficients' original values in the original context also
  has a negligible effect.
- **Controls.** Neither restoration nor noising exceeds size-matched random edits, and the
  selected set does not exceed the control features.
- **Not established:**
  - that these features play no role in any context, or that no other features or combinations
    matter. Interactions and joint interventions were not tested;
  - anything about attack-specific behaviour, because of the label caveat above;
  - any causal role of feature 7693.

## Compute

| item | GPU | CPU (user + system) |
|---|---|---|
| capture job 20931848 | 2 min 9 s (A100) | 56 s + 32 s |
| pilot job 20937290 | 3 min 35 s (A100) | 144 s + 43 s |
| duplicate copies that exited on the lock | 4 s + 4 s | — |
| capture analysis, projections, precheck, pilot analysis (laptop) | 0 | about 2.5 min |
| **this validity check** | **0.10 GPU-h** | |

Cumulative GPU:
- The 2026-09-27 follow-up cap now stands at 0.56 of 1 GPU-h.
- Project GPU since 2026-09-25 is 0.95 GPU-h.

## Reproduce

```
# Anvil (A100): capture, then pilot after a passed precheck
sbatch -A cis260991-gpu -p gpu-debug slurm/anvil_friend/h6_state_capture.sbatch
python3 analyze_capture.py cert_capture cert_capture/capture_analysis.json
python3 project_targets.py --capture cert_capture --sae sae_cert_l26.npz --out cert_projection
python3 summarize_precheck.py cert_projection cert_capture cert_projection/precheck.json
sbatch -A cis260991-gpu -p gpu-debug slurm/anvil_friend/h6_intervention_pilot.sbatch
python3 analyze_pilot.py cert_pilot cert_projection cert_capture cert_pilot/pilot_analysis.json --tau 0.012
```
