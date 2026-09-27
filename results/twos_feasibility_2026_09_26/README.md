# TWOS feasibility pilot: realizable feature edits and donor-policy sensitivity (exploratory, 2026-09-26)

Specification: `~/Documents/mi-paper-review-2026-09-26/claude-aquaman-twos-pilot-prompt.md`
with its research note `mathematical-directions.md`. **Everything here is
exploratory method development.** The TWOS confirmation cohort was already
examined in the corrected run (job 20890658), and the CERT rows used in the
extension were examined in package 4. Nothing here changes the paper's claims.

Two directions were tested:

- **A. Intended versus realized SAE feature edits.** Does an edit that requests
  a code change actually produce it under the frozen encoder, and do
  constrained edits reduce unintended changes?
- **B. Donor-policy sensitivity.** Do intervention conclusions survive
  reasonable reweighting of the donors?

The whole pilot used **0.03 GPU-hours** (one RTX 3070) of its 2-hour cap. The
exploration round has used 0.39 of 8 GPU-hours in total.

## Environment and artifact identity (checked before any computation)

| item | value |
|---|---|
| host | Aquaman, 2 × RTX 3070 8 GB, both idle at start and before each GPU run; only GPU 0 used |
| environment | `/home/suramya/cert-venv`: Python 3.10.12, torch 2.5.1+cu121, transformers 5.14.1, peft 0.19.1, bitsandbytes 0.49.2, scipy 1.15.3, numpy 2.2.6 (not upgraded) |
| model | Qwen/Qwen2.5-3B, NF4 4-bit with double quantization, bf16 compute (`twos_work/qwen3b_twos.yaml`, sha256 `b77e16ac…`); offline cache `insider_mi/hf_cache` |
| adapter | `lm/twos_adapter_s42_fixed`, `adapter_model.safetensors` sha256 `e602ba15…` |
| SAE | layer 24, m = 4, k = 8, `review_2026_09_24/twos_s42_reselect_discovery/layer_24/m04_k08/delta_sae_model.pt` sha256 `a2e1ac26…` |
| ranking | sha256 `12b10690…` (as specified); 8,192 finite statistics; `choose_feature_sets` + `add_active_control_feature_sets` reproduce selected `[6036, 7375, 1197, 7420, 3218]` and control `[7962, 972, 8082, 7232, 3157]` |
| split | discovery file `44b1f032…`, confirmation file `a17e75c4…` (both as in the plan) |
| deltas | `twos_work/v3_deltas_s42/layer_24/chunk_00000.pt`, 228,465 token rows × 2,048 float16, sha256 `747deb15…` |
| candidate rows | `results/twos_corrected_bounded/token_delta_sae_causal_candidate_rows.csv`, sha256 `2fbfe5f2…` (as specified), 8,400 rows |
| run directory | `/data/suramya/insider_mi/twos_feasibility_2026_09_26/` (repo snapshot, inputs, phase outputs, logs). Existing artifacts untouched. |

TWOS records are 30-minute windows serialized as `DAY team=… leader=… machine=…`,
`PSY O=… C=… E=… A=… N=…`, `SESSIONS …`, `SES win=… mouse_ev=… … ev_off=…`, about
100 tokens each. The only context field is `team`; each of the 6 teams has 4
users, of whom 1 or 2 were never malicious. The TWOS scripts are
parameterized for this format. None of the CERT department parsing or feature
ids is used.

## Phase 1: donor-policy LP on saved scores (CPU)

`scripts/twos_donor_policy_lp.py` (outputs in `phase1/`; plot
`phase1/donor_policy_bounds.svg`). The join uses (context, donor type, alpha,
receiver id, donor id); row order is never used. There are 1,088 benign alpha-1
pairs present in both arms (68 receivers, 8 users) and no same-user donor. The
paired contrast t = selected − control is formed per pair, so both arms use
the same weights. Each user has weight 1/8, split equally over that user's
receivers and then uniformly over each receiver's donors. The bounds are
min/max over donor weights with fixed receiver marginals, nonnegative weights,
and total-variation distance at most ρ from uniform.

Checks (88, all pass):

- HiGHS and an exact greedy solution agree within 4e-16. Within a receiver,
  the greedy moves weight to that receiver's most extreme donor, largest gain
  first.
- ρ = 0 reproduces `endpoints.json` exactly.
- ρ = 1 equals the per-receiver extreme donors.
- Two synthetic cases with known analytic answers pass.
- The alpha-0 contrast is donor-independent (spread 2.4e-7), so its width is 0.

| benign donors, alpha 1 | uniform | ρ = 0.05 | ρ = 0.10 | ρ = 0.25 | breakdown radius ρ* |
|---|---|---|---|---|---|
| behaviour, selected − control | −0.00011 | [−0.00176, +0.00129] | [−0.00273, +0.00217] | [−0.00433, +0.00370] | **0.0024** |
| behaviour, minus alpha-0 contrast† | −0.00021 | [−0.00186, +0.00119] | [−0.00283, +0.00207] | [−0.00443, +0.00360] | 0.0052 |
| profile, selected − control | −0.01286 | [−0.01592, −0.00980] | [−0.01779, −0.00773] | [−0.02092, −0.00410] | none (negative even at full concentration) |
| profile, alpha-0 contrast alone | −0.01077 | width 0 | width 0 | width 0 | none |
| profile, minus alpha-0 contrast† | −0.00209 | [−0.00515, +0.00098] | [−0.00702, +0.00305] | [−0.01015, +0.00667] | **0.032** |

† Additive subtraction of the alpha-0 contrast. It does not remove
nonlinear reconstruction–edit interactions. Anomalous donors give the same
picture (behaviour ρ* 0.030; profile net of reconstruction ρ* 0.106).

**Reading.** The TWOS behaviour contrast is null, and its sign carries no
information: moving 0.24% of donor weight reverses it. The profile contrast
keeps its sign under any reweighting of this bank, but only because of the
donor-independent reconstruction term. The edit part alone flips at 3.2%. The
LP therefore separates the policy-invariant part of a contrast from the
policy-sensitive part. A plain interval does not show that.

**Extension to CERT package 4** (`scripts/cert_donor_policy_lp.py`, outputs
in `phase1_cert_extension/`; input sha256 `2387be94…`; 40 checks pass, ρ = 0
equals every published endpoint). At alpha 1 with benign donors:

- The **behaviour** contrast is positive in all four contexts, and its sign
  survives *every* reweighting of the bank. Even with each receiver
  concentrated on its least favourable donor, it stays at +0.0027 to +0.0053.
  At ρ = 0.25 it is [+0.0068, +0.0278] for dept.
- The **profile** contrast flips at ρ* = 0.013–0.021 for benign donors and
  0.054–0.082 for anomalous donors.
- The **full** score flips only near full concentration (ρ* 0.55–0.72).

These are exact bounds for this finite bank, not population intervals. They
agree with the package-4 bootstrap intervals (behaviour excludes zero, profile
includes it). The historical best-candidate difference-in-differences selects
donors separately per arm. It is a different estimand and was not bounded
here. Averaging donor outcomes is also not the same as patching an averaged
donor representation.

## Phase 2: requested versus realized codes (cached activations, CPU)

`scripts/twos_edit_realizability.py` with `scripts/sae_edit_geometry.py`
(15 unit checks, including the research note's Top-2 toy example; outputs in
`phase2/`). The development population is the 8 discovery users: 70 receivers,
each with up to 16 benign same-team donors chosen by sha256 order. The
evaluation population is the exact 1,088 benign and 1,012 anomalous scored
pairs. For each edit: z = E(u); z_req under the union or own-support rule at
alpha 1 with the package-4 prototype; u_new = u + D(z_req − z); z_real =
E(u_new). The simple baseline is the minimum raw-movement encoder projection
that sets the targeted preactivations to the requested values.

Encoder fidelity:

- float32 torch and float64 numpy supports agree on 100% of tokens.
- There are no near-ties at the 8th position.
- Every token has exactly 8 positive winners.

Among the evaluation pairs, 25 of 68 receivers have no active selected feature
(400 of 1,088 pairs), so every edit there is a no-op. They are counted
separately.

Evaluation, benign donors (development is the same within a few points):

| edit | tokens | requested code has > 8 nonzeros | realized/requested target change, median [q10] | off-target change, relative | support Jaccard | raw movement (median) |
|---|---|---|---|---|---|---|
| selected, union, decoder | 2,752 | 14.7% | 1.00 [0.975] | 0.27 | 0.42 | 19.9 |
| selected, own, decoder | 2,752 | 0 | 1.00 [1.00] | 0.28 | 0.44 | 19.1 |
| selected, own, encoder projection | 2,752 | 0 | 1.00 [1.00] | 0.34 | 0.48 | **7.8** |
| control, union, decoder | 1,904 | 65.2% | 0.91 [0.004] | 0.37 | 0.71 | 4.1 |
| control, own, decoder | 1,904 | 0 | 1.00 [0.022] | 0.37 | 0.78 | 0.7 |

Anomalous donors raise the union over-k share to 55.9% (selected) and 70.6%
(control). For the selected features, 78% of union writes into zero
coordinates became active on benign donors. The reconstruction error per token
(raw, median) is 9.7, about half the size of the selected edit.

**Reading.**

1. **Union requests are often unrealizable by construction.** A requested
   code with more than k nonzeros cannot be an output of this encoder. That
   is a statement about the encoder, not about whether the model state is
   natural or the causal effect valid.
2. **Own-support requests are realized exactly for the selected features.**
   They do, however, rewrite about half of each token's active set (roughly 3
   new and 2.4 dropped features per token). The benign donors' prototypes for
   the selected features are mostly about 0, so the requests are deletions,
   and under TopK with all 8 winners positive a deletion always admits a
   replacement.
3. **The decoder edit moves about 2.4× more than a target-exact edit needs.**
   The projection reaches the same targets with 7.8 raw units instead of 19.1,
   and its collateral is not lower.

Bounded solver stages:

- **Fixed-cell minimum-movement QP** (40 exactly-8-positive cells per
  population and arm; ε = 1–25% of each protected feature's value; margin
  1e-3). For selected features, 36 of 40 development cells and 38 of 40
  evaluation cells (benign donors; 18 of 40 with anomalous donors) request a
  value below the margin, so they leave the cell.
  That is a fixed-cell statement, not global infeasibility. The feasible cells
  are feasible at every ε. The QP realizes the targets to 1e-12 after
  re-encoding, keeps the support, and moves 0.40–0.49× as much as the decoder
  edit. For control features it moves 3.3–6× more than the decoder edit, whose
  requested changes are tiny.
- **Exact-code realizability LP**: unfinished. A first attempt with a
  zero-objective LP stalled in constraint generation (1 h 49 min, stopped). The
  second attempt, with a closed form first and an L1 objective, completed 33 of
  1,800 sampled requests within its 25-minute stage budget. All 33
  (development, control, own-support) were realizable.

## Phase 3: small GPU validation (one RTX 3070)

`scripts/twos_edit_validation_gpu.py`, analysis
`scripts/twos_phase3_analysis.py` (outputs in `phase3_smoke/` and `phase3/`).

**Population.** 4 hash-ordered attack windows per confirmation user (32), each
matched to the same user's nearest benign window by day and window time, with
no scores used and no duplicates or exclusions. Donors are 4 benign windows per
team from its never-malicious users (metadata only). The prototype is the team
mean of package-4 prototypes.

**Conditions**, all in the same batches: zero; reconstruction only; union and
own-support residual-preserving decoder edits; the own-support minimum-movement
projection (no rescaling after solving; cell inequalities not enforced); and
for each of the three edits an isotropic per-token random control and a
Gram-preserving random rotation. The rotation keeps per-token norms and every
cross-token inner product, so it preserves directional coherence.

**Smoke test** (2 pairs, batch 1): 19.5 s load, 44 forwards in 5.0 s, peak
2.2 GiB, 38 s wall. Validity:

- Token lengths equal the cache.
- Fresh versus cached delta: median relative difference 0.15%; SAE support
  Jaccard median 1.0.
- A zero edit equals the unhooked model exactly.

**Full run** (32 pairs, 8 users, batch 4): 704 forwards in 42 s, peak 2.8 GiB,
78 s wall. A zero edit again equals the unhooked model exactly. At batch 4,
fresh deltas differ from the cache by 4.3% (median). 22% of the non-selected
SAE support changes (Jaccard median 0.78), while the selected-feature support
is identical on every receiver. bf16 batch-shape noise alone therefore sets a
floor for any support-change diagnostic.

Median edit sizes on active receivers: decoder union and own 45.8,
projection 17.2, reconstruction 64.7.

Behaviour-token loss, mean of 8 user means, descriptive 95% user bootstrap;
all 64 receivers kept, no-ops included:

| contrast | attack windows | same users' benign windows |
|---|---|---|
| own decoder edit − isotropic random | **+0.0051 [+0.0021, +0.0085]**, 7/8 users | +0.0047 [−0.0004, +0.0120] |
| own decoder edit − coherent random | +0.0028 [−0.0006, +0.0076] | +0.0051 [+0.0011, +0.0115] |
| union decoder edit − isotropic random | +0.0030 [+0.0013, +0.0048], 7/8 | +0.0031 [−0.0010, +0.0092] |
| **projection − isotropic random** | +0.0006 [−0.0017, +0.0029] | −0.0005 [−0.0020, +0.0008] |
| **projection − coherent random** | −0.0005 [−0.0024, +0.0014] | −0.0003 [−0.0008, +0.0002] |
| own decoder edit − projection (same realized target change) | +0.0021 [−0.0017, +0.0070] | +0.0054 [+0.0018, +0.0102] |
| union − own | −0.0011 [−0.0025, +0.0000] | −0.0003 [−0.0010, +0.0000] |

Attack-minus-benign differences all include zero. The active-only analysis (7
users with active selected features) gives the same pattern, for example own
decoder − isotropic random +0.0086 [+0.0047, +0.0125]. Profile-token outcomes
are degenerate: edits reach profile positions for at most 2 users. Per-user
values are in `phase3/analysis_phase3.json`.

**Reading.**

- **Realizing the requested code change does not produce the effect.** Two
  edits that realize the same target change (realized fraction 1.0) behave
  differently. The decoder edit raises behaviour loss beyond a size-matched
  independent random direction. The minimum-movement projection does nothing
  beyond random, and about nothing in absolute terms.
- **Direction and size are confounded here.** The projection is 2.7× smaller,
  and no decoder edit scaled to its size was run. A scaled decoder edit would
  no longer realize the target, so the comparison was made, as the
  specification allows, at a common target change rather than a common
  budget.
- **Coherence matters.** Coherence-preserving random rotations perturb more
  than independent random directions, which weakens the decoder edits' excess
  on attack windows (the intervals include zero).
- **Nothing is attack-specific.** No attack-specific effect appears.

## Go / no-go

**Direction A (encoder-realizable, constrained feature edits): no-go as a
main contribution; keep the cheap checks.**

- Worth keeping as routine reporting:
  - the over-k certificate (exact and free; it flags 15–71% of union edits);
  - the realized-versus-requested change;
  - support Jaccard against a measured numerical noise floor.
- The constrained edits are well-posed and verifiable (target errors of 1e-12
  after re-encoding). But the property they enforce, encoder consistency with
  minimal movement, removes the model effect rather than isolating it.
- The diagnostic question in the specification was whether encoder mismatch
  or collateral change explains effects beyond magnitude, support and
  reconstruction. It does not appear to. Own-support edits have no
  realizability problem, yet their effect exceeds random. The union edit's
  frequent unrealizability adds nothing (union − own ≈ 0).
- Remaining alternatives: the effect tracks decoder-direction displacement,
  plain edit size (untested at a matched budget), or cross-token coherence.
  All come from 8 users, previously examined.

**Direction B (donor-policy sensitivity LP): go, as a low-cost robustness
appendix; no-go as a novelty claim.**

- It is exact, fast and verifiable.
- It gives a breakdown radius that separates policy-robust conclusions (CERT
  behaviour: no reweighting of the bank flips it) from fragile ones (CERT
  profile: 1–2% of donor weight; TWOS behaviour: 0.24%).
- It isolates the donor-independent reconstruction term (TWOS profile).
- The mathematics is a standard LP, and generic intervention-distribution
  sensitivity is covered by Certified Interventional Fidelity and by
  distributionally robust causal abstraction. A methods claim would need a
  scientifically defended admissible policy set (metadata balance,
  concentration caps) plus held-out validation.

## Compute and unfinished items

| item | GPU | wall |
|---|---|---|
| Phase 3 smoke (RTX 3070, GPU 0) | 1 | 38 s |
| Phase 3 full (RTX 3070, GPU 0) | 1 | 78 s |
| **total GPU** | | **0.03 GPU-h** (cap 2; round total 0.39 of 8) |
| Phase 1, CERT extension (local CPU) | 0 | about 2 min |
| Phase 2 (Aquaman CPU): stopped slow attempt + completed run | 0 | 1 h 49 min + 40 min |

Unfinished or not run:

- The exact-code realizability LP (1,767 of 1,800 sampled requests not run).
- A constrained QP condition in Phase 3 (not applicable to the deletion
  requests; the projection was used instead).
- A decoder edit size-matched to the projection.
- Profile-token outcomes in Phase 3 (degenerate).
- Any bound on the historical best-candidate estimand.

There were no GPU failures.

## Commands

```
python3 scripts/twos_donor_policy_lp.py results/twos_corrected_bounded/token_delta_sae_causal_candidate_rows.csv results/twos_corrected_bounded/endpoints.json results/twos_feasibility_2026_09_26/phase1
python3 scripts/twos_donor_policy_plot.py results/twos_feasibility_2026_09_26/phase1/donor_policy_bounds.json results/twos_feasibility_2026_09_26/phase1/donor_policy_bounds.svg
python3 scripts/cert_donor_policy_lp.py <package-4 full candidate rows> results/cert_package4_token_class/endpoints_alpha1.0.json results/twos_feasibility_2026_09_26/phase1_cert_extension
# Aquaman, /data/suramya/insider_mi/twos_feasibility_2026_09_26/repo, cert-venv, D=/data/suramya/insider_mi
python scripts/twos_edit_realizability.py --data-dir $D/twos_work/audit_pool_v3 --extract-dir $D/twos_work/v3_deltas_s42 \
  --frontier-dir $D/review_2026_09_24/twos_s42_reselect_discovery --split-dir $D/review_2026_09_24/twos_split \
  --candidate-rows ../inputs/token_delta_sae_causal_candidate_rows.csv --out-dir ../phase2 --stage-budget-s 1500
CUDA_VISIBLE_DEVICES=0 HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/twos_edit_validation_gpu.py \
  --config $D/twos_work/qwen3b_twos.yaml --data-dir $D/twos_work/audit_pool_v3 --adapter-dir $D/lm/twos_adapter_s42_fixed \
  --extract-dir $D/twos_work/v3_deltas_s42 --frontier-dir $D/review_2026_09_24/twos_s42_reselect_discovery \
  --split-dir $D/review_2026_09_24/twos_split --out-dir ../phase3 --batch-size 4      # smoke: --smoke 2 --out-dir ../phase3_smoke
python3 scripts/twos_phase3_analysis.py results/twos_feasibility_2026_09_26/phase3/phase3_rows.csv results/twos_feasibility_2026_09_26/phase3/analysis_phase3.json
```

Closest prior work, from the research note: SplInterp (TopK polyhedral
regions), SAE-Targeted Steering (pseudoinverse and targeted effects),
post-intervention recovery (null-space preservation of SAE values), Certified
Interventional Fidelity, and distributionally robust causal abstraction. None
of the procedures here goes beyond them methodologically.
