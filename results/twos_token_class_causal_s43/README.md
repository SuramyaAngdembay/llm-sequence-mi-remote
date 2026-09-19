# TWOS token-class causal — seed 43 (the replication that failed)

Run 2026-09-19 on Aquaman, zero cluster SU. This is the independent replication
of `results/twos_token_class_causal/` (seed 42), launched specifically to test
the "one seed" limitation that run declared.

**It does not replicate. Every sign reverses.** The seed-42 write-up carries the
comparison table and the corrected reading; this directory holds the seed-43
artifacts.

## What differs between the two runs

Only the pipeline seed. Same layer (24), latent_mult (4), k (8), feature sets
(`top5` vs `control5_active`), alphas (0.25/0.5/0.75/1.0), context mode (`team`),
data (`audit_pool_v3`, 2,275 examples), receivers (138 positive, 16 users) and
candidate rows (35,328 in both).

Different: the adapter (`v3_adapter_seed43`), its delta cache
(`v3_deltas_s43`), and its SAE (`v3_sae_s43`). `top5` is therefore a different
set of five features.

## Result

top5 minus control, pooled over alphas, clustered by the 16 receiver users:

| view | delta | 95% interval |
|---|---|---|
| psychometric only | +0.0120 | [+0.0014, +0.0252] |
| profile only | +0.0109 | [+0.0024, +0.0203] |
| day fields only | +0.0088 | [−0.0082, +0.0290] |
| full | +0.0030 | [+0.0003, +0.0065] |
| behaviour, SES only | −0.0028 | [−0.0061, −0.0004] |
| behaviour only | −0.0024 | [−0.0053, −0.0003] |

Dose-response, profile only: +0.0087 → +0.0104 → +0.0111 → +0.0133 across
α = 0.25 / 0.5 / 0.75 / 1.0. Behaviour only: −0.0024 → −0.0023 → −0.0024 →
−0.0026, every interval excluding zero.

On this seed the selected features **improve behaviour-token prediction and
worsen profile-token prediction**, with a monotone dose-response, which is the
reverse of seed 42 on every count.

## What this establishes

Not that seed 43 is right and seed 42 wrong. That **the sign of this measurement
is not stable across the pipeline seed**, so neither run licenses a claim about
which token class delta-SAE features act on.

It also retires one piece of evidence. A monotone dose-response across patch
strength was the strongest argument that the seed-42 effect was a real causal
handle rather than noise. Seed 43 produces an equally clean dose-response in the
opposite direction, so dose-response alone does not distinguish a real effect
from a seed artifact here.

## What still holds

The measurement machinery, which is independent of the result: per-class sums
reconstructed the scalar NLL on every batch of both runs, and patched per-class
target counts equalled the base's throughout. The decomposition is correct; it
is the thing being decomposed that is unstable.

## Files

`token_delta_sae_causal_candidate_rows.csv` carries base/patched/delta for all
six views per intervention. Reproduce with
`scripts/analyze_token_class_causal.py results/twos_token_class_causal_s43
[--by-alpha]`.
