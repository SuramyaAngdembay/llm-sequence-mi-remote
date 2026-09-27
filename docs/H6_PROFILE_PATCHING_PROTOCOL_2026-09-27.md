# H6 pilot protocol: where does profile-dependent behaviour prediction arise? (frozen 2026-09-27, before scoring)

Status: **exploratory development pilot.** Written before any H6 output
existed. Changes after scoring will be dated and labelled.

## Question

Where, and through which internal components, does the r4.2 adapted model's
profile-dependent behaviour prediction arise? Pilot 2 established the input
effect: replacing a day's DAY/PSY lines raises the adapted model's
behaviour-token loss by about 0.1–0.2 nats per token, while the base model's
loss barely changes. This pilot assumes neither that the dependence is
harmful nor that the selected SAE features are uninvolved; it tests both
readings.

## Frozen objects

Qwen3-8B, NF4, r4.2 adapter `pkg4_share/adapter` (sha256 recorded in the
manifest). SAE layer 26, m = 2, k = 4 (`pkg4_share/frontier`), selected
`[4596, 7693, 2302, 3673, 3455]`, control `[6596, 8017, 6608, 2765, 886]`.
Collaborator environment on Anvil (`cert-qlora` conda env); one A100.

## Receivers and prior exposure

- **Receivers**: the 30 **discovery** users. For each, up to 2 attack days are
  taken in sha256 order (salt `h6-2026-09-27`). Each attack day is matched to
  the same user's benign day nearest in day index (earlier on a tie, each used
  once).
- **Prior exposure**: these users' positive rows were used to rank the SAE
  features, and earlier full-population searches included them. No profile-swap
  or patching analysis has touched them. They are **development data**, not
  confirmation.

## Profile substitutions

The receiver's `week=` value and every session line are always kept.

- **R (reference)**: DAY and PSY lines from a different-department,
  never-malicious eval user. This is Pilot 2's `swap_other` rule applied to
  new receivers. The partner is the first such user in sha256 order, and the
  partner's day is chosen by sha256.
- **D (organisation only)**: only the DAY line from the same partner (role,
  department, team, units, project, admin flag).
- **P (personality only)**: only the PSY line from the same partner.
- **T (matched control source)**: DAY and PSY lines from the *second*
  other-department partner in the same order.

D and P separate organisational context, which can plausibly inform normal
behaviour, from the psychometric scores. This contrast describes where the
dependence comes from. It does not decide which dependence is harmful; no
detection or robustness endpoint is computed here.

## Alignment

Profile values change token counts, because digits are single tokens. Every
token gets a key `(line, field, part, rank from the end of its part)`:

- `part` is `k` for the prefix word, and for a field's key and "=" (with the
  preceding space);
- `part` is `v` for a field's value;
- `part` is `nl` for the newline.

Positions with equal keys are aligned. Session-line and SESSIONS-line tokens
must align one-to-one with identical token ids, which verifies that the
behaviour targets are unchanged; a receiver failing this is excluded and
recorded. Profile value tokens without a counterpart stay unpatched, and
coverage is reported.

## Position groups and layers

- Position groups: **profile** (DAY and PSY tokens) and **session** (SESSIONS
  and SES tokens).
- Layers (hidden-state index ℓ, hook on block ℓ−1): `{2, 6, 10, 14, 18, 22,
  26, 30, 34}`. That is every fourth layer of 36. It includes the SAE layer,
  26, and the other cached-delta layers, 18 and 34.

## Conditions (all on the swapped input R, adapter on, unless stated)

| name | intervention | type |
|---|---|---|
| O, R, D, P, T; adapter on and off | inputs only | input baselines and input sensitivity |
| resid_prof@ℓ, resid_sess@ℓ | residual stream after block ℓ−1 at aligned positions ← O | activation patching |
| attn_sess@ℓ, mlp_sess@ℓ | attention or MLP output of block ℓ−1 at session positions ← O | component patching (not path patching) |
| resid_prof@ℓ←T, resid_sess@ℓ←T | the same residual patches from the third profile T | matched intervention control |
| sel_sess, sel_all | at layer 26, move the 5 selected SAE coefficients to their values in O at session (or session and aligned profile) positions: h' = h_R + x_std ⊙ D_S (z_O,S − z_R,S). The reconstruction residual is preserved. | selected-feature mediation |
| ctrl_sess, ctrl_all | the same for the 5 control features | matched feature control |
| rand_sess | isotropic random direction per token, norm-matched to sel_sess | size control |
| fulldelta_sess, fulldelta_all | h' = h_R + (δ_O − δ_R): restore the adapter's whole layer-26 contribution | upper reference at the SAE layer |
| zero, self | hook adding zero; resid_sess@26 and resid_prof@10 patched from R itself | implementation checks |

The SAE codes are z = E((δ − x_mean)/x_std), with δ = h_adapted − h_base for
the **same input**: O's codes use O's adapter-off run, and R's use R's. The
realized SAE code is verified after the patch hook and its bf16 conversion:
δ' = h'_captured − h_base(R), re-encoded, and compared coordinate-wise with
the intended code.

Genuine path patching (restricting an effect to a sender-to-receiver edge
with other paths frozen) is **not** implemented. The component patches
restore one block's attention or MLP write at session positions; later
layers still read the swapped profile.

## Outcome and aggregation

- **Primary**: behaviour-only target loss (SESSIONS and SES targets), per
  receiver.
- **Input effect**: E = L(R) − L(O), adapter on (and off).
- **Rescue**: L(R) − L(R + patch), absolute, in nats per token.
- Estimates are the mean of user means, reported separately for attack and
  benign receivers, with a user-clustered percentile bootstrap (10,000 draws,
  seed 42). Also reported: the paired within-user attack-minus-benign
  difference, and per-user values.
- A rescue fraction (rescue / E) is reported only for a condition whose
  input effect has a user-mean of at least 0.05 nats with an interval
  excluding zero. Its interval is a bootstrap of the ratio of user means.

## Interpretation rules

1. **Where profile information moves into session positions.** The layer at
   which resid_sess rescue first exceeds resid_prof rescue is the estimate.
   Read it only if the O-sourced rescue exceeds the T-sourced control at that
   layer.
2. **Component patches.** A positive attn_sess@ℓ rescue means block ℓ−1's
   attention writes profile-dependent information into session positions.
   mlp_sess@ℓ is read the same way. A single-component rescue does not show
   that the component is necessary or unique.
3. **Selected-feature mediation.**
   - "Partial mediation by the tested coefficient restoration" requires
     sel_sess rescue > 0, with its interval excluding zero, and larger than
     both ctrl_sess and rand_sess (paired differences, intervals excluding
     zero).
   - If sel ≈ 0 while fulldelta is clearly positive, the adapter's layer-26
     contribution carries the dependence, but the selected coefficients do
     not under this intervention.
   - Neither outcome proves a complete or unique mechanism.
4. **Attack versus benign.** A claim of attack-specific rescue needs the
   paired difference interval to exclude zero. It is exploratory.

## Stopping and validity rules

- **Stop and diagnose** if the zero-hook or self-patch loss differs from the
  unhooked or unpatched loss by more than 1e-5 for any receiver.
- **Stop and diagnose** if more than 20% of receivers fail session
  alignment.
- **Smoke first, inside the same job**: 2 attack/benign pairs, timed. The
  full cohort is then trimmed, in reverse hash order of users, so that it fits
  the remaining wall time.
- **GPU cap for this follow-up**: 1 GPU-hour in total, including the TWOS
  validation repair. The job's wall limit is 30 minutes on one A100.
