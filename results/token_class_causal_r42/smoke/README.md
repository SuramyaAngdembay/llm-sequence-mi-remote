# CERT r4.2 token-class causal — smoke run (job 20840686)

Pre-registration: `docs/PREREGISTRATION_CERT_PACKAGE4.md`, rule 6 ("smoke first
on gpu-debug with --max-receivers 24, one alpha, one context mode").

**Verdict: every rule that gates the full run holds.** Verified from the saved
rows by `scripts/verify_token_class_smoke.py`, not inferred from exit status.

| pre-registered rule | check | result |
|---|---|---|
| 5 — per-class sums reconstruct the scalar NLL | max abs diff over 1,186 rows | **1.788e-07** |
| 5 — patched per-class counts equal the base's | asserted in-run; a violation aborts the job | job completed |
| 1 — the base is recomputed, not cached | `base_is_recomputed` on every row; `delta == delta_recomputed` | **true on all rows** |
| 1 — the discrepancy that rule exists for | cached minus matched-batch base | mean **1.221e-02**, max 3.590e-02 |
| receivers restricted to held-out users | log line + 12 distinct users among 24 receivers | confirmed |

The cache discrepancy is the same order as on TWOS (1.045e-02) and larger than
the effects the full run will measure, which is why rule 1 of
`docs/PREREGISTRATION_CERT_PACKAGE4.md` (commit `c0131c4`) was fixed before
either run.

Token-class shares of scored targets — DAY 10.9 %, PSY 7.9 %, SESCOUNT 3.1 %,
SES 78.1 % — match the manuscript's stated SES token share of 0.75–0.78.

## The first attempt failed, usefully

Job 20838328 died in 81 s: combining `--max-receivers` with the held-out user
file drew the shared receiver sample from all positives and then filtered it,
producing a lookup for rows the filter had removed. Fixed in commit `1dbfb76` (see
`scripts/tests/test_receiver_sampling.py`, verified to fail on the unfixed
code). Had it not raised, the run would have scored fewer receivers than
requested without saying so.

## Full run

Job **20840922**, `gpu` partition, 24 h wall, submitted 2026-09-21 with every
parameter as recorded in `docs/PREREGISTRATION_CERT_PACKAGE4.md` (commit `c0131c4`). Output:
`outputs/token_class_causal_r42_confirmation/l26_m02_k04_top5_control5_active`.
