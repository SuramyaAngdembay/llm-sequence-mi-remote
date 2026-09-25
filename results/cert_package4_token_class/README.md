# CERT r4.2 package 4: which token classes do the SAE interventions change?

Full run **20879549** (collaborator allocation, A100, 9 h 22 min, 530,792
candidate rows) and its reconstruction-only control **20898582** (alpha 0,
2,308 rows). Analysis rules: `docs/PREREGISTRATION_CERT_PACKAGE4.md`, the
original declaration (commit `c0131c4`) plus the dated addendum of 2026-09-24
(written before either run produced rows). Environment gate:
`results/cert_package4_env_gate/GATE_REPORT.txt` (passed).

## Gates

| gate | result |
|---|---|
| per-batch reconstruction and count assertions | both runs completed; nothing raised |
| environment | collaborator smoke matched the owner smoke on all 1,186 rows (max per-row delta difference 9.5e-07) |
| feature sets | top5 `[4596, 7693, 2302, 3673, 3455]`, control5_active `[6596, 8017, 6608, 2765, 886]`, identical in both runs |
| receivers | 638 confirmation days from 30 users; per mode dept 605/29, dept_role 528/27, role 528/27, team 638/30 (≥ 8 users everywhere) |
| baseline | fresh recomputation on every row; no same-user donors |
| reconstruction control coverage | 638 of 638 receivers, both arms |
| same interventions as the published run (job 19379904) | all 530,792 candidate keys (mode, set, donor type, receiver, donor, alpha) identical |

## Primary result (pre-registered comparison, addendum rules)

Benign donors, alpha 1, per receiver the mean over all candidate donors;
mean of receiver-user means with a 95% cluster bootstrap over users. Deltas are
per-token loss changes, patched minus fresh base: positive means the day looks
more anomalous.

| context | behaviour: selected − control | profile: selected − control | direct profile − behaviour |
|---|---|---|---|
| dept | **+0.0172** [+0.0107, +0.0238], 24/29 users + | +0.0069 [−0.0064, +0.0290] | −0.0104 [−0.0267, +0.0144] |
| dept_role | **+0.0158** [+0.0088, +0.0231], 20/27 + | +0.0086 [−0.0058, +0.0333] | −0.0073 [−0.0251, +0.0202] |
| role | **+0.0156** [+0.0086, +0.0227], 20/27 + | +0.0082 [−0.0057, +0.0317] | −0.0073 [−0.0249, +0.0188] |
| team | **+0.0179** [+0.0110, +0.0249], 25/30 + | +0.0065 [−0.0056, +0.0266] | −0.0114 [−0.0270, +0.0115] |

- **Reconstruction cancels additively here, and only additively.** Every receiver day has an active feature of both arms, so both arms reconstruct the same days and the reconstruction-only (alpha 0) effect is identical: behaviour +0.0438, profile −0.0967, full +0.0160 (team receivers). The selected-minus-control differences are therefore the same raw and net of reconstruction. This removes differential reconstruction coverage as an explanation. It does not remove reconstruction–edit interactions: the contrast is L(h+R+E_sel) − L(h+R+E_ctrl), not L(h+E_sel) − L(h+E_ctrl). *(Corrected 2026-09-25.)* Absolute arm effects are dominated by reconstruction and are not read mechanistically. The alpha-0 run was also scored in differently composed batches: its unpatched scores differ from the full run's by about 0.007 nats on profile tokens and 0.0005 on behaviour tokens (user means), so absolute alpha-1-minus-alpha-0 values are not isolated feature edits.
- **Net of reconstruction**, the selected edit raises behaviour loss by +0.017 to +0.019 and the control edit by about +0.001.
- **No negligible claims.** No view's 90% interval lies within the declared ±0.0002 margin.
- **Anomalous donors** give the same pattern: behaviour +0.0137 to +0.0151, every interval excluding zero; profile +0.0056 to +0.0073, every interval including zero.

**All four alphas** (benign donors, net of reconstruction, selected − control):

| alpha | behaviour, range over the 4 modes | profile, range over the 4 modes |
|---|---|---|
| 0.25 | −0.0007 to −0.0005 (dept interval just excludes zero; others include it) | +0.0039 to +0.0048, all include zero |
| 0.5 | −0.0011 to −0.0008, all include zero | +0.0056 to +0.0074, all include zero |
| 0.75 | +0.0047 to +0.0059, all exclude zero | +0.0068 to +0.0089, all include zero |
| 1.0 | +0.0156 to +0.0179, all exclude zero | +0.0065 to +0.0086, all include zero |

**The primary quantity is inconclusive.** The pre-registered primary quantity is the
direct comparison (last column): its interval includes zero in every context.
A detectable behaviour contrast beside an uncertain profile contrast does not
show that the two differ. *(Added 2026-09-25 after external review; the
original text led with the behaviour column.)*

**Against the pre-registered prediction**, which was a negative profile contrast
excluding zero, a behaviour contrast an order of magnitude smaller, and a
monotone profile dose-response: **contradicted.** The profile contrast is
positive and never distinguishable from zero. The behaviour contrast is large
at full strength and sign-changing across alphas. As the pre-registration
requires, this is reported as-is. The uncertain profile component alone does
not refute a positive profile effect, and no profile effect is shown to be
absent. Its motivation, the TWOS result, was already
void (invalid selection).

## Historical endpoint: exact reproduction

The published best-candidate donor difference-in-differences is reproduced to
six decimals when the best candidate is chosen with one base per receiver, as
the published run's cached base did:

| context | published | reproduced (constant base) | with the per-batch fresh base |
|---|---|---|---|
| dept | 0.000758 | 0.000758 | 0.000770 |
| dept_role | 0.000446 | 0.000446 | 0.000461 |
| role | 0.000340 | 0.000340 | 0.000347 |
| team | 0.000398 | 0.000398 | 0.000577 |

The per-batch base of one receiver usually varies by about 6e-08, but by up to
0.037 nats across batches. That changes which candidate is "best" when
candidates are close; it explains the team gap. Mean-based estimands are
insensitive to this.

## Secondary: the published endpoint depends on keeping the best candidate

Motivated before results by issue-ledger entry C6; not part of the
pre-registered primary. Full score, difference-in-differences
`(δ_sel,anom − δ_sel,ben) − (δ_ctrl,anom − δ_ctrl,ben)`, receiver-weighted like
the published statistic (`scripts/did_estimands.py`, output in
`did_estimands.txt`):

| context | best of donors and alphas (published) | best of donors at alpha 1 | **mean over donors at alpha 1** | best, candidate counts equalised |
|---|---|---|---|---|
| dept | +0.00077 | +0.00089 | **−0.0045** | +0.00077 |
| dept_role | +0.00046 | +0.00227 | **−0.0017** | +0.00028 |
| role | +0.00035 | +0.00214 | **−0.0016** | +0.00021 |
| team | +0.00058 | +0.00234 | **−0.0020** | +0.00045 |

On the user-mean scale the average-edit value for dept is −0.0028
[−0.0045, −0.0012], with 18 of 29 users negative. The other three modes are
negative with intervals that include zero.

- **Averaging reverses the published sign.** Averaged over candidate donors at full strength, a benign-donor edit of the selected features raises the score *more* than an anomalous-donor edit, relative to controls.
- **Unequal candidate counts are not the cause.** Equalising benign and anomalous candidate counts leaves the best-candidate value almost unchanged.
- **The positive published value reflects best cases.** Keeping the minimum of up to 64 edits per receiver selects the most favourable tail. At alpha ≤ 0.5 the selected edits lower behaviour loss slightly relative to controls.

## Reading

**Established, on held-out confirmation users (held out for feature selection
only; one adapter).**
- At strengths 0.75 and 1, edits of the selected features raise **behaviour-token** loss more than edits of the control features, by about 0.005 and 0.017 nats per token beyond reconstruction with benign donors, and about 0.003 and 0.015 with anomalous donors. Every one of these intervals excludes zero, in every context mode. Four of the five selected features activate on session tokens.
- At strengths 0.25 and 0.5 the difference is small and slightly negative, meaning selected edits lower behaviour loss a little relative to controls. It is mostly within noise.
- The profile-token difference is uncertain at every strength (every interval includes zero), and the direct profile-versus-behaviour comparison is inconclusive. Neither a profile effect nor its absence is established.

**Not established.**
- *That the features carry anomaly-specific information.* Their edits are larger than control edits: the project's earlier measurements found selected-feature edits several times larger in residual-stream norm. So the behaviour effect may partly be edit size. Every receiver is a malicious day, so the effect is also not shown to be specific to anomalies. *(Corrected 2026-09-25.)* `scripts/select_matched_controls.py` matches proxies (activation frequency, mean coefficient, decoder norm without `x_std`) and its existing report concerns a different top-5 set, so it is not a balance check for this experiment. The test is a residual-preserving edit against norm- and position-matched random directions with matched benign receivers, with measured edit sizes (exploration pilot 1, `docs/HYPOTHESIS_LEDGER_2026-09-25.md`).
- *That editing toward benign values "repairs" malicious days.* Only the best-of-64 estimand says so; the average edit says the opposite.
- *Where the profile shortcut arises inside the model.*

**Structural note** (raised before these results in the session record; not
part of the pre-registration; **corrected 2026-09-25**). The model predicts each
token from earlier tokens only, and profile lines precede session lines. So
once reconstruction is removed, profile-token effects can come only from edits
at organisation or personality positions. The earlier version of this note said
that in this set this means feature 2302. That was wrong. The patch edits
*every* selected coordinate at the union of the set's active positions, including
coordinates that were zero there. At the organisation tokens where 2302 fires,
all five coordinates are written with donor values, so a profile effect cannot
be assigned to 2302 from this joint intervention. The profile-versus-behaviour
comparison remains asymmetric by construction.

**Bootstrap draws.** The pre-registration declared 5,000 draws and the
analyzer used 10,000 (reported above). The 5,000-draw outputs are in
`bootstrap_5000/`, with the interval-by-interval comparison in
`bootstrap_5000/SENSITIVITY.txt` (`scripts/compare_bootstrap_draws.py`).
Bounds move by at most 0.0063 nats. 17 of 2,160 intervals change whether they
exclude zero, each with a bound within 0.0006 of zero; all are absolute
single-arm effects, secondary views, or behaviour differences at alpha 0.5
(for example, anomalous-donor behaviour at alpha 0.5 in dept_role and role just
excludes zero with 10,000 draws and just includes it with 5,000). No primary
quantity changes. `did_estimands.txt` also gains a `mean@all` row (mean over
donors and all four alphas, exploratory, added 2026-09-25): negative in every
context, department user-mean −0.0017 [−0.0033, −0.0005], the others
including zero.

## Files

`endpoints_alpha{0.25,0.5,0.75,1.0}.json` and `report_alpha*.txt` are the
pre-specified analyzer outputs (`--recon-run` with the alpha-0 run,
`--margin 0.0002`). `did_estimands.txt` is the secondary estimand table.
`INPUT_HASHES.txt` gives the sha256 of both runs' candidate rows. The rows
themselves are 335 MB and 1.5 MB CSVs, not tracked in git; they are on the
collaborator scratch under `pkg4_out/confirmation/` and
`pkg4_alpha0/confirmation_alpha0/`. The job records are included.
