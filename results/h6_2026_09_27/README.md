# H6 pilot (exploratory): where the adapted model's profile dependence arises

Protocol: `docs/H6_PROFILE_PATCHING_PROTOCOL_2026-09-27.md`, frozen at commit
`ab7299f` before scoring, plus dated pre-scoring notes. Analysis:
`scripts/analyze_h6.py`, committed before any CERT output. **Exploratory
development data.** The 30 discovery users' positive rows were used to rank
the SAE features.

## Runs

| job | partition, GPU | node | wall | CPU time | role |
|---|---|---|---|---|---|
| 20927962 | gpu-debug, A100 40 GB | g005 | 9 min 25 s | 10 min 9 s | **primary** (`out/`) |
| 20927963 | gpu, A100 40 GB | g012 | 9 min 17 s | 8 min 37 s | duplicate (`out_gpu/`) |
| 20928056 | ai, H100 | h016 | 5 min 17 s | 4 min 35 s | duplicate (`out_ai/`) |

The first two copies started in the same second. The watcher missed that
start because the workstation was offline. The H100 copy started after both
had ended. The primary copy was declared in the ledger before any output was
read. The duplicates serve only as reproducibility checks.

- **Identities.** The adapter, SAE and data sha256 hashes match those
  recorded for the earlier pilots and are in each log.
- **Software.** transformers 5.16.1, peft 0.20.0, torch 2.5.1+cu121,
  bitsandbytes 0.50.2.
- **Command** (from `slurm/anvil_friend/h6_profile.sbatch`):

  ```
  python -u scripts/h6_profile_patching.py \
    --config configs/qwen3_8b_qlora_session_targeted.yaml \
    --data-dir $S/session_jsonl_r42 --adapter-dir $S/adapter --extract-dir $S/token_deltas \
    --frontier-dir $S/frontier --split-dir $S/user_splits_r42 --out-dir $P/out \
    --batch-size 6 --smoke-pairs 2 --wall-budget-s 1500
  python3 scripts/analyze_h6.py results/h6_2026_09_27/out
  ```

## Validity

- **Receivers.** 120 receivers: 60 attack days and 60 matched benign days of
  30 users. No exclusions and no session-alignment failures. Median
  profile-token alignment coverage is 1.00. The manifest is byte-identical to
  the CPU dry run.
- **Smoke test** (4 receivers, in the same job) and full run: zero-hook and
  self-patch losses equal the unpatched loss exactly (max abs 0.0) for all
  120 receivers. The stopping threshold was 1e-5.
- **Full cohort.** Wall-time trimming removed nobody.
- **Reproducibility.** The two A100 copies agree bit for bit. The H100 copy
  differs per receiver by a median of 0.003 nats (max 0.028). Every endpoint
  below changes by at most 0.003 on H100, and every interpretation-rule
  outcome is the same.
- **Applied SAE edits.** Applied norm over intended norm has median 1.001
  (max 1.11), with no numerical failures. The selected-feature edit is
  nonzero on only 369 of the 23,267 session tokens (1.6%), and 31 of 120
  receivers are no-ops. Realization is approximate:

  | edit | tokens edited | normalized target error, median [q90, q99] | support Jaccard, median |
  |---|---|---|---|
  | sel_sess | 369 | 0.10 [1.00, 1.17] | 0.80 |
  | ctrl_sess | 414 | 0.04 [1.00, 1.58] | 1.00 |

  At the 90th percentile the realized code change is as far from the
  intended one as no change at all. The model still received each edit at
  its intended size. The caveat concerns reading the edits as coefficient
  restorations.
- **Full-delta restoration.** fulldelta_sess edits all 23,267 session tokens
  (norm median 22, q99 118). One token reaches 11,992. All 30 users show
  positive rescue, so a single token does not drive the effect.

## Results (behaviour-only loss, nats per token; mean of user means; 95% user bootstrap, 10,000 draws)

**Input effects.** Each row is the variant's loss minus the original's.

| input | adapter on, attack | adapter on, benign | adapter off, attack |
|---|---|---|---|
| R: DAY and PSY replaced | +0.243 [+0.203, +0.283], 30/30 users | +0.280 [+0.239, +0.320] | −0.001 [−0.008, +0.006] |
| D: DAY line only | +0.246 [+0.206, +0.286] | +0.286 [+0.238, +0.335] | +0.005 [−0.000, +0.011] |
| P: PSY line only | +0.004 [−0.003, +0.011] | +0.008 [+0.002, +0.013] | −0.004 [−0.008, −0.000] |
| T: second partner | +0.209 [+0.171, +0.246] | +0.246 [+0.203, +0.287] | +0.002 [−0.004, +0.008] |

**Rescue of the R effect by residual patching from O**, attack days; benign
days are within 0.04 and follow the same pattern. The fraction is rescue
divided by the input effect. The denominators are eligible: 0.243 and 0.280,
with intervals excluding zero.

| hidden state | profile positions | session positions | session positions, patched from T |
|---|---|---|---|
| 2 | +0.219 (fraction 0.90) | −0.000 (0.00) | −0.000 (0.00) |
| 10 | +0.224 (0.92) | +0.001 (0.01) | +0.001 (0.00) |
| 18 | +0.234 (0.96) | +0.005 (0.02) | +0.003 (0.01) |
| 22 | +0.239 (0.98 [0.96, 1.00]) | +0.018 (0.08) | +0.016 (0.06) |
| 26 | +0.069 (0.28) | **+0.204 (0.84 [0.78, 0.90])** | +0.055 (0.23) |
| 30 | +0.057 (0.23) | +0.220 (0.90) | +0.050 (0.21) |
| 34 | +0.025 (0.10) | +0.234 (0.96) | +0.038 (0.16) |

At layer 26, O-sourced minus T-sourced session rescue is +0.149
[+0.117, +0.181] on attack days and +0.170 [+0.132, +0.207] on benign days.

**Component patching at session positions**, attack days. Each row restores
one block's attention or MLP output. Only every fourth block was tested.

| block (0-based) | attention output | MLP output |
|---|---|---|
| 1, 5, 9, 13, 17 | +0.0011 or less, every interval including zero | +0.0011 or less, every interval including zero |
| 21 | +0.013 (0.05) | +0.014 (0.06) |
| 25 | +0.012 (0.05) | +0.071 (0.29) |
| 29 | +0.017 (0.07) | +0.107 (0.44) |
| 33 | +0.092 (0.38) | +0.132 (0.54) |

**Layer-26 SAE interventions at session positions**, attack days:

| intervention | rescue | fraction |
|---|---|---|
| fulldelta_sess: the adapter's whole layer-26 difference | +0.203 [+0.173, +0.232] | 0.83 |
| fulldelta_all: the same, including profile positions | +0.222 [+0.185, +0.259] | 0.91 |
| sel_sess: 5 selected coefficients restored | +0.0008 [−0.0006, +0.0031] | 0.003 |
| sel_all: the same, including profile positions | +0.0008 [−0.0010, +0.0033] | 0.003 |
| ctrl_sess: 5 control coefficients | −0.0002 [−0.0007, +0.0003] | |
| rand_sess: random, size-matched to sel_sess | +0.0013 [−0.0000, +0.0037] | |
| sel_sess − ctrl_sess | +0.0010 [−0.0006, +0.0032] | |
| sel_sess − rand_sess | −0.0005 [−0.0012, +0.0003] | |

Benign days are the same: sel_sess +0.0014 [−0.0003, +0.0035] and
fulldelta_sess +0.235 [+0.200, +0.272].

**Selected-feature codes under the swap.** These are token-level changes at
session positions, layer 26. In absolute terms, features 4596, 3673 and 3455
change by 2.6%, 3.7% and 7.7% of their means. That is no more than the
frequent control features, 6608 (10%) and 6596 (19%).
**Feature 7693 is the exception.** It is active on 166 session tokens of 17
receivers from 5 users in O, and inactive everywhere in R. On those 17
receivers (post hoc; both day types; 5 users), restoring the selected
coefficients gave +0.004 [−0.006, +0.016] against an input effect of +0.173.
A size-matched random edit gave +0.008.

**Attack versus benign (paired, within user).** The R input effect is
−0.037 [−0.077, +0.003], attack minus benign. The T input effect is −0.037
[−0.066, −0.008]. Rescue fractions match between day types, for example
0.838 and 0.832 for session positions at layer 26.

## Reading (against the frozen interpretation rules)

1. **Where the profile information enters session positions (rule 1).**
   Session-position rescue first exceeds profile-position rescue at hidden
   state 26, on both day types. The O-sourced patch clearly beats the
   T-sourced control there. Through hidden state 22, restoring the profile
   positions alone recovers 98% of the effect, and restoring the session
   positions recovers 8%. By 26, the session positions carry 84%. The
   profile-dependent information therefore moves into the session positions
   mainly in blocks 22 to 25. The patched layers bound the transition to that
   window, and nothing here localizes it further.
2. **Components (rule 2).** The attention outputs of blocks 21 and 25 each
   restore about 5%. The move itself must happen through attention, so it
   presumably happens in blocks 22 to 24, which were not tested, or is spread
   across heads and blocks. Late MLPs at session positions (blocks 25, 29
   and 33) each restore 29% to 54%, and so does block 33's attention. These
   single-component rescues overlap and sum to more than the whole. They show
   that late blocks carry the profile-dependent difference forward. They do
   not show that any one component is necessary or unique.
3. **Selected-feature mediation (rule 3).** The second branch applies.
   Selected rescue is near zero and not larger than the control or random
   edits. The full layer-26 adapter contribution restores 83%. **The
   adapter's layer-26 contribution carries the dependence, but the 5 selected
   coefficients do not under this intervention.** Three facts support this:
   - The selected coefficients barely change under the swap.
   - Where one does switch off entirely (7693), restoring it does not help
     either.
   - Restoring the whole residual stream at session positions (+0.204) and
     restoring only the adapter delta (+0.203) agree, so the base model's own
     layer-26 state contributes nothing measurable.

   The caveat is approximate realization of the coefficient edits (q90
   normalized error 1.0). This does not show that no SAE features mediate the
   dependence. Only these 5 were tested.
4. **Attack versus benign (rule 4).** No attack-specific rescue is detected.
   Attack days are, if anything, slightly less profile-dependent (T input
   effect −0.037 [−0.066, −0.008]; the R interval includes zero). That
   direction matches Pilot 2's F2. It is one of many exploratory contrasts.

**Profile dependence comes from the organisation line.** Replacing only the
DAY line (role, department, team, units, project, admin flag) reproduces the
whole effect (+0.246 against +0.243). Replacing only the personality line
does almost nothing (+0.004). A second unrelated partner produces almost the
same effect (+0.209). The dependence is therefore generic to foreign
organisational context. Which DAY fields matter was not tested. The base
model shows none of this.

## What this does not establish

- Whether the dependence harms detection or robustness. No detection
  endpoint was computed.
- A circuit or path. These are activation and component patches, not path
  patching.
- Which heads or which of blocks 22 to 24 move the information.
- Whether other SAE features at layer 26 carry the dependence.
- Anything confirmatory, since the users are development data.

## Files

- `out/`: primary copy.
  - `h6_rows.csv`: per receiver, per condition, behaviour-only loss and SES-only loss.
  - `h6_manifest.json`: receivers, partners, alignment.
  - `h6_smoke.json`.
  - `h6_verification.json`: applied-edit and realization distributions.
  - `h6_token_codes.npz`: selected and control codes at aligned session
    positions, O and R.
  - `analysis_h6.{json,txt}`: every endpoint, with per-user values.
- `out_gpu/` and `out_ai/`: duplicate copies, used for reproducibility only.
- `logs/`: all three jobs' stdout and stderr.
- `dryrun/`: the CPU dry-run manifest.
- `twos_codepath_test/`: the pre-scoring execution test on TWOS; not results.
