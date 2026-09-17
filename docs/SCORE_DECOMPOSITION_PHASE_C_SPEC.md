# Phase C — runnable specification: training-loss × scoring factorial

Prepared 2026-09-17. **Not launched.** Training on the collaborator's Anvil
allocation needs an explicit go-ahead; the scope and cost below are stated so
that decision can be made on numbers.

This design replaces the earlier context-versus-scoring 2×2, which was invalid:
"profile absent but profile tokens scored" is not a cell that exists in an
ordinary autoregressive input.

## Design

Profile context is **present in every cell**. Only two things vary: whether
profile tokens contribute to the *training* loss, and whether they contribute
to the *score*.

|                                   | Score all targets | Score behaviour targets only |
|-----------------------------------|-------------------|------------------------------|
| Train on all target losses        | **A** current baseline | **B** score masking only |
| Train on behaviour losses only    | **C** training mask only | **D** both |

- **A** = existing `adapter_full`, view `full`. No new work.
- **B** = existing `adapter_full`, view `behavior_only`. Produced in Phase A/B.
- **C** = new adapter (`--target-loss-mask profile`), view `full`.
- **D** = same new adapter, view `behavior_only`.

So Phase C needs **one new training run per scale per seed**, each scored
twice. The existing `adapter_no_profile` (context removed entirely) stays as a
separate practical baseline, and `adapter_no_psy` remains available.

Identification:

- **A → B** isolates score inclusion, with the model and the input fixed.
- **B → D** isolates the training objective, with input context *and* scored
  targets fixed. This is the clean contrast.
- **A → C** changes the objective under full scoring. Diagnostic only: C scores
  profile targets the adapter was never trained to predict, so it is not a
  sensible deployment detector.
- **C → D** is score inclusion for the masked adapter.

**What masking does not do.** Setting profile *targets* to `-100` does not cut
gradients through the profile *context*: behavioural-token losses can still
train the adapter to use profile information. C/D therefore test whether being
trained to predict profile tokens is harmful — not whether identity information
is absent from the adapter.

## Implementation (already in the repository)

`scripts/train_qlora.py`, two new flags; both defaults reproduce the existing
recipe byte-for-byte:

- `--target-loss-mask {none,profile}`. `profile` sets `labels[t] = -100` at DAY
  and PSY tokens. HF's causal LM loss compares `logits[..., :-1, :]` with
  `labels[..., 1:]`, so this removes the prediction *of* those tokens and
  leaves `input_ids` untouched. Classes come from
  `token_class_decomposition` (line **prefix**, never line index), the same
  library used by the scorer, so the two phases cannot drift apart. Masked
  runs switch the collator to `DataCollatorForSeq2Seq(label_pad_token_id=-100)`
  so labels pad correctly.
- `--loss-denominator {scored_targets,all_targets}`.
  - `scored_targets` (HF default): mean over non-ignored targets. Under
    masking this **up-weights each behavioural token by 1/(1−p)**, where p is
    the profile share of targets — a gradient-scale change that could become
    the explanation.
  - `all_targets` (**use this for C/D**): sum of target losses divided by the
    number of non-pad targets, so a retained token keeps exactly the weight it
    has in the unmasked recipe and masking only drops the profile terms. With
    no mask the two are arithmetically identical.

`train_summary.json` records `target_loss_mask`, `loss_denominator`, and the
measured `masked_target_frac` (p), so the gradient scaling is documented rather
than inferred. `scripts/tests/test_token_class_decomposition.py` (check 7) pins
this arithmetic.

Everything else is held fixed: same `jsonl_r62_full` data, same 300k/2k caps
and example order, same seed 42, same effective batch 16, same LoRA config,
same schedule, same `max_seq_len`.

## Validation before the real run

1. Cell A reproduction: train ~50 steps with `--target-loss-mask none
   --loss-denominator all_targets` and confirm the loss curve matches the
   default `scored_targets` path (they are arithmetically identical when
   nothing is masked). Guards the custom `compute_loss`.
2. Report the measured p (profile share of training targets) from
   `masked_target_frac`; Phase A/B gives the same quantity on the eval pool, so
   the two should agree to within sampling noise.
3. Confirm on a handful of batches that masked positions are exactly the DAY and
   PSY tokens and that `input_ids` are unchanged.

## Runner

`slurm/anvil_friend/train_target_mask.sh` (in the repo), invoked as
`sbatch train_target_mask.sh 3b` / `8b`. It trains the masked adapter, builds
the identical audit pool, scores it with the class decomposition, and evaluates
all views, so cells C and D come out of one job.

## Cost (measured rates, verified balances 2026-09-17)

Billing on Anvil is 1 SU per GPU-hour (`billing=1` with `gres/gpu=1`;
`billing=4` with `gres/gpu=4`).

| step | resource | estimate |
|---|---|---|
| 3B masked adapter (18,750 steps, 1 epoch of 300k) | 1×A100 | ~3 SU (~3 h) |
| 3B scoring of the pool + views | 1×A100 | ~1–2 SU |
| 8B masked adapter (18,750 steps) | 4×H100 | ~52 SU (~13 h × 4) |
| 8B scoring of the pool + views | 1×A100 | ~2–4 SU |
| **one seed, both scales** | | **~60 SU** (≈52 H100-SU + ≈8 A100-SU) |

Available: `tra250034-ai` 732.5 SU (H100), `cis260991-gpu` 400.0 SU +
`tra250034-gpu` 95.2 SU (A100). One seed at both scales uses ~7 % of the H100
balance.

Recommended scope to authorize, if Phase A/B motivates it: **3B first** (~5 SU,
validates the implementation and the mechanism cheaply), then **8B one seed**
(~56 SU) only if the 3B run behaves as designed. The 3B `full` adapter already
ranks unseen users at 0.926, so it has little headroom; a null mitigation
effect there does **not** invalidate a remedy for the 8B failure, and must not
be read as one.

Seeds: one seed per cell answers "did this change anything in this run". It does
not establish robustness for the recipe, and no number of training seeds
increases CERT r6.2's four independent positive users. Repeated LANL folds add
population variation and are the better place to spend seeds later.
