# Phase C — runnable specification: training-loss × scoring factorial

Prepared 2026-09-17; the **3B arm was authorized and launched** the same day
(job 20810414, ~8-9 SU, behind the loss-path gate below). The **8B arm is
held** pending (a) the 3B run behaving as designed and (b) the Phase A/B
scoring result, since if score masking alone already helps, the 8B training
spend should be judged against that.

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
  - `scored_targets` (HF default): labels carry −100 at profile targets, so the
    trainer normalizes over behaviour targets only. This **up-weights each
    behavioural token by 1/(1−p) = 1.174** (p = 0.1485 measured) — a
    gradient-scale change that could become the explanation.
  - `all_targets` (**use this for C/D**): labels stay unmasked so the trainer's
    `num_items_in_batch` counts every non-pad target across the accumulation
    window; a separate `profile_mask` is applied inside `compute_loss`, which
    divides by that window count. A retained token then keeps exactly the
    weight it has in the unmasked recipe. With no mask this is arithmetically
    identical to the HF default.

  Normalizing per micro-batch here is wrong and was caught by the gate:
  `transformers` 5.16.1 skips its own `loss / gradient_accumulation_steps`
  whenever `num_items_in_batch` is supplied, so a micro-batch mean comes out
  `grad_accum` times too large (measured 4.0x at grad_accum 4, job 20810414).

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

`slurm/anvil_friend/phase_c_3b.sh` (launched as job 20810414) runs the
validation gate, then trains the masked adapter, builds the identical audit
pool, scores it with the class decomposition and evaluates the views, so cells
C and D come out of one job. `slurm/anvil_friend/train_target_mask.sh` is the
generic two-scale form; for 8B it must be split into a 4-GPU training job and a
separate 1-GPU scoring job (see Cost).

## Cost (measured rates, verified balances 2026-09-17)

Billing on Anvil is 1 SU per GPU-hour (`billing=1` with `gres/gpu=1`;
`billing=4` with `gres/gpu=4`).

Measured, not guessed: the 8B factorial logs report **1.07 s/it**, and one
epoch of 300k examples at effective batch 16 is **18,750 steps**, so 8B
training is ~5.6 h of wall time on 4 GPUs. An earlier estimate of ~52 SU was
wrong: it came from the *whole* 13 h 38 m factorial job, which also ran
token-level delta extraction (base + adapted, three layers, batch 1) and SAE
fitting. Training alone is about a third of that.

| step | resource | estimate |
|---|---|---|
| 3B loss-path validation (2 × 200 steps) | 1×A100 | ~0.5 SU |
| 3B masked adapter (18,750 steps) | 1×A100 | ~7 SU |
| 3B scoring of the pool + views | 1×A100 | ~1 SU |
| **Phase C at 3B** | | **~8–9 SU** (12 h wall cap) |
| 8B masked adapter (18,750 steps @ 1.07 s/it) | 4×H100 | ~22 SU |
| 8B scoring of the pool + views (separate 1-GPU job) | 1×A100 | ~2 SU |
| **Phase C at 8B** | | **~20–25 SU** |

The 8B path is split into a 4-GPU training job and a 1-GPU scoring job so four
H100s are not held idle during single-GPU inference — the mistake the original
factorial runner made.

Available: `tra250034-ai` 732.5 SU (H100), `cis260991-gpu` 400.0 SU +
`tra250034-gpu` 95.2 SU (A100). Phase C at both scales is ~3 % of the H100
balance and ~2 % of the A100 balance.

Order of execution: **3B first**, behind a hard gate that trains 200 steps with
nothing masked under both the default and the custom fixed-denominator loss
path and aborts unless they agree to within 1 % — transformers ≥ 4.46 passes
`num_items_in_batch` and may normalize a second time, which would silently
change the effective learning rate and become the explanation for any
difference between cells. Only after the 3B run behaves as designed does the
8B spend make sense. The 3B `full` adapter already ranks unseen users at 0.926,
so it has little headroom; a null mitigation effect there does **not**
invalidate a remedy for the 8B failure and must not be read as one.

Seeds: one seed per cell answers "did this change anything in this run". It does
not establish robustness for the recipe, and no number of training seeds
increases CERT r6.2's four independent positive users. Repeated LANL folds add
population variation and are the better place to spend seeds later.
