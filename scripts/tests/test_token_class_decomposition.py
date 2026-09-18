#!/usr/bin/env python3
"""Focused correctness checks for `token_class_decomposition`.

Runs with stdlib + numpy only (no torch/transformers), so the class mapping and
accumulation logic can be checked without a GPU:

  1. the naive mean-subtraction formula is wrong and can reverse a ranking;
  2. class sums/counts reconstruct the original mean NLL exactly;
  3. padding and the causal shift are aligned (batched == per-example);
  4. boundary-crossing tokens and truncation behave as documented;
  5. classification survives the no_psy / no_profile line-index shift;
  6. LANL field spans classify identity vs behaviour keys.

GPU-side checks (real tokenizer, reproduction of cached `adapted_nll`) live in
`score_token_class_decomposition.py --self-check`.

Usage:  python3 scripts/tests/test_token_class_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_class_decomposition import (  # noqa: E402
    CERT_CLASSES,
    CERT_VIEWS,
    LANL_CLASSES,
    accumulate_class_losses,
    behavior_score_from_means,
    cert_class_spans,
    cert_subclass_spans,
    classify_subclass,
    classify_tokens,
    lanl_class_spans,
    naive_difference,
    view_score,
)

FULL_TEXT = (
    "DAY week=0 project=114 role=31 b_unit=0 f_unit=9 dept=11 team=62 itadmin=0\n"
    "PSY O=36 C=35 E=19 A=44 N=21\n"
    "SESSIONS total=2 kept=2\n"
    "SES idx=0 pc=0 duration=187.667 n_logon=1\n"
    "SES idx=1 pc=0 duration=205.467 n_logon=1"
)
NO_PSY_TEXT = "\n".join(l for l in FULL_TEXT.split("\n") if not l.startswith("PSY "))
NO_PROFILE_TEXT = "\n".join(
    l for l in FULL_TEXT.split("\n") if not (l.startswith("PSY ") or l.startswith("DAY "))
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


class MockTokenizer:
    """Whitespace-ish tokenizer returning HF-style offset mappings.

    `merge_newline=True` emits tokens that begin with "\\n" and continue into
    the next line, reproducing the boundary-crossing case real BPE produces.
    `n_special` prepends that many zero-width special tokens.
    """

    def __init__(self, merge_newline: bool = False, n_special: int = 0):
        self.merge_newline = merge_newline
        self.n_special = n_special

    def encode_offsets(self, text: str, max_len: int | None = None):
        offsets: list[tuple[int, int]] = [(0, 0)] * self.n_special
        i, n = 0, len(text)
        while i < n:
            if text[i] == "\n":
                if self.merge_newline and i + 1 < n:
                    j = i + 1
                    while j < n and text[j] not in " \n":
                        j += 1
                    offsets.append((i, j))
                    i = j
                else:
                    offsets.append((i, i + 1))
                    i += 1
                continue
            if text[i] == " ":
                i += 1
                continue
            j = i
            while j < n and text[j] not in " \n":
                j += 1
            # split long fields so tokens are sub-word sized
            k = i
            while k < j:
                e = min(k + 4, j)
                offsets.append((k, e))
                k = e
            i = j
        if max_len is not None:
            offsets = offsets[:max_len]
        return offsets


def test_naive_subtraction_is_wrong() -> None:
    print("1. naive mean subtraction vs the correct conditional score")
    n_p, n_b = 10, 90
    cases = [(10.0, 2.0), (1.0, 1.5)]
    naive, correct = [], []
    for s_p, s_b in cases:
        s_full = (n_p * s_p + n_b * s_b) / (n_p + n_b)
        naive.append(naive_difference(s_full, s_p))
        correct.append(behavior_score_from_means(s_full, s_p, n_p, n_b))
    check(
        "correct formula recovers s_B exactly",
        np.allclose(correct, [c[1] for c in cases]),
        f"{correct} vs {[c[1] for c in cases]}",
    )
    check(
        "naive difference reverses the behaviour ranking",
        (correct[0] > correct[1]) and (naive[0] < naive[1]),
        f"correct {correct[0]:.3f}>{correct[1]:.3f}; naive {naive[0]:.3f}<{naive[1]:.3f}",
    )
    identity = naive[0] - (n_b / (n_p + n_b)) * (cases[0][1] - cases[0][0])
    check("naive difference equals (N_B/N)(s_B - s_P)", abs(identity) < 1e-12)


def _decompose(text: str, tok: MockTokenizer, losses: np.ndarray, mask: np.ndarray | None = None):
    offsets = tok.encode_offsets(text)
    classes, crosses = classify_tokens(offsets, cert_class_spans(text))
    subflags = classify_subclass(offsets, cert_subclass_spans(text))
    if mask is None:
        mask = np.ones(len(offsets) - 1)
    return accumulate_class_losses(
        classes, losses, mask, CERT_CLASSES, crosses=crosses, subclass_flags=subflags
    ), offsets, classes


def test_reconstruction() -> None:
    print("2. class sums/counts reconstruct the original mean NLL")
    rng = np.random.default_rng(0)
    for name, text in (("full", FULL_TEXT), ("no_psy", NO_PSY_TEXT), ("no_profile", NO_PROFILE_TEXT)):
        tok = MockTokenizer()
        offsets = tok.encode_offsets(text)
        losses = rng.uniform(0.1, 5.0, size=len(offsets) - 1)
        row, _, _ = _decompose(text, tok, losses)
        ref_mean = float(losses.mean())
        got_mean = row["loss_sum_total"] / row["n_targets"]
        part_sum = sum(row[f"loss_sum_{c}"] for c in CERT_CLASSES)
        part_n = sum(row[f"n_{c}"] for c in CERT_CLASSES)
        check(
            f"{name}: partition sums to total",
            abs(part_sum - row["loss_sum_total"]) < 1e-9 and part_n == row["n_targets"],
        )
        check(f"{name}: mean NLL reconstructed", abs(got_mean - ref_mean) < 1e-12)
        check(
            f"{name}: n_targets == n_tokens - 1",
            row["n_targets"] == len(offsets) - 1,
            f"{row['n_targets']} vs {len(offsets) - 1}",
        )
        s_full = view_score(row, CERT_VIEWS["full"])
        s_prof = view_score(row, CERT_VIEWS["profile_only"])
        s_beh = view_score(row, CERT_VIEWS["behavior_only"])
        n_p = sum(row[f"n_{c}"] for c in CERT_VIEWS["profile_only"])
        n_b = sum(row[f"n_{c}"] for c in CERT_VIEWS["behavior_only"])
        if n_p and n_b:
            check(
                f"{name}: direct behaviour score == algebraic identity",
                abs(s_beh - behavior_score_from_means(s_full, s_prof, n_p, n_b)) < 1e-9,
            )
            check(
                f"{name}: weighted classes recompose s_full",
                abs((n_p * s_prof + n_b * s_beh) / (n_p + n_b) - s_full) < 1e-9,
            )
        else:
            check(f"{name}: no profile targets -> behaviour == full", abs(s_beh - s_full) < 1e-12)


def test_padding_and_shift() -> None:
    print("3. padding and causal-shift alignment")
    tok = MockTokenizer()
    rng = np.random.default_rng(1)
    short, long_ = NO_PROFILE_TEXT, FULL_TEXT
    off_s = tok.encode_offsets(short)
    off_l = tok.encode_offsets(long_)
    pad_to = len(off_l)
    losses_s = rng.uniform(0.1, 5.0, size=len(off_s) - 1)
    row_single, _, _ = _decompose(short, tok, losses_s)

    # right-padded batch row: classes padded, losses garbage past the real end,
    # mask = attention_mask[1:] so padded targets are excluded.
    cls_s, crosses_s = classify_tokens(off_s, cert_class_spans(short))
    sub_s = classify_subclass(off_s, cert_subclass_spans(short))
    cls_pad = cls_s + ["SPECIAL"] * (pad_to - len(cls_s))
    crosses_pad = crosses_s + [False] * (pad_to - len(crosses_s))
    sub_pad = sub_s + [False] * (pad_to - len(sub_s))
    losses_pad = np.concatenate([losses_s, rng.uniform(50, 99, size=pad_to - len(off_s))])
    mask_pad = np.concatenate([np.ones(len(off_s) - 1), np.zeros(pad_to - len(off_s))])
    row_batch = accumulate_class_losses(
        cls_pad, losses_pad, mask_pad, CERT_CLASSES, crosses=crosses_pad, subclass_flags=sub_pad
    )
    same = all(
        abs(row_batch[k] - row_single[k]) < 1e-9
        for k in row_single
        if k.startswith(("loss_sum_", "n_"))
    )
    check("padded batch row == unpadded single row", same)
    check(
        "garbage pad losses excluded",
        abs(row_batch["loss_sum_total"] - float(losses_s.sum())) < 1e-9,
    )

    # deliberate off-by-one: attributing loss to the context class instead of
    # the predicted token's class must change the answer on this text.
    cls_shifted = ["SPECIAL"] + cls_s[:-1]
    row_off = accumulate_class_losses(cls_shifted, losses_s, np.ones(len(off_s) - 1), CERT_CLASSES)
    check(
        "off-by-one attribution is detectably different",
        row_off[f"n_SES"] != row_single["n_SES"] or row_off["n_SESCOUNT"] != row_single["n_SESCOUNT"],
    )


def test_boundaries_and_truncation() -> None:
    print("4. boundary-crossing tokens and truncation")
    tok = MockTokenizer(merge_newline=True)
    rng = np.random.default_rng(2)
    offsets = tok.encode_offsets(FULL_TEXT)
    classes, crosses = classify_tokens(offsets, cert_class_spans(FULL_TEXT))
    check("boundary-crossing tokens are detected", any(crosses), f"{sum(crosses)} tokens")
    i = crosses.index(True)
    a, b = offsets[i]
    check(
        "crossing token assigned by its FIRST character",
        classes[i] == dict((s, c) for s, _, c in cert_class_spans(FULL_TEXT))[
            max(s for s, _, _ in cert_class_spans(FULL_TEXT) if s <= a)
        ],
        f"token {FULL_TEXT[a:b]!r} -> {classes[i]}",
    )
    losses = rng.uniform(0.1, 5.0, size=len(offsets) - 1)
    row = accumulate_class_losses(
        classes, losses, np.ones(len(offsets) - 1), CERT_CLASSES, crosses=crosses
    )
    n_bound = sum(row[f"n_boundary_{c}"] for c in CERT_CLASSES)
    check("boundary counts are recorded per class", n_bound > 0, f"{n_bound} scored boundary targets")

    # truncation: tokenizer cut mid-text must not produce unassigned classes and
    # must still reconstruct the mean over the retained targets.
    max_len = 12
    off_t = tok.encode_offsets(FULL_TEXT, max_len=max_len)
    cls_t, cr_t = classify_tokens(off_t, cert_class_spans(FULL_TEXT))
    losses_t = rng.uniform(0.1, 5.0, size=len(off_t) - 1)
    row_t = accumulate_class_losses(cls_t, losses_t, np.ones(len(off_t) - 1), CERT_CLASSES, crosses=cr_t)
    check("truncated: all classes known", all(c in CERT_CLASSES for c in cls_t))
    check(
        "truncated: reconstruction holds",
        abs(row_t["loss_sum_total"] / row_t["n_targets"] - float(losses_t.mean())) < 1e-12,
    )
    check("truncated: no SES targets in a 12-token prefix", row_t["n_SES"] == 0)

    # empty-span special tokens are not scored as content classes
    tok_sp = MockTokenizer(n_special=2)
    off_sp = tok_sp.encode_offsets(FULL_TEXT)
    cls_sp, _ = classify_tokens(off_sp, cert_class_spans(FULL_TEXT))
    check("special tokens classified SPECIAL", cls_sp[:2] == ["SPECIAL", "SPECIAL"])


def test_line_index_trap() -> None:
    print("5. prefix classification survives the no_psy / no_profile line shift")
    for name, text, expect_psy, expect_day in (
        ("full", FULL_TEXT, True, True),
        ("no_psy", NO_PSY_TEXT, False, True),
        ("no_profile", NO_PROFILE_TEXT, False, False),
    ):
        spans = cert_class_spans(text)
        got = [c for _, _, c in spans]
        check(f"{name}: PSY present == {expect_psy}", ("PSY" in got) == expect_psy)
        check(f"{name}: DAY present == {expect_day}", ("DAY" in got) == expect_day)
        check(f"{name}: no OTHER lines", "OTHER" not in got, f"classes={got}")
        # line index 1 is PSY in `full` but SESCOUNT in `no_psy`: the old
        # index-based rule would mislabel it.
        if name == "no_psy":
            check("no_psy: line index 1 is SESCOUNT, not PSY", got[1] == "SESCOUNT")
        if name == "no_profile":
            check("no_profile: line index 0 is SESCOUNT, not DAY", got[0] == "SESCOUNT")

    tok = MockTokenizer()
    offsets = tok.encode_offsets(FULL_TEXT)
    subflags = classify_subclass(offsets, cert_subclass_spans(FULL_TEXT))
    classes, _ = classify_tokens(offsets, cert_class_spans(FULL_TEXT))
    week_classes = {classes[i] for i, f in enumerate(subflags) if f}
    check("DAY_WEEK tokens are a subset of DAY", week_classes == {"DAY"}, str(week_classes))
    check("DAY_WEEK is non-empty on full text", any(subflags))


def test_lanl_spans() -> None:
    print("6. LANL field spans (schema ready, not yet validated on data)")
    text = (
        "t08 su=U620 du=U620 sc=C586 dc=C529 at=Kerberos lt=Network or=LogOn res=Success\n"
        "t09 su=U620 du=U1 sc=C586 dc=C625 at=NTLM lt=Network or=LogOff res=Success"
    )
    spans = lanl_class_spans(text)
    got = [c for _, _, c in spans]
    check("hour field -> HOUR", got[0] == "HOUR")
    check("su/du -> ID_USER", got[1] == "ID_USER" and got[2] == "ID_USER")
    check("sc/dc -> ID_HOST", got[3] == "ID_HOST" and got[4] == "ID_HOST")
    check("at/lt/or/res -> BEHAV", got[5:9] == ["BEHAV"] * 4)
    check("spans cover the text contiguously", spans[0][0] == 0 and spans[-1][1] == len(text))
    check("all classes known", all(c in LANL_CLASSES for c in got))


def test_training_loss_masking_semantics() -> None:
    """Pin the Phase-C objective arithmetic (no torch needed).

    Masking profile *targets* sets labels[t] = -100 at DAY/PSY tokens. HF's
    causal LM loss compares logits[..., :-1] with labels[..., 1:], so this
    removes the prediction OF those tokens while input_ids (the context) are
    untouched. The two denominators then differ by exactly 1/(1 - p).
    """
    print("7. training-loss masking semantics and denominator")
    tok = MockTokenizer()
    offsets = tok.encode_offsets(FULL_TEXT)
    classes, _ = classify_tokens(offsets, cert_class_spans(FULL_TEXT))
    rng = np.random.default_rng(7)
    losses = rng.uniform(0.1, 5.0, size=len(offsets) - 1)

    profile = {"DAY", "PSY"}
    # labels[t] = -100 for profile tokens; target t uses losses[t-1]
    keep = np.array([classes[t] not in profile for t in range(1, len(offsets))])
    n_all = len(losses)
    n_kept = int(keep.sum())
    p = 1.0 - n_kept / n_all

    baseline = float(losses.mean())
    masked_scored = float(losses[keep].sum() / n_kept)          # HF default
    masked_all = float(losses[keep].sum() / n_all)              # fixed denominator

    check("mask removes only profile targets", n_kept < n_all and n_kept > 0,
          f"p(masked)={p:.3f}, kept {n_kept}/{n_all}")
    check("all_targets denominator preserves per-token weight",
          abs(masked_all - (losses[keep].sum() / n_all)) < 1e-12)
    check("scored_targets denominator up-weights kept tokens by 1/(1-p)",
          abs(masked_scored - masked_all / (1.0 - p)) < 1e-9,
          f"{masked_scored:.4f} vs {masked_all / (1.0 - p):.4f}")
    check("unmasked: the two denominators agree with the baseline",
          abs(float(losses.sum() / n_all) - baseline) < 1e-12)

    # the mask must not touch the context: input ids are unchanged, only labels
    ids_before = list(range(len(offsets)))
    labels = [-100 if classes[i] in profile else ids_before[i] for i in range(len(offsets))]
    check("context tokens unchanged by masking",
          ids_before == list(range(len(offsets))))
    check("masked label positions are exactly the profile tokens",
          {i for i, l in enumerate(labels) if l == -100} ==
          {i for i, c in enumerate(classes) if c in profile})


def test_accumulation_window_normalization() -> None:
    """Pin the normalization contract that job 20810414's gate caught.

    transformers >= 4.46 counts targets over the whole gradient-accumulation
    window (`num_items_in_batch`) and then SKIPS its own
    `/ gradient_accumulation_steps`. A compute_loss that normalizes per
    micro-batch therefore returns a loss `grad_accum` times too large, which
    scales the gradients and silently changes the effective learning rate.
    Measured on the cluster: exactly 4.0x at grad_accum=4.
    """
    print("8. accumulation-window normalization")
    rng = np.random.default_rng(8)
    grad_accum = 4
    micro = [rng.uniform(0.1, 3.0, size=n) for n in (40, 55, 33, 61)]
    keep = [rng.random(len(m)) > 0.1485 for m in micro]   # ~p of targets masked

    n_window_all = sum(len(m) for m in micro)
    sum_behaviour = sum(float(m[k].sum()) for m, k in zip(micro, keep))

    # correct: every micro-batch divides by the WINDOW count, and the four
    # micro-batch losses sum to the window average
    per_micro_correct = [float((m * k).sum()) / n_window_all for m, k in zip(micro, keep)]
    window_loss = sum(per_micro_correct)
    check("window-normalized micro-batches sum to the window average",
          abs(window_loss - sum_behaviour / n_window_all) < 1e-12)

    # wrong: dividing by the micro-batch count, which is ~1/grad_accum the size
    per_micro_wrong = [float((m * k).sum()) / len(m) for m, k in zip(micro, keep)]
    inflation = (sum(per_micro_wrong) / grad_accum) / (window_loss / grad_accum)
    check("micro-batch normalization inflates the loss by ~grad_accum",
          abs(inflation - grad_accum) / grad_accum < 0.10,
          f"inflation = {inflation:.2f}x vs grad_accum = {grad_accum}")

    # unmasked, the fixed-denominator form must equal the plain mean
    per_micro_unmasked = [float(m.sum()) / n_window_all for m in micro]
    plain_mean = sum(float(m.sum()) for m in micro) / n_window_all
    check("with nothing masked, fixed denominator == plain mean",
          abs(sum(per_micro_unmasked) - plain_mean) < 1e-12)

    # and the two denominator choices differ by exactly 1/(1-p)
    n_kept = sum(int(k.sum()) for k in keep)
    p = 1.0 - n_kept / n_window_all
    scored = sum_behaviour / n_kept
    all_t = sum_behaviour / n_window_all
    check("scored_targets == all_targets / (1-p)",
          abs(scored - all_t / (1.0 - p)) < 1e-9, f"p = {p:.4f}")


def main() -> int:
    print("token_class_decomposition correctness checks\n")
    test_naive_subtraction_is_wrong()
    test_reconstruction()
    test_padding_and_shift()
    test_boundaries_and_truncation()
    test_line_index_trap()
    test_lanl_spans()
    test_training_loss_masking_semantics()
    test_accumulation_window_normalization()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
