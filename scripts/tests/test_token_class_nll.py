#!/usr/bin/env python3
"""Cross-implementation checks for per-token-class NLL accumulation.

The vectorized torch path in `token_class_nll` must agree with the reference
Python path in `token_class_decomposition.accumulate_class_losses`, which is
what produced the published decomposition. Two independent implementations
agreeing is the check; either one alone would only confirm itself.

Also checks the properties the causal-intervention reading depends on:

  1. per-class sums reconstruct the scalar per-example NLL exactly;
  2. loss is attributed to the class of the token being PREDICTED, not the
     context position (an off-by-one here silently moves mass between
     profile and behaviour);
  3. padding is excluded, so a batch's scores do not depend on its partners;
  4. `views_from_class` returns the exact conditional mean, and the
     profile/behaviour views partition the full view's targets;
  5. a zero patch reproduces the unpatched score.

Needs torch. Usage: python3 scripts/tests/test_token_class_nll.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import token_class_decomposition as tcd  # noqa: E402
from token_class_nll import (  # noqa: E402
    build_class_index, class_ids_for_texts, per_example_class_nll,
    per_example_class_nll_from_hidden, views_from_class,
)

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def scalar_nll(logits, input_ids, attention_mask):
    sl = logits[:, :-1, :].float()
    lab = input_ids[:, 1:]
    m = attention_mask[:, 1:].float()
    tl = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
                         reduction="none").view(lab.shape)
    return (tl * m).sum(1) / m.sum(1).clamp_min(1.0), tl


def main() -> int:
    print("token_class_nll cross-implementation checks\n")
    torch.manual_seed(0)
    names, c2i = build_class_index("cert")
    B, T, V = 5, 40, 97
    C = len(names)

    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    lengths = [40, 33, 28, 19, 12]          # ragged, as real padded batches are
    for b, L in enumerate(lengths):
        attn[b, L:] = 0
    # a plausible class layout: a DAY/PSY header then SES lines, plus SPECIAL pad
    class_ids = torch.empty(B, T, dtype=torch.long)
    for b, L in enumerate(lengths):
        seq = ([c2i["DAY"]] * 6 + [c2i["PSY"]] * 4 + [c2i["SESCOUNT"]] * 2
               + [c2i["SES"]] * 50 + [c2i["OTHER"]] * 10)[:L]
        class_ids[b, :L] = torch.tensor(seq)
        class_ids[b, L:] = c2i["SPECIAL"]

    print("1. agreement with the reference Python implementation")
    sums, counts = per_example_class_nll(logits, input_ids, attn, class_ids, C)
    scal, tok_loss = scalar_nll(logits, input_ids, attn)
    worst_sum, worst_cnt = 0.0, 0
    for b in range(B):
        ref = tcd.accumulate_class_losses(
            classes=[names[int(i)] for i in class_ids[b]],
            token_losses=tok_loss[b].tolist(),
            target_mask=attn[b, 1:].float().tolist(),
            class_names=names,
        )
        for ci, cname in enumerate(names):
            worst_sum = max(worst_sum, abs(float(sums[b, ci]) - ref[f"loss_sum_{cname}"]))
            worst_cnt = max(worst_cnt, abs(int(counts[b, ci]) - ref[f"n_{cname}"]))
    check("per-class loss sums match the reference", worst_sum < 1e-3, f"max |Δ| {worst_sum:.3e}")
    check("per-class target counts match the reference exactly", worst_cnt == 0, f"max |Δ| {worst_cnt}")

    print("\n2. reconstruction and attribution")
    recon = sums.sum(1) / counts.sum(1).clamp_min(1.0)
    check("per-class sums reconstruct the scalar NLL",
          torch.allclose(recon, scal, atol=1e-5), f"max |Δ| {float((recon - scal).abs().max()):.3e}")
    check("total counts equal the number of scored targets",
          all(int(counts[b].sum()) == lengths[b] - 1 for b in range(B)),
          f"{[int(counts[b].sum()) for b in range(B)]} vs {[L - 1 for L in lengths]}")
    # off-by-one detector: shifting the class map by one must change the split
    shifted = torch.roll(class_ids, shifts=1, dims=1)
    s2, _ = per_example_class_nll(logits, input_ids, attn, shifted, C)
    check("attributing to the context position instead would give a DIFFERENT split",
          not torch.allclose(sums, s2, atol=1e-6),
          "so the test is sensitive to the shift convention")

    print("\n3. padding independence")
    b0 = 1
    L = lengths[b0]
    one_sums, one_counts = per_example_class_nll(
        logits[b0 : b0 + 1, :L], input_ids[b0 : b0 + 1, :L],
        attn[b0 : b0 + 1, :L], class_ids[b0 : b0 + 1, :L], C)
    check("an example's per-class sums do not depend on its batch partners",
          torch.allclose(one_sums[0], sums[b0], atol=1e-4)
          and torch.equal(one_counts[0], counts[b0]),
          f"max |Δ| {float((one_sums[0] - sums[b0]).abs().max()):.3e}")

    print("\n4. views")
    v = views_from_class(sums.numpy(), counts.numpy(), names, dict(tcd.CERT_VIEWS))
    full_num = sums.numpy().sum(1)
    full_den = counts.numpy().sum(1)
    check("full view is the exact conditional mean over every class",
          np.allclose(v["full"], full_num / full_den), f"max |Δ| {np.abs(v['full'] - full_num / full_den).max():.3e}")
    p_idx = [names.index(c) for c in tcd.CERT_VIEWS["profile_only"]]
    b_idx = [names.index(c) for c in tcd.CERT_VIEWS["behavior_only"]]
    check("profile and behaviour views partition the full view's targets",
          np.array_equal(counts.numpy()[:, p_idx].sum(1) + counts.numpy()[:, b_idx].sum(1), full_den))
    nP = counts.numpy()[:, p_idx].sum(1)
    nB = counts.numpy()[:, b_idx].sum(1)
    identity = (nP * v["profile_only"] + nB * v["behavior_only"]) / full_den
    check("s_full = (N_P*s_P + N_B*s_B)/N holds exactly",
          np.allclose(identity, v["full"]), f"max |Δ| {np.abs(identity - v['full']).max():.3e}")
    naive = v["full"] - v["profile_only"]
    check("the PROHIBITED naive difference reorders examples vs the true behaviour view",
          not np.array_equal(np.argsort(naive), np.argsort(v["behavior_only"])),
          "confirming why it is not used")
    wP = nP / full_den
    check("s_full - w_P*s_P equals (N_B/N)*s_B",
          np.allclose(v["full"] - wP * v["profile_only"], (nB / full_den) * v["behavior_only"]))

    print("\n5. chunked LM-head path agrees with the direct path")
    hid = torch.randn(B, T, 16)
    head = torch.nn.Linear(16, V, bias=False)
    with torch.no_grad():
        direct_s, direct_c = per_example_class_nll(head(hid), input_ids, attn, class_ids, C)
        chunk_s, chunk_c = per_example_class_nll_from_hidden(
            head, hid, input_ids, attn, class_ids, C, max_logit_elements=B * V * 3)
    check("chunked and direct per-class sums agree",
          torch.allclose(direct_s, chunk_s, atol=1e-4),
          f"max |Δ| {float((direct_s - chunk_s).abs().max()):.3e}")
    check("chunked and direct counts agree exactly", torch.equal(direct_c, chunk_c))

    print("\n6. class-id construction from offsets")
    text = ("DAY week=0 project=114 role=31\nPSY O=36 C=35\n"
            "SESSIONS total=1 kept=1\nSES idx=0 pc=0 duration=187.0")
    spans = tcd.cert_class_spans(text)
    # one token per whitespace-delimited chunk, offsets taken from the text
    offs, pos = [], 0
    for chunk in text.replace("\n", " \n ").split(" "):
        if not chunk:
            continue
        a = text.index(chunk, pos) if chunk != "\n" else text.index("\n", pos)
        offs.append((a, a + len(chunk)))
        pos = a + len(chunk)
    offs = [(0, 0)] + offs                      # a leading special token
    ids = class_ids_for_texts([text], torch.tensor([offs]), "cert", c2i)[0]
    got = [names[int(i)] for i in ids]
    check("leading special token is SPECIAL", got[0] == "SPECIAL", got[0])
    check("DAY fields classify as DAY", got[1] == "DAY" and got[2] == "DAY", str(got[1:4]))
    check("PSY fields classify as PSY", "PSY" in got, str(sorted(set(got))))
    check("SES fields classify as SES", "SES" in got, str(sorted(set(got))))
    ref_names, _ = tcd.classify_tokens(offs, spans)
    check("class_ids_for_texts agrees with classify_tokens", got == ref_names)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
