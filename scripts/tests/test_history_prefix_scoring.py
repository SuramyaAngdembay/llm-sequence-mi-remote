#!/usr/bin/env python3
"""End-to-end check of the history-prefix scoring contract, without a model.

`run_history_prefix_probe.score_batch` passes the SCORED mask where
`per_example_class_nll` expects an attention mask, so that prefix tokens are
attended by the model but contribute nothing to the loss. That substitution is
the one place the probe could silently score the wrong tokens, so it is checked
here against explicitly enumerated target positions.

What must hold for a B-vs-C comparison to mean anything:
  1. A, B and C score the same NUMBER of targets;
  2. they score the same targets, i.e. the per-class count vectors are equal;
  3. no prefix token contributes loss, at any position;
  4. with identical logits over the current day, the three conditions give
     identical per-class sums -- so any difference later is caused by the
     prefix changing the model's predictions, not by the bookkeeping.

Needs torch. Usage: python3 scripts/tests/test_history_prefix_scoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import token_class_decomposition as tcd  # noqa: E402
from history_prefix import assemble, validate_conditions  # noqa: E402
from token_class_nll import build_class_index, per_example_class_nll, views_from_class  # noqa: E402

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def pack_batch(built, cls_map, n_classes, pad_id=0):
    width = max(len(i) for i, _ in built)
    B = len(built)
    ids = torch.full((B, width), pad_id, dtype=torch.long)
    cls = torch.zeros((B, width), dtype=torch.long)
    scored = torch.zeros((B, width), dtype=torch.long)
    for b, ((i_, t_), c_) in enumerate(zip(built, cls_map)):
        n = len(i_)
        ids[b, :n] = torch.tensor(i_)
        cls[b, :n] = torch.tensor(c_)
        scored[b, :n] = torch.tensor([1 if t else 0 for t in t_])
    return ids, cls, scored


def main() -> int:
    print("history-prefix scoring contract\n")
    torch.manual_seed(3)
    names, c2i = build_class_index("cert")
    C = len(names)
    V = 400                                            # must exceed every synthetic id below

    # a current day whose classes span profile and behaviour
    cur_cls_names = (["DAY"] * 8 + ["PSY"] * 5 + ["SESCOUNT"] * 3 + ["SES"] * 20)
    n_cur = len(cur_cls_names)
    cur_ids = list(range(5, 5 + n_cur))
    cur_cls = [c2i[n] for n in cur_cls_names]
    want = 11
    pre_self = list(range(200, 200 + want))
    pre_other = list(range(300, 300 + want))          # same length, different ids
    pre_cls = [c2i["SPECIAL"]] * want

    pack = {"A": assemble([], cur_ids),
            "B": assemble(pre_self, cur_ids),
            "C": assemble(pre_other, cur_ids)}
    rec = validate_conditions(pack, n_current=n_cur, max_seq_len=2048)
    print("1. construction")
    check("validator accepts the triple", rec["n_scored_targets"] == n_cur - 1,
          f"{rec['n_scored_targets']} targets")
    check("B and C have equal prefix lengths",
          rec["prefix_tokens"]["B"] == rec["prefix_tokens"]["C"] == want)

    cls_map = {"A": cur_cls, "B": pre_cls + cur_cls, "C": pre_cls + cur_cls}

    # Identical predictions over the current day in all three conditions: the
    # per-position logits for the current-day block are shared, so only the
    # bookkeeping can differ.
    shared_cur_logits = torch.randn(1, n_cur, V)
    print("\n2. same targets, same counts, prefix contributes nothing")
    per_cond = {}
    for cond in ("A", "B", "C"):
        ids_t, cls_t, scored_t = pack_batch([pack[cond]], [cls_map[cond]], C)
        L = ids_t.shape[1]
        logits = torch.randn(1, L, V) * 5.0            # arbitrary over the prefix
        logits[:, L - n_cur :, :] = shared_cur_logits  # identical over the current day
        s, n = per_example_class_nll(logits, ids_t, scored_t, cls_t, C)
        per_cond[cond] = (s.numpy(), n.numpy())

    nA = per_cond["A"][1]
    check("A, B and C score the same number of targets",
          all(int(per_cond[c][1].sum()) == n_cur - 1 for c in ("A", "B", "C")),
          f"{[int(per_cond[c][1].sum()) for c in ('A','B','C')]}")
    check("per-class target counts are identical across conditions",
          np.array_equal(per_cond["B"][1], nA) and np.array_equal(per_cond["C"][1], nA))
    check("no SPECIAL (prefix) target is ever counted",
          int(nA[0, names.index("SPECIAL")]) == 0, f"{int(nA[0, names.index('SPECIAL')])}")
    check("counts match the class layout of the current day minus its first token",
          int(nA[0, names.index("DAY")]) == 7 and int(nA[0, names.index("PSY")]) == 5
          and int(nA[0, names.index("SES")]) == 20,
          f"DAY {int(nA[0, names.index('DAY')])} PSY {int(nA[0, names.index('PSY')])} "
          f"SES {int(nA[0, names.index('SES')])}")

    print("\n3. identical current-day predictions give identical losses")
    check("per-class sums are identical across conditions",
          np.allclose(per_cond["B"][0], per_cond["A"][0], atol=1e-5)
          and np.allclose(per_cond["C"][0], per_cond["A"][0], atol=1e-5),
          f"max |Δ| {max(np.abs(per_cond['B'][0] - per_cond['A'][0]).max(), np.abs(per_cond['C'][0] - per_cond['A'][0]).max()):.3e}")
    vA = views_from_class(per_cond["A"][0], nA, names, dict(tcd.CERT_VIEWS))
    vB = views_from_class(per_cond["B"][0], per_cond["B"][1], names, dict(tcd.CERT_VIEWS))
    check("profile_only and behavior_only views also agree",
          np.allclose(vA["profile_only"], vB["profile_only"], atol=1e-5)
          and np.allclose(vA["behavior_only"], vB["behavior_only"], atol=1e-5))

    print("\n4. the test is sensitive: a wrong prefix mask would be caught")
    ids_t, cls_t, scored_t = pack_batch([pack["B"]], [cls_map["B"]], C)
    bad = scored_t.clone()
    bad[0, :want] = 1                                  # score the prefix too
    L = ids_t.shape[1]
    logits = torch.randn(1, L, V) * 5.0
    logits[:, L - n_cur :, :] = shared_cur_logits
    s_bad, n_bad = per_example_class_nll(logits, ids_t, bad, cls_t, C)
    check("scoring the prefix changes the counts, so the check would fail",
          not np.array_equal(n_bad.numpy(), nA),
          f"{int(n_bad.sum())} targets vs {int(nA.sum())}")

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
