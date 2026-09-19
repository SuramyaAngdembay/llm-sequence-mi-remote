#!/usr/bin/env python3
"""Work package 3: does the same frozen LM copy a profile from its context?

Three predeclared conditions on ONE frozen adapter, scoring identical targets:

  A  the current day's text alone (the published scoring condition)
  B  the SAME user's static profile, from a strictly earlier record, prepended
  C  a DIFFERENT user's static profile, prepended in the identical format and
     matched to B in token length

If the adapted model's low loss on profile tokens comes from having memorized
each user's profile during adaptation, a prefix should not help much and B and C
should differ little. If instead it comes from copying an identity block that is
present in context, B should reduce the profile loss sharply and C should not.

What this design fixes, and why each point matters, is documented in
`scripts/history_prefix.py`. The two that decide whether any number here means
anything: the current day's token ids are identical across conditions (the
prefix and the current text are tokenized separately and concatenated, never
re-tokenized as a joined string), and the scored target set is identical across
conditions (the current day's FIRST token is excluded as a target in ALL three,
so adding a prefix cannot add a target that A never had).

Declared BEFORE any loss is computed, and not changed afterwards:
  * B's prefix comes from the user's most recent record strictly earlier than
    the scored day.
  * C's donor pool is every other user in the same split. Donors are examined
    in a seeded hash order that depends only on the recipient id, and the first
    whose prefix tokenizes to exactly B's prefix length is taken. Donor
    profiles are read from each donor's EARLIEST record, so the choice does not
    depend on the scored day.
  * An example with no length-matched donor within 200 candidates is EXCLUDED
    and counted, never approximated.
  * CERT profiles may be static per user. This script therefore measures and
    reports how often the earlier profile actually differs from the current
    one, and the field overlap between B's and C's prefixes, rather than
    assuming either.

Usage:
  python3 scripts/run_history_prefix_probe.py --config ... --adapter-dir ... \
      --data-dir ... --out-dir ... [--max-examples 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import token_class_decomposition as tcd
from history_prefix import (
    assemble, build_prefix, choose_length_matched_donor, parse_profile,
    profile_overlap, validate_conditions,
)
from remote_common import dump_json, ensure_dir, load_yaml, read_jsonl
from token_class_nll import build_class_index, per_example_class_nll, views_from_class

CONDITIONS = ("A", "B", "C")


def load_model(cfg: Dict, adapter_dir: Path):
    """Load the frozen adapted model exactly as `score_adapter_examples.py` does.

    Same quantization block from the config, same tokenizer source (the adapter
    directory, not the base repo), same `use_cache=False`. A probe that loaded
    the model differently from the scorer that produced the published numbers
    would not be measuring the same model.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = cfg["model_name_or_path"]
    q = cfg["quantization"]
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(q["load_in_4bit"]),
        bnb_4bit_quant_type=str(q["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(torch, str(q["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(q["bnb_4bit_use_double_quant"]),
    )
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if bool(cfg["training"].get("bf16", True)) else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = False
    return model, tok


@torch.no_grad()
def score_batch(model, ids_list, tgt_list, cls_list, n_classes, pad_id, device):
    """Per-class summed loss and counts for a padded batch of built sequences."""
    width = max(len(i) for i in ids_list)
    B = len(ids_list)
    input_ids = torch.full((B, width), pad_id, dtype=torch.long)
    attn = torch.zeros((B, width), dtype=torch.long)
    cls = torch.zeros((B, width), dtype=torch.long)
    # `scored` marks the tokens we attribute loss to; it is NOT the attention
    # mask. Prefix tokens are attended but never scored.
    scored = torch.zeros((B, width), dtype=torch.float32)
    for b, (ids, tgt, c) in enumerate(zip(ids_list, tgt_list, cls_list)):
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids, dtype=torch.long)
        attn[b, :n] = 1
        cls[b, :n] = torch.tensor(c, dtype=torch.long)
        scored[b, :n] = torch.tensor([1.0 if t else 0.0 for t in tgt])
    input_ids, attn, cls = input_ids.to(device), attn.to(device), cls.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn,
                    return_dict=True, use_cache=False)
    # reuse the shared accumulator, but with the SCORED mask in place of the
    # attention mask so prefix tokens contribute nothing
    sums, counts = per_example_class_nll(
        out.logits, input_ids, scored.to(device).long(), cls, n_classes)
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sums.float().cpu().numpy(), counts.float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-file", default="eval.jsonl")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-donor-candidates", type=int, default=200)
    ap.add_argument("--prefix-include-week", action="store_true",
                    help="include the earlier record's own `week` value in the "
                         "prefix DAY line, restoring the training format. Run "
                         "both ways: the DAY view is sensitive to this, the PSY "
                         "view is not.")
    ap.add_argument("--max-per-user", type=int, default=0,
                    help="at most this many eligible days per user, taken evenly "
                         "spaced through the user's timeline (deterministic). 0 = all.")
    ap.add_argument("--user-limit", type=int, default=0,
                    help="sample this many users with a seeded shuffle. Users with "
                         "any positive day are ALWAYS kept, so the sample can never "
                         "be made easier by dropping malicious users.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    max_seq_len = int(cfg["training"]["max_seq_len"])
    out_dir = ensure_dir(args.out_dir)

    rows = list(read_jsonl(args.data_dir / args.split_file))
    by_user: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_user[str(r["user_id"])].append(r)
    for u in by_user:
        by_user[u].sort(key=lambda r: int(r["day_index"]))
    n_users_total = len(by_user)
    print(f"loaded {len(rows)} examples over {n_users_total} users", flush=True)

    # Sampling is declared here, before any scoring, and is label-aware only in
    # the direction that cannot flatter the result: every user with a positive
    # day is retained.
    pos_users = {u for u, recs in by_user.items() if any(int(r["y"]) == 1 for r in recs)}
    if args.user_limit and args.user_limit < n_users_total:
        rng = np.random.default_rng(args.seed)
        others = sorted(set(by_user) - pos_users)
        rng.shuffle(others)
        keep = pos_users | set(others[: max(0, args.user_limit - len(pos_users))])
        by_user = {u: recs for u, recs in by_user.items() if u in keep}
        print(f"user sample: {len(by_user)} users "
              f"({len(pos_users)} with a positive day, all retained)", flush=True)
    if args.max_per_user:
        for u, recs in list(by_user.items()):
            # positions 1.. are eligible (position 0 has no earlier record);
            # take an evenly spaced, deterministic subset of them
            elig = list(range(1, len(recs)))
            if len(elig) > args.max_per_user:
                pick = np.linspace(0, len(elig) - 1, args.max_per_user).round().astype(int)
                elig = [elig[i] for i in sorted(set(pick.tolist()))]
            by_user[u] = [recs[0]] + [recs[i] for i in elig]
        print(f"per-user cap: {args.max_per_user} eligible days", flush=True)

    model, tok = load_model(cfg, args.adapter_dir)
    device = next(model.parameters()).device
    class_names, class_to_idx = build_class_index("cert")
    n_classes = len(class_names)
    views = dict(tcd.CERT_VIEWS)

    inc_week = bool(args.prefix_include_week)

    def tok_len(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    # Donor profiles: each donor's EARLIEST record. Fixed before any scoring.
    donor_profile = {u: parse_profile(recs[0]["text"]) for u, recs in by_user.items()}
    pool = sorted(donor_profile)

    def cls_for(text: str, ids_len: int) -> List[int]:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        names, _ = tcd.classify_tokens(enc["offset_mapping"], tcd.cert_class_spans(text))
        idx = [class_to_idx.get(n, class_to_idx["OTHER"]) for n in names]
        return idx[:ids_len] + [class_to_idx["SPECIAL"]] * max(0, ids_len - len(idx))

    built: List[Dict] = []
    n_no_earlier = n_no_donor = n_truncated = 0
    for u, recs in by_user.items():
        for pos, rec in enumerate(recs):
            if args.max_examples and len(built) >= args.max_examples:
                break
            if pos == 0:
                n_no_earlier += 1
                continue
            prev = recs[pos - 1]
            prof_self = parse_profile(prev["text"])
            try:
                pre_self = build_prefix(prof_self, include_week=inc_week)
            except ValueError:
                n_no_earlier += 1
                continue
            pre_ids_self = tok(pre_self, add_special_tokens=False)["input_ids"]
            want = len(pre_ids_self)
            donor, _ = choose_length_matched_donor(
                u, want, pool, donor_profile,
                lambda t: tok_len(t), seed=args.seed,
                max_candidates=args.max_donor_candidates, include_week=inc_week)
            if donor is None:
                n_no_donor += 1
                continue
            pre_ids_other = tok(build_prefix(donor_profile[donor], include_week=inc_week),
                                add_special_tokens=False)["input_ids"]
            assert len(pre_ids_other) == want

            cur_full = tok(rec["text"], truncation=True, max_length=max_seq_len)["input_ids"]
            budget = max_seq_len - want
            cur = cur_full[:budget]
            if len(cur) < len(cur_full):
                n_truncated += 1
            if len(cur) < 2:
                continue
            cur_cls = cls_for(rec["text"], len(cur))
            pre_cls = [class_to_idx["SPECIAL"]] * want
            pack = {
                "A": assemble([], cur),
                "B": assemble(pre_ids_self, cur),
                "C": assemble(pre_ids_other, cur),
            }
            validate_conditions(pack, n_current=len(cur), max_seq_len=max_seq_len)
            built.append({
                "example_id": rec["example_id"], "user_id": u,
                "day_index": int(rec["day_index"]), "y": int(rec["y"]),
                "donor_user": donor, "prefix_tokens": want,
                "n_current_tokens": len(cur),
                "self_profile_changed": int(prof_self != parse_profile(rec["text"])),
                "overlap_self_vs_current": profile_overlap(prof_self, parse_profile(rec["text"])),
                "overlap_donor_vs_current": profile_overlap(donor_profile[donor],
                                                            parse_profile(rec["text"])),
                "pack": pack,
                "cls": {"A": cur_cls, "B": pre_cls + cur_cls, "C": pre_cls + cur_cls},
            })
        if args.max_examples and len(built) >= args.max_examples:
            break

    print(f"built {len(built)} examples; excluded {n_no_earlier} with no earlier record, "
          f"{n_no_donor} with no length-matched donor; {n_truncated} truncated by the "
          f"prefix budget", flush=True)
    if not built:
        raise SystemExit("no eligible examples")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    out_rows: List[Dict] = []
    for start in range(0, len(built), args.batch_size):
        chunk = built[start : start + args.batch_size]
        rec_out = [{k: v for k, v in c.items() if k not in ("pack", "cls")} for c in chunk]
        counts_ref = None
        for cond in CONDITIONS:
            ids_list = [c["pack"][cond][0] for c in chunk]
            tgt_list = [c["pack"][cond][1] for c in chunk]
            cls_list = [c["cls"][cond] for c in chunk]
            sums, counts = score_batch(model, ids_list, tgt_list, cls_list,
                                       n_classes, pad_id, device)
            if counts_ref is None:
                counts_ref = counts
            elif not np.array_equal(counts, counts_ref):
                raise RuntimeError(
                    f"condition {cond} scores a different set of targets than A; "
                    "the conditions are not comparable")
            v = views_from_class(sums, counts, class_names, views)
            for i, r in enumerate(rec_out):
                for vname in views:
                    r[f"{cond}_{vname}"] = float(v[vname][i])
                r[f"{cond}_n_targets"] = int(counts[i].sum())
        out_rows.extend(rec_out)
        if (start // args.batch_size) % 25 == 0:
            print(f"  scored {start + len(chunk)}/{len(built)}", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(out_dir / "history_prefix_per_example.csv", index=False)

    summary: List[Dict] = []
    for vname in views:
        a, b, c = df[f"A_{vname}"], df[f"B_{vname}"], df[f"C_{vname}"]
        summary.append({
            "view": vname, "n": len(df),
            "A_mean": float(a.mean()), "B_mean": float(b.mean()), "C_mean": float(c.mean()),
            "B_minus_A": float((b - a).mean()), "C_minus_A": float((c - a).mean()),
            "B_minus_C": float((b - c).mean()),
            "B_minus_C_sd": float((b - c).std(ddof=1)),
            "frac_B_below_C": float((b < c).mean()),
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(out_dir / "history_prefix_summary.csv", index=False)

    dump_json(out_dir / "history_prefix_meta.json", {
        "adapter_dir": str(args.adapter_dir), "split_file": args.split_file,
        "max_seq_len": max_seq_len, "seed": args.seed,
        "prefix_include_week": inc_week,
        "n_users_total": n_users_total, "n_users_sampled": len(by_user),
        "n_users_with_a_positive_day": len(pos_users),
        "user_limit": args.user_limit, "max_per_user": args.max_per_user,
        "n_built": len(built), "n_excluded_no_earlier_record": n_no_earlier,
        "n_excluded_no_length_matched_donor": n_no_donor,
        "n_truncated_by_prefix_budget": n_truncated,
        "frac_self_profile_changed": float(df["self_profile_changed"].mean()),
        "mean_overlap_self_vs_current": float(df["overlap_self_vs_current"].mean()),
        "mean_overlap_donor_vs_current": float(df["overlap_donor_vs_current"].mean()),
        "donor_rule": "earliest record of each donor user; seeded hash order; "
                      "first exact prefix-token-length match within "
                      f"{args.max_donor_candidates} candidates",
        "target_rule": "current day's tokens at within-day positions 1..n-1, "
                       "identical in all three conditions",
    })
    print("\n=== history-prefix probe ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
