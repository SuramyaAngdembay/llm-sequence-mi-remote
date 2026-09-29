#!/usr/bin/env python3
"""Stage B.B inputs: the r4.2 class-score bridge set and the missing-coverage set (protocol sections 1-2).

  * Checks that every record of the portability eval.jsonl (the text behind the 40,519-row batch-1 class cache)
    is byte-identical, as a parsed record, to the same example in the original r4.2 eval.jsonl.
  * bridge/eval.jsonl: the first 256 portability records (exactly the cached batch-1 pilot set, max_examples=256).
  * extension/eval.jsonl: every unseen (val+eval) example absent from the class cache, copied from the original
    eval.jsonl in its order, with the historical y (the recovery evaluator relabels afterwards).
Writes a manifest with hashes and counts; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original-eval', type=Path, required=True)
    ap.add_argument('--portability-eval', type=Path, required=True)
    ap.add_argument('--class-cache', type=Path, required=True)
    ap.add_argument('--bridge-cache', type=Path, required=True, help='cached batch-1 pilot class_scores.parquet')
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f'refusing to overwrite {a.out_dir}')
    orig = {}
    order = []
    for line in a.original_eval.open():
        r = json.loads(line); orig[r['example_id']] = (line, r); order.append(r['example_id'])
    port_ids, mismatched = [], []
    port_lines = []
    for line in a.portability_eval.open():
        r = json.loads(line); port_ids.append(r['example_id']); port_lines.append(line)
        if r != orig[r['example_id']][1]:
            mismatched.append(r['example_id'])
    if mismatched:
        raise SystemExit(f'{len(mismatched)} portability records differ from the original eval.jsonl, e.g. {mismatched[:3]}')
    cache = pd.read_parquet(a.class_cache)
    bridge_cache = pd.read_parquet(a.bridge_cache)
    if list(cache.example_id) != port_ids:
        raise SystemExit('class cache order differs from the portability eval.jsonl')
    if list(bridge_cache.sort_values('example_idx').example_id) != port_ids[:256]:
        raise SystemExit('cached batch-1 pilot is not the first 256 portability records')
    have = set(port_ids)
    missing = [e for e in order if e not in have]
    a.out_dir.mkdir(parents=True)
    (a.out_dir / 'bridge').mkdir(); (a.out_dir / 'extension').mkdir()
    (a.out_dir / 'bridge/eval.jsonl').write_text(''.join(port_lines[:256]))
    (a.out_dir / 'extension/eval.jsonl').write_text(''.join(orig[e][0] for e in missing))
    users = sorted({orig[e][1]['user_id'] for e in missing})
    man = {'original_eval_sha256': sha(a.original_eval), 'portability_eval_sha256': sha(a.portability_eval),
           'portability_records': len(port_ids), 'portability_records_identical_to_original': True,
           'class_cache_sha256': sha(a.class_cache), 'bridge_cache_sha256': sha(a.bridge_cache),
           'bridge_rows': 256, 'bridge_sha256': sha(a.out_dir / 'bridge/eval.jsonl'),
           'extension_rows': len(missing), 'extension_users': users,
           'extension_historical_positive_rows': sum(orig[e][1]['y'] for e in missing),
           'extension_sha256': sha(a.out_dir / 'extension/eval.jsonl')}
    (a.out_dir / 'inputs_manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
