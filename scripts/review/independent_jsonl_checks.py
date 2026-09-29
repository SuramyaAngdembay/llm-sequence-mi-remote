#!/usr/bin/env python3
"""Independent check of the corrected evaluation JSONLs (review of commit c31efcf).

For each dataset: parse the staged original and the corrected copy line by line and require
  * the same number of rows in the same order (example_id sequence);
  * every original field other than y unchanged (parsed-value equality, text included);
  * y_original == original y; y == the reviewer's own window labels (from independent_label_checks);
  * no added fields other than y_original, calendar_date and label_version; calendar_date == origin + day_index;
  * the key set == the metadata's val + eval keys, with no duplicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ADDED = {'y_original', 'calendar_date', 'label_version'}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--label-check', type=Path, required=True, help='independent_label_checks.json (for archive identity only)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    R = a.recovery
    res = {}
    for tag in ['r42', 'r62']:
        meta = pd.read_parquet(R / f'labels/{tag}/example_metadata.parquet')
        # reviewer's labels: recomputed in independent_label_checks and found equal to meta.y on every row;
        # re-derive from the sparse table here to avoid trusting the overlay column itself
        lab = pd.read_parquet(R / f'labels/{tag}/incident_calendar_labels.parquet')
        pos = set(zip(lab.user_id, lab.calendar_date.astype(str)))
        unseen = meta.loc[meta.split.isin(['val', 'eval'])]
        want = list(unseen.example_id)
        orig_p, corr_p = R / f'jsonl/{tag}/original_eval.jsonl', R / f'jsonl/{tag}/eval_calendar_v1.jsonl'
        n = changed = 0; order_ids = []; bad = []
        origin = date(2009, 12, 28)
        with orig_p.open() as fo, corr_p.open() as fc:
            for lo, lc in zip(fo, fc):
                o, c = json.loads(lo), json.loads(lc)
                n += 1; order_ids.append(o['example_id'])
                extra = set(c) - set(o)
                if extra != ADDED or set(o) - set(c):
                    bad.append((o['example_id'], 'field set')); continue
                if any(c[k] != o[k] for k in o if k != 'y'):
                    bad.append((o['example_id'], 'nonlabel value')); continue
                cal = (origin + timedelta(days=int(o['day_index']))).isoformat()
                if c['calendar_date'] != cal or c['y_original'] != o['y']:
                    bad.append((o['example_id'], 'date or y_original')); continue
                if c['y'] != int((o['user_id'], cal) in pos):
                    bad.append((o['example_id'], 'y')); continue
                changed += c['y'] != o['y']
            tail_o, tail_c = fo.read(), fc.read()
        res[tag] = {'original_sha256': sha256(orig_p), 'corrected_sha256': sha256(corr_p), 'rows': n,
                    'extra_trailing_content': bool(tail_o.strip() or tail_c.strip()),
                    'violations': len(bad), 'first_violations': bad[:5], 'changed_labels': changed,
                    'unique_ids': len(set(order_ids)) == n,
                    'key_set_equals_metadata_val_eval': set(order_ids) == set(want),
                    'order_equals_metadata_val_eval_order': order_ids == want,
                    'positives': int(sum(int((u, (origin + timedelta(days=int(d))).isoformat()) in pos)
                                         for u, d in zip(unseen.user_id, unseen.day_index)))}
        print(tag, json.dumps(res[tag]), flush=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
