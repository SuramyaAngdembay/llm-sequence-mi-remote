#!/usr/bin/env python3
"""
Stream sessions_full.jsonl once and emit a small manifest CSV that makes every
later stage cheap (no re-streaming of the 100GB file):

  user, day, n_events, label, offset, length, luo_fold

- offset/length : byte position of the line in sessions_full.jsonl, so any
                  subset can be materialized with seek() instead of a full scan.
- luo_fold      : md5(user) % 5 -> leave-users-out folds. Deterministic, so the
                  same user is always in the same fold.

Split semantics (applied downstream, documented here):
  temporal : train = label==0 sessions with day <= TEMPORAL_TRAIN_LAST_DAY (11,
             Tuor-style days 1-12 dev window); test = day >= 12. Field-standard.
  luo      : train = benign sessions of users in folds != k ; test = ALL
             sessions of fold-k users (seen-vs-unseen contrast per fold).

Also prints a validation summary (session/positive counts must match the ETL
build report: 444900 sessions, 176 positives).
"""
import json, hashlib, sys, csv

SRC = sys.argv[1] if len(sys.argv) > 1 else 'sessions_full.jsonl'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'manifest.csv'

n = 0
n_pos = 0
pos_days = {}
fold_users = [set() for _ in range(5)]
fold_pos = [0] * 5

with open(SRC, 'rb') as f, open(OUT, 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['user', 'day', 'n_events', 'label', 'offset', 'length', 'luo_fold'])
    offset = 0
    for raw in f:
        length = len(raw)
        try:
            r = json.loads(raw)
        except json.JSONDecodeError:
            sys.stderr.write(f"BAD JSON at offset {offset}\n")
            offset += length
            continue
        fold = int(hashlib.md5(r['user'].encode()).hexdigest(), 16) % 5
        w.writerow([r['user'], r['day'], r['n_events'], r['label'], offset, length, fold])
        fold_users[fold].add(r['user'])
        if r['label'] == 1:
            n_pos += 1
            fold_pos[fold] += 1
            pos_days[r['day']] = pos_days.get(r['day'], 0) + 1
        n += 1
        offset += length

print("=== MANIFEST REPORT ===")
print("sessions:", n)
print("positives:", n_pos)
print("positive_days:", dict(sorted(pos_days.items())))
print("fold_sizes(users):", [len(s) for s in fold_users])
print("fold_positives:", fold_pos)
print("out:", OUT)
