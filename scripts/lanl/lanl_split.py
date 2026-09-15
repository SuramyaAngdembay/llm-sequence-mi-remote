#!/usr/bin/env python3
"""
Materialize per-condition train/val/eval jsonl dirs from windows_full.jsonl,
leave-users-out (LUO) protocol, fold 4 = unseen.

Per condition (full / user_anon / host_anon / shuffle):
  train.jsonl : 300k benign (y=0) windows from folds 0-3, split="train"
  val.jsonl   : 2k benign folds 0-3 (disjoint from train), split="val"
  eval.jsonl  : ALL positive windows (any fold)
                + ~30k benign folds 0-3 ("seen" pool)
                + ~30k benign fold 4    ("unseen" pool)
                each with fold + seen fields, split="eval"

Deterministic: sampling via md5(example_id). Two passes (count, then emit).
Validation gates printed at the end; counts must be identical across conditions.
"""
import json, hashlib, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else 'windows_full.jsonl'
OUTROOT = sys.argv[2] if len(sys.argv) > 2 else 'lanl_conditions'
TRAIN_N, VAL_N, SEEN_N, UNSEEN_N = 300_000, 2_000, 30_000, 30_000
COND = {'full': 'text', 'user_anon': 'text_user_anon',
        'host_anon': 'text_host_anon', 'shuffle': 'text_shuffle'}

def fold_of(user):
    return int(hashlib.md5(user.encode()).hexdigest(), 16) % 5

def h(example_id, salt):
    return int(hashlib.md5((salt + example_id).encode()).hexdigest(), 16)

# ---- pass 1: counts ----
n_benign_03, n_benign_4, n_pos = 0, 0, 0
with open(SRC) as f:
    for line in f:
        r = json.loads(line)
        fold = fold_of(r['user_id'])
        if r['y'] == 1:
            n_pos += 1
        elif fold < 4:
            n_benign_03 += 1
        else:
            n_benign_4 += 1
print(f"[pass1] benign_f03={n_benign_03} benign_f4={n_benign_4} pos={n_pos}")
M = 1_000_000
thr_train = int(M * min(1.0, TRAIN_N / max(1, n_benign_03)))
thr_val = int(M * min(1.0, VAL_N / max(1, n_benign_03)))
thr_seen = int(M * min(1.0, SEEN_N / max(1, n_benign_03)))
thr_unseen = int(M * min(1.0, UNSEEN_N / max(1, n_benign_4)))

# ---- pass 2: emit ----
files = {}
for c in COND:
    d = os.path.join(OUTROOT, c)
    os.makedirs(d, exist_ok=True)
    files[c] = {s: open(os.path.join(d, s + '.jsonl'), 'w')
                for s in ('train', 'val', 'eval')}
counts = {c: {'train': 0, 'val': 0, 'eval_pos': 0, 'eval_seen': 0, 'eval_unseen': 0}
          for c in COND}

def emit(c, s, r, textkey, extra=None):
    rec = {'example_id': r['example_id'], 'user_id': r['user_id'],
           'day_index': r['day_index'], 'split': s if s != 'eval' else 'eval',
           'y': r['y'], 'n_events': r['n_events'], 'text': r[textkey]}
    if extra:
        rec.update(extra)
    files[c][s if s in ('train', 'val') else 'eval'].write(json.dumps(rec) + '\n')

with open(SRC) as f:
    for line in f:
        r = json.loads(line)
        fold = fold_of(r['user_id'])
        eid = r['example_id']
        if r['y'] == 1:
            for c, tk in COND.items():
                emit(c, 'eval', r, tk, {'fold': fold, 'seen': int(fold < 4)})
                counts[c]['eval_pos'] += 1
            continue
        if fold < 4:
            b1, b2, b3 = h(eid, 'tr') % M, h(eid, 'va') % M, h(eid, 'se') % M
            if b1 < thr_train:
                for c, tk in COND.items():
                    emit(c, 'train', r, tk); counts[c]['train'] += 1
            elif b2 < thr_val:
                for c, tk in COND.items():
                    emit(c, 'val', r, tk); counts[c]['val'] += 1
            elif b3 < thr_seen:
                for c, tk in COND.items():
                    emit(c, 'eval', r, tk, {'fold': fold, 'seen': 1})
                    counts[c]['eval_seen'] += 1
        else:
            if h(eid, 'un') % M < thr_unseen:
                for c, tk in COND.items():
                    emit(c, 'eval', r, tk, {'fold': fold, 'seen': 0})
                    counts[c]['eval_unseen'] += 1

for c in COND:
    for fh in files[c].values():
        fh.close()

print("=== SPLIT REPORT ===")
for c in COND:
    print(c, counts[c])
same = len({tuple(sorted(v.items())) for v in counts.values()}) == 1
print("CHECK identical counts across conditions:", same)
print("out:", OUTROOT)
