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

Deterministic: window sampling via md5(example_id); user folds via a salted
SHA-256 of the user id that is independent of the ETL's user-sampling hash.
Two passes (count, then emit). Validation gates printed at the end; counts must
be identical across conditions, and the user-level population checks in
lanl_hashing.check_population must pass or the split exits non-zero.

Before 2026-09-24 folds were md5(user) % 5, which, combined with the ETL's
md5(user) % 10 user sampling, put every sampled ordinary user in fold 0.
--legacy-md5-folds reproduces that split for the historical artifacts only.
"""
import argparse, json, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanl_hashing import (  # noqa: E402
    FOLD_SALT, SAMPLE_SALT, UNSEEN_FOLD, check_population, check_salts_independent,
    fold_of as _fold_of, population_report,
)

ap = argparse.ArgumentParser()
ap.add_argument('src', nargs='?', default='windows_full.jsonl')
ap.add_argument('outroot', nargs='?', default='lanl_conditions')
ap.add_argument('--fold-salt', default=FOLD_SALT)
ap.add_argument('--sample-salt', default=SAMPLE_SALT,
                help='the salt the ETL sampled users with; checked to differ from --fold-salt')
ap.add_argument('--legacy-md5-folds', action='store_true',
                help='reproduce the pre-2026-09-24 md5 folds (coupled with md5 sampling)')
ap.add_argument('--legacy-md5-sample', action='store_true',
                help='declare that the source was built with the legacy md5 user sampling')
ap.add_argument('--train-n', type=int, default=300_000)
ap.add_argument('--val-n', type=int, default=2_000)
ap.add_argument('--seen-n', type=int, default=30_000)
ap.add_argument('--unseen-n', type=int, default=30_000)
ap.add_argument('--allow-population-problems', action='store_true',
                help='write the split even if population checks fail (historical reproduction only)')
args = ap.parse_args()
if not args.allow_population_problems:
    check_salts_independent(args.sample_salt, args.fold_salt,
                            legacy_sample=args.legacy_md5_sample, legacy_fold=args.legacy_md5_folds)

SRC = args.src
OUTROOT = args.outroot
TRAIN_N, VAL_N, SEEN_N, UNSEEN_N = args.train_n, args.val_n, args.seen_n, args.unseen_n
COND = {'full': 'text', 'user_anon': 'text_user_anon',
        'host_anon': 'text_host_anon', 'shuffle': 'text_shuffle'}
_FOLD_CACHE = {}

def fold_of(user):
    if user not in _FOLD_CACHE:
        _FOLD_CACHE[user] = _fold_of(user, 5, salt=args.fold_salt, legacy=args.legacy_md5_folds)
    return _FOLD_CACHE[user]

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
        elif fold < UNSEEN_FOLD:
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

POP_RECORDS, TRAIN_USERS = [], set()

def emit(c, s, r, textkey, extra=None):
    if c == 'full':
        if s == 'train':
            TRAIN_USERS.add(r['user_id'])
        elif s == 'eval':
            POP_RECORDS.append({'user_id': r['user_id'], 'y': r['y'], 'seen': (extra or {}).get('seen')})
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
                emit(c, 'eval', r, tk, {'fold': fold, 'seen': int(fold < UNSEEN_FOLD)})
                counts[c]['eval_pos'] += 1
            continue
        if fold < UNSEEN_FOLD:
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

# ---- user-level population checks ----
report = population_report(POP_RECORDS, fold_by_user=_FOLD_CACHE)
unseen_users = {r['user_id'] for r in POP_RECORDS if r['seen'] == 0}
problems = check_population(report, unseen_users=unseen_users, train_users=TRAIN_USERS)
report.update({
    'fold_hash': 'legacy_md5' if args.legacy_md5_folds else f'sha256({args.fold_salt}|user)',
    'sample_hash_declared': 'legacy_md5' if args.legacy_md5_sample else f'sha256({args.sample_salt}|user)',
    'n_train_users': len(TRAIN_USERS),
    'problems': problems,
})
with open(os.path.join(OUTROOT, 'population_report.json'), 'w') as fh:
    json.dump(report, fh, indent=2)
print("POPULATION", json.dumps({k: report[k] for k in ('seen', 'unseen')}))
if problems:
    print("POPULATION CHECK FAILED:", *problems, sep="\n  ")
    if not args.allow_population_problems:
        sys.exit(2)
else:
    print("POPULATION CHECK passed")
print("out:", OUTROOT)
