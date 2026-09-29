#!/usr/bin/env python3
"""Corrected pilot populations (docs/review/STAGE_B_PROTOCOL_2026-09-28.md section 6). Deterministic; no scores used.

Indices are line numbers of the r4.2 eval.jsonl (= example_idx of the token-delta score index, checked).
  receivers   up to --per-user corrected-positive days per confirmation user, sha256(SALT|recv|example_id) order
  matched     for each receiver, the same user's nearest window-negative day (corrected y = 0), earlier on a tie,
              each used once; these are NOT called benign activity
  donors      per department, up to --donors held-out benign (val split, non-insider) users, sha256 order,
              one day each in sha256 order; official insiders are never donors
Every confirmation user is listed with the number of receivers kept and any exclusion reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SALT = 'pilot-calendar-v1-2026-09-28'


def h(*parts):
    return hashlib.sha256('|'.join((SALT,) + parts).encode()).hexdigest()


def dept(text):
    first = text.split('\n', 1)[0]
    assert first.startswith('DAY ')
    return dict(kv.split('=', 1) for kv in first[4:].split())['dept']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-jsonl', type=Path, required=True)
    ap.add_argument('--score-index', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True, help='corrected metadata (example_id, y, split)')
    ap.add_argument('--confirmation-users', type=Path, required=True)
    ap.add_argument('--insiders', type=Path, required=True)
    ap.add_argument('--per-user', type=int, default=6)
    ap.add_argument('--donors', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    meta = pd.read_parquet(a.metadata).set_index('example_id')
    idx = pd.read_parquet(a.score_index).sort_values('example_idx')
    rows = []
    for i, line in enumerate(a.eval_jsonl.open()):
        r = json.loads(line)
        rows.append({'idx': i, 'id': r['example_id'], 'user': r['user_id'], 'day': int(r['day_index']),
                     'split': r['split'], 'dept': dept(r['text']), 'y': int(meta.loc[r['example_id'], 'y'])})
    if [r['id'] for r in rows] != idx.example_id.tolist() or idx.example_idx.tolist() != list(range(len(rows))):
        raise SystemExit('eval.jsonl line order differs from the token-delta score index')
    conf = sorted(l.strip() for l in a.confirmation_users.read_text().splitlines() if l.strip())
    insiders = {l.strip() for l in a.insiders.read_text().splitlines() if l.strip()}
    by_user = defaultdict(list)
    for r in rows:
        by_user[r['user']].append(r)
    donor_users = sorted(u for u, rs in by_user.items() if rs[0]['split'] == 'val')
    assert not set(donor_users) & insiders and all(not any(r['y'] for r in by_user[u]) for u in donor_users)

    receivers, matched, users_report = [], [], []
    for u in conf:
        pos = sorted((r for r in by_user[u] if r['y'] == 1), key=lambda r: h('recv', r['id']))[: a.per_user]
        neg = [r for r in by_user[u] if r['y'] == 0]
        used, kept = set(), 0
        reason = None
        if not pos:
            reason = 'no corrected-positive day'
        for p in pos:
            cands = sorted((r for r in neg if r['idx'] not in used),
                           key=lambda r: (abs(r['day'] - p['day']), r['day'] > p['day'], r['day']))
            if not cands:
                reason = reason or 'window-negative days exhausted'
                continue
            used.add(cands[0]['idx']); receivers.append(p['idx']); matched.append(cands[0]['idx']); kept += 1
        users_report.append({'user': u, 'positive_days': sum(r['y'] for r in by_user[u]),
                             'window_negative_days': len(neg), 'receivers_kept': kept, 'note': reason})

    depts = sorted({rows[i]['dept'] for i in receivers + matched})
    donor_policy, no_donor = {}, []
    for d in depts:
        us = sorted((v for v in donor_users if by_user[v][0]['dept'] == d), key=lambda v: h('donor-user', d, v))
        donor_policy[d] = [sorted(by_user[v], key=lambda r: h('donor-day', r['id']))[0]['idx'] for v in us[: a.donors]]
        if not donor_policy[d]:
            no_donor.append(d)
    # receivers whose department has no donor are excluded (recorded), with their matched day
    pairs = [(p, m) for p, m in zip(receivers, matched) if rows[p]['dept'] not in no_donor and rows[m]['dept'] not in no_donor]
    excluded_pairs = [(rows[p]['id'], rows[m]['id']) for p, m in zip(receivers, matched) if (p, m) not in pairs]
    for rep in users_report:
        rep['receivers_after_donor_rule'] = sum(1 for p, _ in pairs if rows[p]['user'] == rep['user'])
    donors_flat = [i for v in donor_policy.values() for i in v]
    assert all(rows[i]['user'] not in insiders and rows[i]['y'] == 0 for i in donors_flat)
    assert all(rows[p]['y'] == 1 and rows[m]['y'] == 0 and rows[p]['user'] == rows[m]['user'] for p, m in pairs)
    out = {'salt': SALT, 'protocol': 'docs/review/STAGE_B_PROTOCOL_2026-09-28.md section 6',
           'eval_jsonl_sha256': hashlib.sha256(a.eval_jsonl.read_bytes()).hexdigest(),
           'metadata_sha256': hashlib.sha256(a.metadata.read_bytes()).hexdigest(),
           'confirmation_users': conf, 'matched_day_semantics': 'window-negative day of the same insider (not benign)',
           'malicious_receivers': [p for p, _ in pairs], 'matched_benign_receivers': [m for _, m in pairs],
           'receiver_pairs': [[p, m] for p, m in pairs], 'donor_policy_by_dept': donor_policy,
           'departments_without_donor': no_donor, 'excluded_pairs_no_donor': excluded_pairs,
           'users': users_report,
           'counts': {'pairs': len(pairs), 'users_with_pairs': len({rows[p]['user'] for p, _ in pairs}),
                      'departments': len(depts), 'donor_days': len(donors_flat)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ['counts', 'departments_without_donor']}, indent=1))
    print(pd.DataFrame(users_report).to_string(index=False))


if __name__ == '__main__':
    main()
