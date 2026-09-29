#!/usr/bin/env python3
"""Exact reuse inventory of historical CERT r4.2 intervention tuples against the corrected pilot (Stage B.D).

A tuple = (adapter, tokenizer, dictionary + normalisation, feature set, edit rule and alpha, receiver, donor(s),
hook layer, score target, numerical configuration). The corrected claim's tuples use a refitted dictionary and
reselected features (protocol sections 3-5), so a historical tuple can be EXACTLY reusable only if its dictionary
file hash equals the corrected dictionary's hash. Every historical tuple is classified as
  exact           same dictionary hash, features, edit, receiver, donor and numerics as a proposed tuple
  diagnostic      label-valid under corrected labels (attack receiver positive; matched/benign receiver and every
                  donor non-insider or window-negative as declared) but made with the historical dictionary/features
  invalid         a receiver called malicious is now window-negative, a day called benign is now positive, or a
                  donor is an official insider / positive day
and the corrected pilot's tuples not covered by any exact historical tuple are 'missing, requires execution'.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', type=Path, required=True, help='example_id, y (corrected calendar labels)')
    ap.add_argument('--insiders', type=Path, required=True)
    ap.add_argument('--pkg4-candidates', type=Path, required=True)
    ap.add_argument('--pkg4-dictionary', type=Path, required=True)
    ap.add_argument('--pilot1-rows', type=Path, required=True)
    ap.add_argument('--pilot1-populations', type=Path, required=True)
    ap.add_argument('--eval-jsonl-ids', type=Path, required=True, help='example_id per eval.jsonl line (one per line)')
    ap.add_argument('--h6-receivers', type=Path, required=True)
    ap.add_argument('--corrected-dictionary-sha256', default='pending: refit on aquaman-l26-v1 (differs by construction)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    y = pd.read_parquet(a.labels).set_index('example_id')['y'].to_dict()
    ins = {l.strip() for l in a.insiders.read_text().splitlines() if l.strip()}
    user = lambda eid: eid.split(':')[0]
    res = {'corrected_dictionary_sha256': a.corrected_dictionary_sha256, 'families': {}}

    # Package 4: full candidate table (every searched donor x alpha x context x set)
    c = pd.read_csv(a.pkg4_candidates, usecols=['context_mode', 'feature_set', 'donor_type', 'receiver_example_id',
                                                 'donor_example_id', 'alpha'])
    rec_pos = c.receiver_example_id.map(y)
    don_pos = c.donor_example_id.map(y)
    don_ins = c.donor_example_id.map(user).isin(ins)
    benign_donor = c.donor_type == 'benign'
    invalid_rec = rec_pos != 1                                   # every package-4 receiver was called malicious
    invalid_don = benign_donor & (don_ins | (don_pos == 1))       # a "benign" donor from an insider or a positive day
    anom_don_now_neg = (~benign_donor) & (don_pos != 1)          # an "anomalous" donor that is now window-negative
    cls = pd.Series('diagnostic', index=c.index)
    cls[invalid_rec | invalid_don | anom_don_now_neg] = 'invalid'
    res['families']['package4_confirmation'] = {
        'dictionary': str(a.pkg4_dictionary), 'dictionary_sha256': sha(a.pkg4_dictionary),
        'features': 'top5 [4596, 7693, 2302, 3673, 3455]; control5_active [6596, 8017, 6608, 2765, 886] (historical labels)',
        'edit_rule': 'reconstruct every token, then set every set coordinate to the donor prototype at every position where '
                     'any set feature is active (union support); alphas 0.25/0.5/0.75/1; batch 12',
        'rows': int(len(c)), 'receivers': int(c.receiver_example_id.nunique()),
        'receivers_now_window_negative': int(c.loc[invalid_rec, 'receiver_example_id'].nunique()),
        'benign_donor_days_from_insiders_or_positive': int(c.loc[invalid_don, 'donor_example_id'].nunique()),
        'anomalous_donor_days_now_window_negative': int(c.loc[anom_don_now_neg, 'donor_example_id'].nunique()),
        'classification_rows': cls.value_counts().to_dict(), 'exact_reusable_rows': 0,
        'reason_not_exact': 'dictionary differs from the corrected refit (hash), features reselected, edit rule differs '
                            '(package-4 union edit after reconstruction vs residual-preserving pilot conditions)'}

    # Pilot 1 (exploratory, 2026-09-25)
    p = pd.read_csv(a.pilot1_rows, usecols=['condition', 'receiver_id', 'user', 'kind'])
    pops = json.loads(a.pilot1_populations.read_text())
    ids = a.eval_jsonl_ids.read_text().split()
    donors = [ids[i] for v in pops['donor_policy_by_dept'].values() for i in v]
    donor_bad = sorted({d for d in donors if user(d) in ins or y.get(d) == 1})
    bad_depts = [d for d, v in pops['donor_policy_by_dept'].items() if any(ids[i] in donor_bad for i in v)]
    p['y'] = p.receiver_id.map(y)
    inval = ((p.kind == 'malicious') & (p.y != 1)) | ((p.kind == 'benign') & (p.y == 1))
    res['families']['pilot1'] = {
        'rows': int(len(p)), 'receivers': int(p.receiver_id.nunique()),
        'malicious_receivers_now_window_negative': int(p.loc[(p.kind == 'malicious') & (p.y != 1), 'receiver_id'].nunique()),
        'benign_receivers_now_positive': int(p.loc[(p.kind == 'benign') & (p.y == 1), 'receiver_id'].nunique()),
        'donor_days_from_insiders_or_positive': donor_bad, 'departments_with_invalid_donor_prototype': bad_depts,
        'classification_rows': {'invalid': int(inval.sum()), 'diagnostic_if_department_prototype_valid': int((~inval).sum())},
        'note': 'every condition of a receiver in a department whose donor prototype includes an insider day is invalid '
                'as a benign-donor edit; the pairwise receiver checks above are necessary, not sufficient',
        'exact_reusable_rows': 0}

    # H6 (profile-patching study, 2026-09-27)
    h = json.loads(a.h6_receivers.read_text())
    recs = h if isinstance(h, list) else h.get('receivers', h)
    rows = []
    for r in recs:
        eid = r.get('example_id') or r.get('receiver_id') or r.get('id')
        kind = r.get('kind') or r.get('group') or r.get('label')
        rows.append({'example_id': eid, 'kind': kind, 'y': y.get(eid)})
    hr = pd.DataFrame(rows)
    res['families']['h6'] = {'receivers': int(len(hr)), 'kinds': hr.kind.value_counts().to_dict() if 'kind' in hr else None,
                             'crosstab_kind_by_corrected_y': pd.crosstab(hr.kind, hr.y).to_dict() if len(hr) else None,
                             'exact_reusable_rows': 0,
                             'scope': 'fixed-state diagnostics only (swap responses do not depend on labels); class '
                                      'splits must use corrected labels'}
    a.out.write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
