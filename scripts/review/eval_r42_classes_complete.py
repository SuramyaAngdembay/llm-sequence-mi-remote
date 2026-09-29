#!/usr/bin/env python3
"""C32 completion (protocol section 2): combine the 40,519-row batch-1 class cache with the 1,949-row extension,
only if the bridge was bit-exact, and evaluate behavior-only vs full on all 70 corrected positive users.

Uses the reviewer's evaluator (independent_metrics.py). Writes a new complete cache; the inputs are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_metrics import attach_labels, class_views, fold_table, roc  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-cache', type=Path, required=True)
    ap.add_argument('--extension-cache', type=Path, required=True)
    ap.add_argument('--bridge-gate', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True, help='incident_calendar_labels.parquet')
    ap.add_argument('--answers', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f'refusing to overwrite {a.out_dir}')
    gate = json.loads(a.bridge_gate.read_text())
    if not gate.get('bit_exact'):
        raise SystemExit('bridge gate not bit-exact: the extension may not be combined with the cached rows')
    old, ext = pd.read_parquet(a.old_cache), pd.read_parquet(a.extension_cache)
    meta = pd.read_parquet(a.metadata)
    unseen = meta[meta.split.isin(['val', 'eval'])]
    if set(old.example_id) & set(ext.example_id):
        raise SystemExit('extension overlaps the cached rows')
    if set(ext.example_id) != set(unseen.example_id) - set(old.example_id):
        raise SystemExit('extension is not exactly the missing unseen keys')
    both = pd.concat([old, ext.assign(example_idx=ext.example_idx + 10 ** 6)], ignore_index=True)
    both = both.set_index('example_id').loc[unseen.example_id].reset_index()
    a.out_dir.mkdir(parents=True)
    both.to_parquet(a.out_dir / 'class_scores_complete_calendar_v1.parquet', index=False)
    lab = pd.read_parquet(a.labels)
    d = attach_labels(both, lab)
    views, checks = class_views(d)
    views['profile_only'] = ((d.loss_sum_DAY + d.loss_sum_PSY) / (d.n_DAY + d.n_PSY)).to_numpy()
    with tarfile.open(a.answers, 'r:bz2') as t:
        m = [x for x in t.getmembers() if x.name.endswith('insiders.csv')][0]
        scen = {r['user']: r['scenario'] for r in csv.DictReader(io.StringIO(t.extractfile(m).read().decode('utf-8-sig')))
                if r['dataset'] == '4.2'}
    ft = fold_table(d, views, 'y_corr')
    ft['scenario'] = ft.heldout.map(scen)
    ft['in_old_cache'] = ft.heldout.isin(set(old.user_id))
    res = {'rows': int(len(d)), 'positive_days': int(d.y_corr.sum()), 'positive_users': int(d.user_id[d.y_corr == 1].nunique()),
           'checks': checks, 'bridge_gate': gate, 'n_folds': int(ft.heldout.nunique()),
           'fold_mean': {v: {m: float(g[m].mean()) for m in ['day_roc', 'day_ap', 'prevalence', 'user_roc']}
                         for v, g in ft.groupby('view')},
           'pooled_day_roc': {v: roc(d.y_corr, s) for v, s in views.items()},
           'pooled_prevalence': float(d.y_corr.mean())}
    users = sorted(ft.heldout.unique())
    idx = np.random.default_rng(42).integers(0, len(users), (10000, len(users)))
    for metric in ['day_roc', 'user_roc']:
        piv = ft.pivot(index='heldout', columns='view', values=metric).loc[users]
        dlt = piv.behavior_only - piv.full
        b = dlt.to_numpy()[idx].mean(axis=1)
        res[f'behavior_minus_full_{metric}'] = {'estimate': float(dlt.mean()),
                                                'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    res['by_scenario'] = {f'{s} | {v}': {'n_users': int(len(g)), 'day_roc': float(g.day_roc.mean()), 'user_roc': float(g.user_roc.mean())}
                          for (s, v), g in ft.groupby(['scenario', 'view'])}
    res['restricted_60_vs_complete_70'] = {v: {'old_cache_users_day_roc': float(g[g.in_old_cache].day_roc.mean()),
                                               'new_users_day_roc': float(g[~g.in_old_cache].day_roc.mean())}
                                           for v, g in ft.groupby('view')}
    res['output_sha256'] = sha(a.out_dir / 'class_scores_complete_calendar_v1.parquet')
    ft.to_csv(a.out_dir / 'fold_metrics_complete_70.csv', index=False)
    (a.out_dir / 'c32_complete_results.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('by_scenario', 'checks', 'bridge_gate')}, indent=1))


if __name__ == '__main__':
    main()
