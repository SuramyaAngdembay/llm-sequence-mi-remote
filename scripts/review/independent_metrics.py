#!/usr/bin/env python3
"""Independent reproduction of a predetermined selection of recovery metrics (see
metric_reproduction_selection.md in the review directory). Imports no recovery or metric-core code."""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CLASSES = ['DAY', 'PSY', 'SESCOUNT', 'SES', 'OTHER', 'SPECIAL']
BEHAVIOR = ['SESCOUNT', 'SES', 'OTHER', 'SPECIAL']


def roc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)                                   # average ranks: ties count 1/2
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(y, s):
    """Average precision with tied scores grouped into one threshold."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind='mergesort'); ys, ss = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]      # last index of each tied block
    tp = np.cumsum(ys)[last]; k = last + 1
    prec, rec = tp / k, tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def folds(uid, y, seed=42, cap=800):
    d = pd.DataFrame({'u': uid, 'y': y})
    pos = sorted(d.u[d.y == 1].unique())
    allu = sorted(d.u.unique()); ben = np.asarray(sorted(set(allu) - set(pos)))
    out = []
    for i, p in enumerate(pos):
        rng = np.random.default_rng(seed + 1000 + i)
        out.append((p, set(rng.choice(ben, size=min(cap, len(ben)), replace=False).tolist()) | {p}))
    return out


def fold_table(df, scores: dict, ycol):
    uid = df.user_id.to_numpy(); y = df[ycol].to_numpy(int)
    umax = {k: pd.Series(v).groupby(uid).max() for k, v in scores.items()}
    uy = pd.Series(y).groupby(uid).max()
    rows = []
    for held, test in folds(uid, y):
        m = np.isin(uid, list(test)); users = sorted(test)
        for k, s in scores.items():
            rows.append(dict(view=k, heldout=held, n_rows=int(m.sum()), prevalence=float(y[m].mean()),
                             day_roc=roc(y[m], s[m]), day_ap=ap(y[m], s[m]),
                             user_roc=roc(uy.loc[users].to_numpy(), umax[k].loc[users].to_numpy())))
    return pd.DataFrame(rows)


def strict_dedup(df):
    cols = [c for c in df.columns if c != 'example_idx']
    first = df.drop_duplicates('example_id', keep='first')
    dup = df[df.duplicated('example_id', keep='first')]
    ref = first.set_index('example_id').loc[dup.example_id, [c for c in cols if c != 'example_id']]
    same = (ref.reset_index(drop=True) == dup[[c for c in cols if c != 'example_id']].reset_index(drop=True)).all().all()
    assert same, 'duplicate rows disagree'
    return first.reset_index(drop=True), len(dup)


def attach_labels(df, lab):
    cal = (pd.Timestamp('2009-12-28') + pd.to_timedelta(df.day_index, unit='D')).dt.strftime('%Y-%m-%d')
    pos = set(zip(lab.user_id, lab.calendar_date.astype(str)))
    df = df.copy(); df['y_corr'] = [int(k in pos) for k in zip(df.user_id, cal)]
    return df


def class_views(df):
    for c in CLASSES:
        assert (df[f'n_{c}'] >= 0).all()
    n = sum(df[f'n_{c}'] for c in CLASSES); l = sum(df[f'loss_sum_{c}'] for c in CLASSES)
    checks = {'partition_count_mismatch': int((n != df.n_targets).sum()),
              'targets_eq_tokens_minus_1_mismatch': int((df.n_targets != df.n_tokens - 1).sum()),
              'partition_loss_max_err': float((l - df.loss_sum_total).abs().max()),
              'full_vs_adapted_nll_max_err': float((l / n - df.adapted_nll).abs().max())}
    nb = sum(df[f'n_{c}'] for c in BEHAVIOR)
    assert (nb > 0).all()
    views = {'full': (l / n).to_numpy(), 'behavior_only': (sum(df[f'loss_sum_{c}'] for c in BEHAVIOR) / nb).to_numpy()}
    assert all(np.isfinite(v).all() for v in views.values())
    return views, checks


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--recovery', type=Path, required=True); ap_.add_argument('--out', type=Path, required=True)
    a = ap_.parse_args(); R = a.recovery
    lab = {t: pd.read_parquet(R / f'labels/{t}/incident_calendar_labels.parquet') for t in ['r42', 'r62']}
    res = {}

    def load(tag, name):
        d = pd.read_parquet(R / f'inputs/{tag}/{name}.parquet'); d, nd = strict_dedup(d)
        d = attach_labels(d, lab[tag]); return d, nd

    for i, (tag, name) in enumerate([('r62', 'classes_8b_full'), ('r62', 'classes_3b_full'), ('r42', 'classes_headline')], 1):
        d, nd = load(tag, name); d = d[d.split.isin(['val', 'eval'])].reset_index(drop=True)
        views, checks = class_views(d)
        ft = fold_table(d, views, 'y_corr')
        res[f'{i}_{tag}_{name}'] = dict(rows=len(d), duplicates_removed=nd, checks=checks, n_folds=int(ft.heldout.nunique()),
            fold_mean={v: {m: float(g[m].mean()) for m in ['day_roc', 'day_ap', 'prevalence', 'user_roc']} for v, g in ft.groupby('view')},
            pooled_day_roc={v: roc(d.y_corr, s) for v, s in views.items()})
    # 4: main detector, unseen pool, original vs corrected labels
    d, nd = load('r42', 'main_fullpop'); d = d[d.split.isin(['val', 'eval'])].reset_index(drop=True)
    s = {'adapted_nll': d.adapted_nll.to_numpy(float)}
    out4 = {}
    for ycol in ['y', 'y_corr']:
        ft = fold_table(d, s, ycol)
        out4[ycol] = dict(n_folds=int(ft.heldout.nunique()), day_roc=float(ft.day_roc.mean()), user_roc=float(ft.user_roc.mean()),
                          folds=[(h, sorted(t)) for h, t in folds(d.user_id.to_numpy(), d[ycol].to_numpy())])
    ft_corr = fold_table(d, s, 'y_corr')
    # 6: newly positive scenario-3 insiders
    with tarfile.open(R / 'source/answers.tar.bz2', 'r:bz2') as tar:
        m = [x for x in tar.getmembers() if x.name.endswith('insiders.csv')][0]
        ins = list(csv.DictReader(io.StringIO(tar.extractfile(m).read().decode('utf-8-sig'))))
    s3 = sorted(r['user'] for r in ins if r['dataset'] == '4.2' and r['scenario'] == '3')
    old_pos = set(d.user_id[d.y == 1]); new3 = [u for u in s3 if u not in old_pos][:3]
    out4['scenario3_newly_positive_folds'] = ft_corr[ft_corr.heldout.isin(new3)].to_dict('records')
    res['4_r42_main_unseen'] = {k: (v if k != 'y' and k != 'y_corr' else {kk: vv for kk, vv in v.items() if kk != 'folds'}) for k, v in out4.items()}
    res['4_fold_lists'] = {k: out4[k]['folds'] for k in ['y', 'y_corr']}
    # 5: factorial user ROC and a paired bootstrap on common keys
    arms = {}
    for sc in ['3b', '8b']:
        for mode in ['full', 'no_profile']:
            dd, nd = load('r62', f'factorial_{sc}_{mode}'); arms[(sc, mode)] = dd[dd.split.isin(['val', 'eval'])].set_index('example_id')
    out5 = {}
    for sc in ['3b', '8b']:
        ids = arms[(sc, 'full')].index
        assert set(ids) == set(arms[(sc, 'no_profile')].index)
        base = arms[(sc, 'full')].reset_index()
        sc_views = {mode: arms[(sc, mode)].loc[ids, 'adapted_nll'].to_numpy(float) for mode in ['full', 'no_profile']}
        ft = fold_table(base, sc_views, 'y_corr')
        piv = ft.pivot(index='heldout', columns='view', values='user_roc').sort_index()
        users = list(piv.index); idx = np.random.default_rng(42).integers(0, len(users), (10000, len(users)))
        delta = piv.no_profile.to_numpy()[idx].mean(1) - piv.full.to_numpy()[idx].mean(1)
        out5[sc] = dict(user_roc_full=float(piv.full.mean()), user_roc_no_profile=float(piv.no_profile.mean()),
                        delta=float((piv.no_profile - piv.full).mean()), ci=[float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
                        per_user=piv.to_dict())
    res['5_r62_factorial_user_roc'] = out5
    a.out.write_text(json.dumps(res, indent=1, default=str) + '\n')
    for k, v in res.items():
        if k != '4_fold_lists':
            print(k, json.dumps(v, default=str)[:700])


if __name__ == '__main__':
    main()
