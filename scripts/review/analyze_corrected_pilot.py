#!/usr/bin/env python3
"""Pre-specified analysis of the corrected pilot (docs/review/STAGE_B_PROTOCOL_2026-09-28.md section 7).

Written before any pilot row existed. Per receiver, delta_<view> = patched minus the same run's zero condition.
Receivers whose edit is zero contribute 0 (the policy edits only where features are active); the share edited
is reported. Per user: mean over that user's receivers. Estimate: mean of user means. Interval: 95% percentile
bootstrap over users, 10,000 draws, seed 42 (the same draws for every endpoint).

Endpoints (behavior_only):
  primary      rpU_S - randI_S, attack days
  co-primary   rpU_S - rpU_C,   attack days
  specificity  [rpU_S - rpU_C](attack) - [rpU_S - rpU_C](matched window-negative), within user
  secondary    rpU_S - randD_S; rpU_C - randI_C; rpU_S - rpO_S; orig_S - rpU_S - recon_S; rpU_S@0.5 vs rpU_S
Diagnostics: every condition's mean delta for full / profile_only / behavior_only, by day type; edit coverage.
Heterogeneity: per user, by scenario, newly included positive vs previously positive users.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_pilot_contracts import load_validated_run
from cert_calendar_labels import file_sha256

DRAWS, SEED = 10_000, 42


def user_values(rows: pd.DataFrame, value: pd.Series) -> pd.Series:
    return value.groupby(rows['user']).mean()


def boot(user_vals: pd.Series, users: list[str], idx: np.ndarray) -> dict:
    v = user_vals.reindex(users).to_numpy(float)
    ok = np.isfinite(v)
    if not ok.all():
        raise SystemExit(f'undefined user value for {np.array(users)[~ok][:5]}')
    b = v[idx].mean(axis=1)
    return {'estimate': float(v.mean()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            'n_users': len(users), 'users_positive': int((v > 0).sum()), 'users_negative': int((v < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--answers', type=Path, required=True, help='official answers.tar.bz2 (scenario per insider)')
    ap.add_argument('--newly-included-users', default='GTD0219,CSC0217,BBS0039,JGT0221,JLM0364')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise ValueError('Refusing to overwrite an analysis')
    rows, man, validity, bal = load_validated_run(a.run_dir)
    if man.get('answers_sha256') != file_sha256(a.answers):
        raise ValueError('Scenario archive differs from the canonical label source')
    with tarfile.open(a.answers, 'r:bz2') as t:
        m = [x for x in t.getmembers() if x.name.endswith('insiders.csv')][0]
        scen = {r['user']: r['scenario'] for r in csv.DictReader(io.StringIO(t.extractfile(m).read().decode('utf-8-sig')))
                if r['dataset'] == '4.2'}
    newly_included = set(a.newly_included_users.split(','))
    conds = man['conditions']
    missing = [c for c in conds if c not in set(rows.condition)]
    required = {'zero', 'rpU_S', 'randI_S', 'rpU_C', 'randD_S', 'randI_C', 'rpO_S', 'orig_S', 'recon_S'}
    if not required <= set(conds):
        raise ValueError('Pilot is missing conditions required by the prespecified endpoints')
    wide = {}
    for v in ('behavior_only', 'full', 'profile_only'):
        w = rows.pivot(index=['receiver_idx', 'user', 'kind', 'pair'], columns='condition', values=f'delta_{v}')
        wide[v] = w.reset_index()
    B = wide['behavior_only']
    att, neg = B[B.kind == 'attack'], B[B.kind == 'window_negative']
    users = sorted(att.user.unique())
    idx = np.random.default_rng(SEED).integers(0, len(users), (DRAWS, len(users)))

    def ep(frame, expr):
        return boot(user_values(frame, expr(frame)), users, idx)

    res = {'run_dir': str(a.run_dir), 'manifest_conditions': conds, 'conditions_missing': missing, 'validity': validity,
           'n_attack_receivers': int(len(att)), 'n_window_negative_receivers': int(len(neg)), 'n_users': len(users),
           'draws': DRAWS, 'seed': SEED, 'endpoints': {}, 'secondary': {}, 'conditions': {}, 'heterogeneity': {}}
    E = res['endpoints']
    E['primary rpU_S - randI_S (attack)'] = ep(att, lambda f: f['rpU_S'] - f['randI_S'])
    E['co-primary rpU_S - rpU_C (attack)'] = ep(att, lambda f: f['rpU_S'] - f['rpU_C'])
    sa = user_values(att, att['rpU_S'] - att['rpU_C']); sn = user_values(neg, neg['rpU_S'] - neg['rpU_C'])
    E['specificity [rpU_S - rpU_C](attack) - (window-negative)'] = boot(sa - sn.reindex(sa.index), users, idx)
    S = res['secondary']
    S['rpU_S - randD_S (attack)'] = ep(att, lambda f: f['rpU_S'] - f['randD_S'])
    S['rpU_C - randI_C (attack)'] = ep(att, lambda f: f['rpU_C'] - f['randI_C'])
    S['rpU_S - rpO_S (attack)'] = ep(att, lambda f: f['rpU_S'] - f['rpO_S'])
    S['orig_S - rpU_S - recon_S (attack)'] = ep(att, lambda f: f['orig_S'] - f['rpU_S'] - f['recon_S'])
    if 'rpU_S@0.5' in att:
        S['rpU_S@0.5 (attack)'] = ep(att, lambda f: f['rpU_S@0.5'])
        S['rpU_S - rpU_S@0.5 (attack)'] = ep(att, lambda f: f['rpU_S'] - f['rpU_S@0.5'])
        S['rpU_C@0.5 (attack)'] = ep(att, lambda f: f['rpU_C@0.5'])
        um_full, um_half = user_values(att, att['rpU_S']).reindex(users), user_values(att, att['rpU_S@0.5']).reindex(users)
        den = um_full.to_numpy()[idx].mean(1)
        key = 'dose ratio mean(rpU_S@0.5) / mean(rpU_S)'
        if abs(um_full.mean()) <= 1e-12 or (np.abs(den) <= 1e-12).any() or np.percentile(den, 2.5) <= 0 <= np.percentile(den, 97.5):
            S[key] = {'estimate': None, 'ci95': None, 'status': 'undefined: denominator near zero or its interval crosses zero'}
        else:
            ratio = um_half.to_numpy()[idx].mean(1) / den
            S[key] = {'estimate': float(um_half.mean() / um_full.mean()),
                      'ci95': [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]}
    for v, w in wide.items():
        for kind in ('attack', 'window_negative'):
            f = w[w.kind == kind]
            for c in conds:
                if c == 'zero' or c not in f:
                    continue
                res['conditions'][f'{v} | {kind} | {c}'] = boot(user_values(f, f[c]), users, idx)
    cov = bal.assign(any_edit=bal.n_edited > 0).groupby(['condition', 'kind'])[['any_edit', 'n_edited', 'norm_l2_total']].mean()
    res['edit_coverage'] = {f'{c} | {k}': {kk: float(vv) for kk, vv in r.items()} for (c, k), r in cov.iterrows()}
    pu = pd.DataFrame({'primary': user_values(att, att['rpU_S'] - att['randI_S']),
                       'co_primary': user_values(att, att['rpU_S'] - att['rpU_C']),
                       'specificity': sa - sn.reindex(sa.index),
                       'n_attack': att.groupby('user').size()}).reindex(users)
    pu['scenario'] = [scen[u] for u in pu.index]
    pu['newly_included_positive_user'] = [u in newly_included for u in pu.index]
    res['heterogeneity']['per_user'] = pu.round(6).reset_index().rename(columns={'index': 'user'}).to_dict('records')
    for key, g in pu.groupby('scenario'):
        res['heterogeneity'][f'scenario {key}'] = {c: float(g[c].mean()) for c in ['primary', 'co_primary', 'specificity']} | {'n_users': len(g)}
    for key, g in pu.groupby('newly_included_positive_user'):
        res['heterogeneity'][f'newly_included_positive_user={key}'] = {c: float(g[c].mean()) for c in ['primary', 'co_primary', 'specificity']} | {'n_users': len(g)}
    res['scope'] = 'Exploratory transductive analysis; newly included users are not an unexposed holdout. Edit balance measures proposed shifts; live-state/realized-edit validation is separate.'
    a.out.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    for k, v in list(E.items()) + list(S.items()):
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
