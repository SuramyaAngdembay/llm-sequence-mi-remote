#!/usr/bin/env python3
"""CPU-only, checksum-bound reanalysis of frozen CERT score caches.

Run after build_cert_calendar_labels.py. Original inputs are never overwritten.
The shared metric core defines scores' summaries; this file defines populations,
provenance checks, paired fold comparisons and the old/corrected registry.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from cert_calendar_labels import VERSION, file_sha256, load_label_artifact, relabel_frame
from eval_metrics_core import ap, dedup_by_id, pooled_metrics, recall_at_fpr, roc, user_max
from eval_fold_aligned_detector_metrics import make_folds
from token_class_decomposition import CERT_CLASSES, CERT_VIEWS

METRICS = ['day_roc', 'day_ap', 'user_roc', 'user_ap', 'within_user_roc',
           'heldout_user_rank_midpoint', 'day_recall_at_fpr_0.001', 'day_recall_at_fpr_0.01']


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, data):
    Path(path).write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + '\n')


def strict_dedup(df: pd.DataFrame):
    if df[['user_id', 'day_index', 'example_id']].isna().any().any():
        raise ValueError('Null score identity')
    # example_idx identifies extraction order, which legitimately differs for duplicates.
    fields = {c: df[c].to_numpy() for c in df if c != 'example_idx'}
    keep = dedup_by_id(df.example_id.to_numpy(), fields)
    out = df.iloc[keep].reset_index(drop=True)
    if out.duplicated(['user_id', 'day_index']).any():
        raise ValueError('Different example IDs refer to the same user-day')
    return out, len(df) - len(out)


def score_views(df: pd.DataFrame, kind: str):
    checks = {}
    numeric = [c for c in df if c.startswith(('loss_sum_', 'n_')) or c.endswith('_nll')]
    if df.empty or not np.isfinite(df[numeric].to_numpy(float)).all():
        raise ValueError('Nonfinite score/count input or empty cache')
    for c in numeric:
        if c.startswith('n_') and (not df[c].ge(0).all() or not df[c].eq(np.floor(df[c])).all()):
            raise ValueError(f'Invalid integer count: {c}')
        if c.startswith('loss_sum_') and not df[c].ge(0).all():
            raise ValueError(f'Negative loss sum: {c}')
    if kind == 'nll':
        views = {'adapted_nll': df.adapted_nll.to_numpy(float),
                 'base_nll': df.base_nll.to_numpy(float),
                 'neg_delta_nll': -df.delta_nll.to_numpy(float)}
        checks['delta_nll_arithmetic_max_error'] = float(np.max(np.abs(df.delta_nll - (df.adapted_nll - df.base_nll))))
        if checks['delta_nll_arithmetic_max_error'] > 2e-6:
            raise ValueError('NLL delta is inconsistent with base and adapted scores')
    elif kind == 'classes':
        for c in CERT_CLASSES:
            if not df[f'n_{c}'].ge(0).all() or not df[f'n_{c}'].eq(np.floor(df[f'n_{c}'])).all():
                raise ValueError('Invalid class target counts')
        total_n = sum(df[f'n_{c}'] for c in CERT_CLASSES)
        total_l = sum(df[f'loss_sum_{c}'] for c in CERT_CLASSES)
        checks['partition_loss_max_error'] = float(np.max(np.abs(total_l - df.loss_sum_total)))
        checks['partition_count_mismatches'] = int((total_n != df.n_targets).sum())
        checks['target_token_count_mismatches'] = int((df.n_targets != df.n_tokens - 1).sum())
        if checks['partition_loss_max_error'] > 1e-8 or checks['partition_count_mismatches'] or checks['target_token_count_mismatches']:
            raise ValueError('Class partition/next-token target gate failed')
        views = {}
        for name, classes in CERT_VIEWS.items():
            counts = sum(df[f'n_{c}'] for c in classes)
            if not counts.gt(0).all():
                raise ValueError(f'View {name} is undefined on some rows; explicit common eligibility required')
            views[name] = (sum(df[f'loss_sum_{c}'] for c in classes) / counts).to_numpy(float)
        checks['full_nll_max_error'] = float(np.max(np.abs(views['full'] - df.adapted_nll)))
        if checks['full_nll_max_error'] > 1e-6:
            raise ValueError('Class sums do not reconstruct the stored NLL')
    else:
        raise ValueError(kind)
    for name, s in views.items():
        if not np.isfinite(s).all():
            raise ValueError(f'Nonfinite score in {name}; refuse silent eligibility change')
    return views, checks


def attach_metadata(df, canonical):
    if not set(df.example_id) <= set(canonical.example_id):
        raise ValueError('Cache contains examples outside the canonical session domain')
    ref = canonical.set_index('example_id').loc[df.example_id]
    for c in ['user_id', 'day_index', 'split']:
        if c in df and not np.array_equal(df[c].to_numpy(), ref[c].to_numpy()):
            raise ValueError(f'Cache metadata conflicts with canonical {c}')
    if not np.array_equal(df.y.to_numpy(), ref.y_original.to_numpy()):
        raise ValueError('Input label history differs from the preserved original metadata')
    return ref


def per_user_within(uid, y, scores):
    groups = pd.DataFrame({'u': uid}).groupby('u', sort=True).indices
    return {name: {str(u): roc(y[idx], s[idx]) for u, idx in groups.items()
                   if np.unique(y[idx]).size == 2} for name, s in scores.items()}


def evaluate_population(df, scores, run, population, state, out_dir, folds=True):
    """All views share exactly the input frame and one fold manifest."""
    uid = df.user_id.to_numpy(); y = df[state].to_numpy(int)
    within = per_user_within(uid, y, scores)
    aggregate = {name: user_max(uid, y, s, np.arange(len(df))) for name, s in scores.items()}
    summary, per_fold = [], []
    for name, s in scores.items():
        r = pooled_metrics(uid, y, s, np.arange(len(df)))
        w = list(within[name].values())
        r.update(run=run, population=population, label_state=state, view=name,
                 within_user_roc=float(np.mean(w)) if w else np.nan, within_user_count=len(w))
        summary.append(r)
    if not folds:
        return summary, per_fold
    fs = make_folds(pd.DataFrame({'user_id': uid, 'y': y}), 42, 800)
    write_json(out_dir / f'{run}__{population}__{state}__folds.json',
        [{'heldout_user': f['heldout_pos_user'], 'test_users': sorted(f['test_users'])} for f in fs])
    for f in fs:
        idx = np.flatnonzero(np.isin(uid, list(f['test_users'])))
        yy = y[idx]
        for name, s in scores.items():
            g = aggregate[name]; g = g.loc[g.user_id.isin(f['test_users'])]
            held = str(f['heldout_pos_user'])
            hs = float(g.loc[g.user_id == held, 'score'].iloc[0])
            rank = 1 + int((g.score > hs).sum()) + 0.5 * (int((g.score == hs).sum()) - 1)
            per_fold.append(dict(run=run, population=population, label_state=state, view=name,
                heldout_user=held, n_rows=len(idx), n_positive=int(yy.sum()), prevalence=float(yy.mean()),
                n_users=len(g), day_roc=roc(yy, s[idx]), day_ap=ap(yy, s[idx]),
                user_roc=roc(g.y.to_numpy(), g.score.to_numpy()), user_ap=ap(g.y.to_numpy(), g.score.to_numpy()),
                within_user_roc=within[name].get(held, np.nan), heldout_user_rank_midpoint=rank,
                **{f'day_recall_at_fpr_{b}': recall_at_fpr(yy, s[idx], b) for b in [0.001, 0.01]}))
    return summary, per_fold


def bootstrap_fold_table(folds: pd.DataFrame, draws=10000):
    summaries, contrasts = [], []
    for keys, group in folds.groupby(['run', 'population', 'label_state']):
        users = sorted(group.heldout_user.unique())
        idx = np.random.default_rng(42).integers(0, len(users), (draws, len(users)))
        baseline = 'full' if 'full' in set(group.view) else 'adapted_nll'
        for metric in METRICS:
            arrays = {}
            for view, sub in group.groupby('view'):
                vals = sub.set_index('heldout_user').reindex(users)[metric].to_numpy(float)
                finite = np.isfinite(vals).all()
                boot = vals[idx].mean(axis=1) if finite else np.full(draws, np.nan)
                arrays[view] = (vals, boot)
                summaries.append(dict(zip(['run', 'population', 'label_state'], keys), view=view, metric=metric,
                    estimate=float(vals.mean()), ci_lo=float(np.percentile(boot, 2.5)),
                    ci_hi=float(np.percentile(boot, 97.5)), n_clusters=len(users), all_folds_defined=bool(finite)))
            if baseline not in arrays:
                continue
            for view, (vals, boot) in arrays.items():
                if view == baseline:
                    continue
                delta = boot - arrays[baseline][1]
                contrasts.append(dict(zip(['run', 'population', 'label_state'], keys), view=view, baseline=baseline,
                    metric=metric, estimate=float((vals-arrays[baseline][0]).mean()),
                    ci_lo=float(np.percentile(delta, 2.5)), ci_hi=float(np.percentile(delta, 97.5)),
                    n_clusters=len(users), interpretation='descriptive positive-user bootstrap; fixed models and negative cohort'))
    return summaries, contrasts


class Recovery:
    def __init__(self, root: Path, output: Path):
        self.root, self.output = root, output
        if output.exists():
            raise FileExistsError('Use a new evaluation directory; never overwrite an analysis')
        output.mkdir(parents=True)
        (output / 'folds').mkdir()
        (output / 'relabeled').mkdir()
        self.meta = {t: pd.read_parquet(root/'labels'/t/'example_metadata.parquet') for t in ['r42', 'r62']}
        self.labels = {t: load_label_artifact(root/'labels'/t/'incident_calendar_labels.parquet') for t in self.meta}
        self.inputs = json.loads((root/'input_manifest.json').read_text())
        self.hashes = {r['local']: r['sha256'] for r in self.inputs['files']}
        self.rows, self.folds, self.coverage, self.gaps, self.loss_contrasts = [], [], [], [], []

    def read(self, path):
        if self.hashes.get(str(path)) != file_sha256(path):
            raise ValueError(f'Input hash mismatch: {path}')
        return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)

    def prepare(self, tag, name, kind, expected='eval', complete=True):
        path = self.root/'inputs'/tag/(name+'.parquet')
        raw = self.read(path)
        df, nd = strict_dedup(raw)
        attach_metadata(df, self.meta[tag])
        expected_meta = self.meta[tag]
        if expected == 'eval':
            expected_meta = expected_meta.loc[expected_meta.split.isin(['val', 'eval'])]
        present = set(df.example_id); want = set(expected_meta.example_id)
        missing = expected_meta.loc[~expected_meta.example_id.isin(present)]
        if present - want or (complete and len(missing)):
            raise ValueError(f'{tag}/{name}: unexpected or missing intended population')
        views, checks = score_views(df, kind)
        corrected = relabel_frame(df, self.labels[tag])
        if not np.array_equal(corrected.y, self.meta[tag].set_index('example_id').loc[df.example_id, 'y']):
            raise ValueError('Independent label lookup disagrees with metadata overlay')
        dest = self.output/'relabeled'/f'{tag}_{name}.parquet'
        corrected.to_parquet(dest, index=False)
        self.coverage.append(dict(run=f'{tag}_{name}', raw_rows=len(raw), unique_rows=len(df),
            duplicates_removed=nd, missing_rows=len(missing), missing_positives=int(missing.y.sum()),
            missing_users=sorted(set(missing.user_id)), complete_intended_population=not len(missing),
            old_positives=int(corrected.y_original.sum()), corrected_positives=int(corrected.y.sum()),
            input_sha256=file_sha256(path), output_sha256=file_sha256(dest), checks=checks))
        if len(missing):
            missing[['example_id', 'user_id', 'day_index', 'y']].to_csv(self.output/f'{tag}_{name}_missing.csv', index=False)
        return corrected, views

    def evaluate(self, df, views, run, population, *, folds=True):
        for state in ['y_original', 'y']:
            print(f'Evaluate {run} {population} {state}: {len(df):,} rows', flush=True)
            rows, ff = evaluate_population(df, views, run, population, state, self.output/'folds', folds)
            self.rows.extend(rows); self.folds.extend(ff)

    def record_gap(self, df, views, run):
        if 'loss_sum_total' not in df:
            return
        for state in ['y_original', 'y']:
            pos = df[state].to_numpy() == 1
            gap = views['full'][pos].mean()-views['full'][~pos].mean()
            pieces = {}
            for c in CERT_CLASSES:
                v = (df[f'loss_sum_{c}']/df.n_targets).to_numpy(float)
                pieces[c] = float(v[pos].mean()-v[~pos].mean())
            if abs(sum(pieces.values())-gap) > 1e-9:
                raise ValueError('Mean-gap partition failure')
            self.gaps.append(dict(run=run, label_state=state, full_gap=gap, contributions=pieces,
                profile_contribution=pieces['DAY']+pieces['PSY'],
                profile_share=(pieces['DAY']+pieces['PSY'])/gap if gap != 0 else None,
                scope='pooled available cohort; signed mean-score accounting, not causal or AUC decomposition'))

    def paired_conditions(self, records, run, population, score='adapted_nll'):
        names = list(records)
        anchor = records[names[0]][0]
        ids = anchor.example_id
        views = {}
        for name, (df, vv) in records.items():
            if set(df.example_id) != set(ids):
                raise ValueError('Cross-arm populations differ')
            idx = pd.Index(df.example_id).get_indexer(ids)
            for col in ['y', 'y_original', 'split']:
                if not np.array_equal(df[col].to_numpy()[idx], anchor[col]):
                    raise ValueError(f'Cross-arm {col} differs')
            views[name] = vv[score][idx]
        self.evaluate(anchor, views, run, population)

    def run(self):
        for tag in ['r42', 'r62']:
            df, views = self.prepare(tag, 'main_fullpop', 'nll', expected='all')
            self.evaluate(df, views, f'{tag}_main', 'all_users')
            for population, splits in [('unseen_users', ['val', 'eval']), ('seen_benign_comparison', ['train', 'eval'])]:
                ix = np.flatnonzero(df.split.isin(splits))
                self.evaluate(df.iloc[ix].reset_index(drop=True), {k:v[ix] for k,v in views.items()}, f'{tag}_main', population,
                              folds=population == 'unseen_users')
            base = self.prepare(tag, 'main_eval', 'nll')
            arms = {'full': base}
            for mode in ['no_psy', 'no_profile']:
                arms[mode] = self.prepare(tag, f'masked_{mode}', 'nll')
            self.paired_conditions(arms, f'{tag}_masking', 'unseen_users')
        for scale in ['3b', '8b']:
            arms = {mode:self.prepare('r62', f'factorial_{scale}_{mode}', 'nll')
                    for mode in ['full', 'no_psy', 'no_profile', 'shuffle_profile']}
            self.paired_conditions(arms, f'r62_factorial_{scale}', 'unseen_users')
        classes = {}
        for tag, name in [('r62', 'classes_3b_full'), ('r62', 'classes_8b_full'),
                          ('r62', 'classes_3b_targetmask'), ('r42', 'classes_headline')]:
            df, views = self.prepare(tag, name, 'classes', complete=(tag=='r62'))
            classes[name] = (df, views)
            population = 'unseen_users' if tag=='r62' else 'restricted_139_users'
            self.evaluate(df, views, f'{tag}_{name}', population)
            self.record_gap(df, views, f'{tag}_{name}')
        self.paired_conditions({'full':classes['classes_3b_full'], 'targetmask':classes['classes_3b_targetmask']},
                               'r62_targetmask_training', 'unseen_users', score='behavior_only')
        self.history()
        self.swaps()
        self.timeseries()
        pd.DataFrame(self.rows).to_csv(self.output/'pooled_metrics.csv', index=False)
        ff = pd.DataFrame(self.folds)
        ff.to_csv(self.output/'fold_metrics.csv', index=False)
        summaries, contrasts = bootstrap_fold_table(ff)
        pd.DataFrame(summaries).to_csv(self.output/'fold_summary.csv', index=False)
        pd.DataFrame(contrasts).to_csv(self.output/'paired_fold_contrasts.csv', index=False)
        write_json(self.output/'coverage.json', self.coverage)
        write_json(self.output/'mean_gap_decomposition.json', self.gaps)
        write_json(self.output/'fixed_sample_loss_contrasts.json', self.loss_contrasts)
        write_json(self.output/'results.json', dict(pooled=self.rows, fold_summary=summaries, paired_contrasts=contrasts,
                   coverage=self.coverage, mean_gap=self.gaps, fixed_sample=self.loss_contrasts))
        scripts = ['eval_cert_recovery.py', 'cert_calendar_labels.py', 'eval_metrics_core.py',
                   'eval_fold_aligned_detector_metrics.py', 'token_class_decomposition.py']
        write_json(self.output/'manifest.json', dict(status='complete', label_version=VERSION,
            protocol_sha256=file_sha256(self.root/'protocol_before_evaluation.md'),
            input_manifest_sha256=file_sha256(self.root/'input_manifest.json'),
            code_hashes={s:file_sha256(Path(__file__).parent/s) for s in scripts},
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
            bootstrap_draws=10000, bootstrap_seed=42,
            scope='cached-score reanalysis; no clean SAE, causal, or baseline-model reruns performed'))

    def loss_summary(self, frame, values, run, contrast):
        for state in ['y_original', 'y']:
            for group in ['all', 'positive', 'negative']:
                mask = np.ones(len(frame), bool) if group=='all' else (frame[state].to_numpy() == (group=='positive'))
                d = pd.DataFrame({'user':frame.user_id.to_numpy()[mask], 'v':values[mask]})
                means = d.groupby('user').v.mean().to_numpy()
                if not len(means):
                    continue
                ix = np.random.default_rng(42).integers(0, len(means), (10000, len(means)))
                b = means[ix].mean(axis=1)
                self.loss_contrasts.append(dict(run=run, contrast=contrast, label_state=state, group=group,
                    n_rows=int(mask.sum()), n_users=len(means), estimate=float(means.mean()),
                    ci_lo=float(np.percentile(b,2.5)), ci_hi=float(np.percentile(b,97.5)),
                    scope='equal-user mean on historical selected records; not a repaired population experiment'))

    def history(self):
        for path in sorted((self.root/'inputs/history_prefix').glob('*/history_prefix_per_example.csv')):
            tag = path.parent.name[:3]
            df, nd = strict_dedup(self.read(path))
            attach_metadata(df, self.meta[tag])
            df = relabel_frame(df, self.labels[tag])
            if not ((df.A_n_targets == df.B_n_targets) & (df.A_n_targets == df.C_n_targets)).all():
                raise ValueError('History conditions have different target counts')
            for view in CERT_VIEWS:
                for c, base in [('B','A'), ('C','A'), ('B','C')]:
                    values = (df[f'{c}_{view}']-df[f'{base}_{view}']).to_numpy(float)
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite history contrast')
                    self.loss_summary(df, values, 'history_'+path.parent.name, f'{c}-{base}_{view}')
            self.coverage.append(dict(run='history_'+path.parent.name, unique_rows=len(df), duplicates_removed=nd,
                old_positives=int(df.y_original.sum()), corrected_positives=int(df.y.sum()),
                scope='historical selected sample; no detection metrics'))

    def swaps(self):
        for tag in ['r42','r62']:
            original, views = self.prepare(tag,'swaps_original','nll',complete=False)
            ids = original.example_id
            for mode in ['profile_swapped','behavior_swapped']:
                df, vv = self.prepare(tag,'swaps_'+mode,'nll',complete=False)
                if set(ids)!=set(df.example_id):
                    raise ValueError('Swap conditions differ in receiver IDs')
                idx = pd.Index(df.example_id).get_indexer(ids)
                self.loss_summary(original,vv['adapted_nll'][idx]-views['adapted_nll'],tag+'_swaps',mode+'-original')

    def timeseries(self):
        def read_npz(path):
            if self.hashes.get(str(path))!=file_sha256(path):
                raise ValueError('NPZ hash mismatch')
            with np.load(path,allow_pickle=True) as z:
                return {k:z[k] for k in z.files}
        matrix = read_npz(self.root/'inputs/r62/timeseries_matrix.npz')
        raw = pd.DataFrame({k:matrix[k] for k in ['user_id','day_index','y','split']})
        raw['example_id'] = raw.user_id+':'+raw.day_index.astype(str)
        attach_metadata(raw,self.meta['r62'])
        df = relabel_frame(raw,self.labels['r62'])
        cache = {name:read_npz(self.root/'inputs/r62'/f'timeseries_{name}.npz') for name in ['w1','w7','fc7']}
        eligible = df.split.isin(['val','eval']).to_numpy()
        for name, c in cache.items():
            for col in ['user_id','y','split']:
                if not np.array_equal(c[col],matrix[col]):
                    raise ValueError('Time-series score/matrix row sequence mismatch')
            eligible &= c['has_history'].astype(bool)
        ix = np.flatnonzero(eligible); part = df.iloc[ix].reset_index(drop=True)
        for name,c in cache.items():
            if not np.allclose(c['err_sum_total'], c['err_sum_PROFILE']+c['err_sum_BEHAV'],rtol=0,atol=1e-9):
                raise ValueError('Time-series class partition mismatch')
            views={v:(c['err_sum_'+k]/c['n_'+k])[ix] for v,k in [('full','total'),('profile_only','PROFILE'),('behavior_only','BEHAV')]}
            if not all(np.isfinite(v).all() for v in views.values()):
                raise ValueError('Nonfinite time-series score')
            self.evaluate(part,views,'r62_timeseries_'+name,'common_history_unseen')
            self.coverage.append(dict(run='r62_timeseries_'+name, unique_rows=len(part),
                old_positives=int(part.y_original.sum()),corrected_positives=int(part.y.sum()),
                scope='common saved history eligibility; NPZ aligned by matrix order; historical checkpoint gates remain applicable'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    Recovery(a.root,a.out_dir).run()
