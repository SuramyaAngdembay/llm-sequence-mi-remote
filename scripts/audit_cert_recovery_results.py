#!/usr/bin/env python3
"""Independent arithmetic checks and supplements for the cached recovery run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cert_calendar_labels import file_sha256
from eval_cert_recovery import score_views, strict_dedup, write_json


def independent_metrics(y, scores):
    """Pair-count ROC and grouped-threshold AP; no sklearn/metric-core calls."""
    y, s = np.asarray(y, int), np.asarray(scores, float)
    assert np.isfinite(s).all() and set(y) == {0, 1}
    neg = np.sort(s[y == 0]); pos = s[y == 1]
    left = np.searchsorted(neg, pos, side='left')
    right = np.searchsorted(neg, pos, side='right')
    auc = np.mean((left + .5*(right-left))/len(neg))
    order = np.argsort(-s, kind='stable')
    ends = np.r_[np.flatnonzero(s[order][1:] != s[order][:-1]), len(s)-1]
    tp = np.cumsum(y[order])[ends]
    ap = np.sum(np.diff(np.r_[0, tp])*(tp/(ends+1)))/y.sum()
    return {'day_roc': float(auc), 'day_ap': float(ap)}


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    ev = root/'evaluation_v1'
    pooled = pd.read_csv(ev/'pooled_metrics.csv')
    manifest = json.loads((ev/'manifest.json').read_text())
    for name, digest in manifest['code_hashes'].items():
        assert file_sha256(Path(__file__).parent/name) == digest, name
    inputs = json.loads((root/'input_manifest.json').read_text())
    for r in inputs['files']:
        assert file_sha256(Path(r['local'])) == r['sha256'], r['local']
    meta = {t:pd.read_parquet(root/'labels'/t/'example_metadata.parquet') for t in ['r42','r62']}
    checks = []
    specs = [('r42','main_fullpop','r42_main','unseen_users','nll'),
             ('r62','main_fullpop','r62_main','unseen_users','nll'),
             ('r42','classes_headline','r42_classes_headline','restricted_139_users','classes')]
    specs += [('r62','classes_'+s+'_full','r62_classes_'+s+'_full','unseen_users','classes') for s in ['3b','8b']]
    for tag, name, run, population, kind in specs:
        df = pd.read_parquet(ev/'relabeled'/f'{tag}_{name}.parquet')
        if name == 'main_fullpop':
            df = df.loc[df.split.isin(['val','eval'])].reset_index(drop=True)
        views, _ = score_views(df, kind)
        for state in ['y_original','y']:
            for view, s in views.items():
                actual = independent_metrics(df[state], s)
                ref = pooled[(pooled.run==run)&(pooled.population==population)&
                             (pooled.label_state==state)&(pooled.view==view)].iloc[0]
                for metric, value in actual.items():
                    err = abs(value-ref[metric])
                    assert err < 1e-12, (run, view, metric, err)
                    checks.append(dict(run=run,state=state,view=view,metric=metric,error=err))
    # Report duplicate effects separately, on a fixed pooled population.
    duplicate_rows = []
    for path in sorted((root/'inputs/r62').glob('*.parquet')):
        if not (path.stem.startswith('factorial_') or path.stem.startswith('classes_')):
            continue
        raw = pd.read_parquet(path); unique, count = strict_dedup(raw)
        lookup = meta['r62'].set_index('example_id').y
        for state in ['y_original','y']:
            for label, frame in [('raw',raw), ('deduplicated',unique)]:
                y = frame.y if state=='y_original' else lookup.loc[frame.example_id].to_numpy()
                vals = independent_metrics(y, frame.adapted_nll)
                duplicate_rows.append(dict(run=path.stem,label_state=state,rows=label,n_rows=len(frame),
                    n_positive=int(np.sum(y)),prevalence=float(np.mean(y)),duplicates=count,**vals))
    pd.DataFrame(duplicate_rows).to_csv(output/'duplicate_effects.csv',index=False)
    # Full-population and token-extraction caches are different scoring passes.
    overlap = []
    for tag in ['r42','r62']:
        a=pd.read_parquet(ev/'relabeled'/f'{tag}_main_fullpop.parquet').set_index('example_id')
        b=pd.read_parquet(ev/'relabeled'/f'{tag}_main_eval.parquet').set_index('example_id')
        for col in ['adapted_nll','base_nll']:
            d=np.abs(a.loc[b.index,col]-b[col])
            overlap.append(dict(dataset=tag,score=col,rows=len(d),exactly_equal=int((d==0).sum()),
                mean_abs_difference=float(d.mean()),max_abs_difference=float(d.max()),
                scope='Separate preserved scoring passes; do not silently substitute one for another'))
    write_json(output/'main_cache_overlap.json',overlap)
    # Restore the numerical models' prespecified mean-gap comparison.
    with np.load(root/'inputs/r62/timeseries_matrix.npz',allow_pickle=True) as z:
        uid=z['user_id']; days=z['day_index']; split=z['split']
    ids=np.char.add(np.char.add(uid.astype(str),':'),days.astype(str))
    ref=meta['r62'].set_index('example_id').loc[ids]
    assert not set(uid[split=='train']) & set(meta['r62'].loc[meta['r62'].y==1,'user_id'])
    caches={}
    for name in ['w1','w7','fc7']:
        with np.load(root/'inputs/r62'/f'timeseries_{name}.npz',allow_pickle=True) as z:
            caches[name]={k:z[k] for k in z.files}
    eligible=np.isin(split,['val','eval'])
    for c in caches.values():eligible &= c['has_history'].astype(bool)
    gaps=[]
    for name,c in caches.items():
        for state in ['y_original','y']:
            positive=ref[state].to_numpy()==1
            pieces={k:float((c['err_sum_'+k]/c['n_total'])[eligible&positive].mean()-
                           (c['err_sum_'+k]/c['n_total'])[eligible&~positive].mean())
                    for k in ['PROFILE','BEHAV']}
            total=float((c['err_sum_total']/c['n_total'])[eligible&positive].mean()-
                        (c['err_sum_total']/c['n_total'])[eligible&~positive].mean())
            assert abs(total-sum(pieces.values()))<1e-9
            gaps.append(dict(run=name,label_state=state,full_gap=total,contributions=pieces,
                profile_share=pieces['PROFILE']/total,
                scope='Common eligible population; signed mean-gap accounting, not causal attribution'))
    write_json(output/'timeseries_mean_gap.json',gaps)
    # Separate master-window / observable-event sensitivity, without selecting a preferred result.
    sensitivity=[]
    event=pd.read_parquet(root/'labels/r62/observable_calendar_labels.parquet')
    event_keys=set(zip(event.user_id,event.day_index))
    for scale in ['3b','8b']:
        df=pd.read_parquet(ev/'relabeled'/f'r62_classes_{scale}_full.parquet')
        labels=np.array([(u,int(d)) in event_keys for u,d in zip(df.user_id,df.day_index)],int)
        vv,_=score_views(df,'classes')
        for view,s in vv.items():
            sensitivity.append(dict(run='r62_classes_'+scale+'_full',view=view,
                target='named-user observable day',n_rows=len(df),n_positive=int(labels.sum()),
                changed_from_master=int((labels!=df.y.to_numpy()).sum()),
                prevalence=float(labels.mean()),**independent_metrics(labels,s)))
    pd.DataFrame(sensitivity).to_csv(output/'observable_day_sensitivity_pooled.csv',index=False)
    keys=['run','population','view','metric']
    folds=pd.read_csv(ev/'fold_summary.csv')
    old=folds[folds.label_state=='y_original'].drop(columns='label_state')
    new=folds[folds.label_state=='y'].drop(columns='label_state')
    registry=old.merge(new,on=keys,suffixes=('_old','_corrected'),validate='one_to_one')
    registry['change']=registry.estimate_corrected-registry.estimate_old
    registry.to_csv(output/'old_vs_corrected_fold_registry.csv',index=False)
    write_json(output/'independent_metric_checks.json',dict(status='pass',checks=len(checks),
        max_error=max(r['error'] for r in checks),details=checks))
    write_json(output/'manifest.json',dict(status='complete',source_evaluation_sha256=file_sha256(ev/'manifest.json'),
        script_sha256=file_sha256(Path(__file__)),inputs_rehashed=len(inputs['files']),
        sensitivity_scope='Pooled r6.2 class-score views only; not a replacement for the primary incident-window target'))
    print(f'Passed {len(checks)} independent ROC/AP comparisons; supplements saved to {output}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out_dir)
