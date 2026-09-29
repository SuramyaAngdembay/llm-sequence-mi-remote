#!/usr/bin/env python3
"""C32 completion (protocol section 2): combine the 40,519-row batch-1 class cache with the 1,949-row extension,
only if the bridge was bit-exact, and evaluate behavior-only vs full on all 70 corrected positive users.

Uses the reviewer's evaluator (independent_metrics.py). Writes a new complete cache; the inputs are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_metrics import fold_table, roc  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import VERSION, file_sha256, load_label_artifact
from cert_label_contracts import corrected_cache, eval_index, unique_identity
from eval_cert_recovery import score_views


def sha(p: Path) -> str:
    return file_sha256(p)


def prepare_complete_cache(old, ext, metadata, labels, order):
    """Attach corrected targets before any publication; reject ambiguous joins."""
    old = corrected_cache(old, metadata, labels)
    ext = corrected_cache(ext, metadata, labels)
    if set(old.example_id) & set(ext.example_id):
        raise ValueError('Extension overlaps the cached rows')
    if set(ext.example_id) != set(order.example_id) - set(old.example_id):
        raise ValueError('Extension is not exactly the missing unseen keys')
    if set(old.example_id) - set(order.example_id):
        raise ValueError('Old cache has rows outside the unseen domain')
    both = pd.concat([old.assign(score_source='historical'), ext.assign(score_source='extension')], ignore_index=True)
    both = both.rename(columns={'example_idx': 'source_example_idx'})
    both = both.set_index('example_id').loc[order.example_id].reset_index()
    both.insert(0, 'example_idx', order.example_idx.to_numpy())
    unique_identity(both)
    score_views(both, 'classes')
    return both


def validate_bridge(old, bridge, extension, manifests):
    """Recompute the numerical bridge against the actual old cache, not a boolean."""
    for f in (old, bridge, extension):
        unique_identity(f)
        score_views(f, 'classes')
    if bridge.empty or not set(bridge.example_id) <= set(old.example_id):
        raise ValueError('Bridge is empty or outside the old cache')
    cols = [c for c in old if c != 'example_idx']
    ref = old.set_index('example_id', drop=False).loc[bridge.example_id]
    if any(c not in bridge or not np.array_equal(ref[c].to_numpy(), bridge[c].to_numpy()) for c in cols):
        raise ValueError('Bridge is not bit-exact against the actual old cache')
    fields = ['adapter_digest', 'tokenizer_digest', 'model_name_or_path', 'max_seq_len',
              'batch_size', 'length_sorted', 'with_base', 'schema', 'classes', 'views',
              'scoring_convention', 'gpu', 'torch']
    for key in fields:
        if any(key not in m for m in manifests) or any(m[key] != manifests[0][key] for m in manifests[1:]):
            raise ValueError(f'Scoring manifest mismatch: {key}')
    if manifests[0]['batch_size'] != 1 or manifests[0]['schema'] != 'cert':
        raise ValueError('This completion requires the CERT batch-1 scoring pass')
    for f, m in zip((old, bridge, extension), manifests):
        if m.get('n_examples') != len(f):
            raise ValueError('Scoring manifest row count mismatch')
    return {'bit_exact': True, 'bridge_rows': len(bridge), 'compared_columns': cols,
            'matching_manifest_fields': fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-cache', type=Path, required=True)
    ap.add_argument('--extension-cache', type=Path, required=True)
    ap.add_argument('--bridge-gate', type=Path, required=True)
    ap.add_argument('--bridge-cache', type=Path, required=True)
    ap.add_argument('--old-manifest', type=Path, required=True)
    ap.add_argument('--bridge-manifest', type=Path, required=True)
    ap.add_argument('--extension-manifest', type=Path, required=True)
    ap.add_argument('--input-manifest', type=Path, required=True, help='frozen B1 inputs_manifest.json')
    ap.add_argument('--bridge-jsonl', type=Path, required=True)
    ap.add_argument('--extension-jsonl', type=Path, required=True)
    ap.add_argument('--eval-jsonl', type=Path, required=True, help='original complete eval JSONL (canonical line indices)')
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True, help='incident_calendar_labels.parquet')
    ap.add_argument('--answers', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f'refusing to overwrite {a.out_dir}')
    gate = json.loads(a.bridge_gate.read_text())
    if gate.get('bit_exact') is not True:
        raise SystemExit('bridge gate not bit-exact: the extension may not be combined with the cached rows')
    old, ext = pd.read_parquet(a.old_cache), pd.read_parquet(a.extension_cache)
    meta = pd.read_parquet(a.metadata)
    inputs = json.loads(a.input_manifest.read_text())
    for key, path in [('class_cache_sha256', a.old_cache), ('original_eval_sha256', a.eval_jsonl),
                      ('bridge_sha256', a.bridge_jsonl), ('extension_sha256', a.extension_jsonl)]:
        if inputs.get(key) != sha(path):
            raise ValueError(f'Frozen B1 input checksum mismatch: {key}')
    bridge = pd.read_parquet(a.bridge_cache)
    for frame, path, nkey in [(bridge, a.bridge_jsonl, 'bridge_rows'), (ext, a.extension_jsonl, 'extension_rows')]:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != inputs[nkey] or frame.example_id.tolist() != [r['example_id'] for r in records]:
            raise ValueError('Scored rows differ from frozen B1 inputs')
    manifests = [json.loads(p.read_text()) for p in (a.old_manifest, a.bridge_manifest, a.extension_manifest)]
    gate = validate_bridge(old, bridge, ext, manifests)
    lab = load_label_artifact(a.labels)
    if lab.attrs['manifest'].get('dataset') != '4.2' or lab.attrs['manifest'].get('archive_sha256') != sha(a.answers):
        raise ValueError('Scenario archive is not the r4.2 label source')
    d = prepare_complete_cache(old, ext, meta, lab, eval_index(a.eval_jsonl, meta))
    all_views, checks = score_views(d, 'classes')
    views = {name: all_views[name] for name in ('full', 'behavior_only', 'profile_only')}
    with tarfile.open(a.answers, 'r:bz2') as t:
        m = [x for x in t.getmembers() if x.name.endswith('insiders.csv')][0]
        scen = {r['user']: r['scenario'] for r in csv.DictReader(io.StringIO(t.extractfile(m).read().decode('utf-8-sig')))
                if r['dataset'] == '4.2'}
    ft = fold_table(d, views, 'y')
    ft['scenario'] = ft.heldout.map(scen)
    ft['in_old_cache'] = ft.heldout.isin(set(old.user_id))
    res = {'rows': int(len(d)), 'positive_days': int(d.y.sum()), 'positive_users': int(d.user_id[d.y == 1].nunique()),
           'checks': checks, 'bridge_gate': gate, 'n_folds': int(ft.heldout.nunique()),
           'fold_mean': {v: {m: float(g[m].mean()) for m in ['day_roc', 'day_ap', 'prevalence', 'user_roc']}
                         for v, g in ft.groupby('view')},
           'pooled_day_roc': {v: roc(d.y, s) for v, s in views.items()},
           'pooled_prevalence': float(d.y.mean()), 'label_version': VERSION,
           'inputs': {k: {'path': str(v), 'sha256': sha(v)} for k, v in vars(a).items() if isinstance(v, Path) and k != 'out_dir'}}
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
    # Publish the directory only after every label, score and metric gate passes.
    a.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.c32-', dir=a.out_dir.parent) as temp:
        stage = Path(temp) / 'complete'
        stage.mkdir()
        cache_path = stage / 'class_scores_complete_calendar_v1.parquet'
        d.to_parquet(cache_path, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(cache_path), d)
        res['output_sha256'] = sha(cache_path)
        ft.to_csv(stage / 'fold_metrics_complete_70.csv', index=False)
        (stage / 'c32_complete_results.json').write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
        completion = {'status': 'complete', 'label_version': VERSION,
                      'files': {p.name: sha(p) for p in sorted(stage.iterdir())}}
        (stage / 'COMPLETED.json').write_text(json.dumps(completion, indent=1) + '\n')
        if a.out_dir.exists():
            raise ValueError('Output appeared while evaluating; refusing overwrite')
        stage.rename(a.out_dir)
    print(json.dumps({k: v for k, v in res.items() if k not in ('by_scenario', 'checks', 'bridge_gate')}, indent=1))


if __name__ == '__main__':
    main()
