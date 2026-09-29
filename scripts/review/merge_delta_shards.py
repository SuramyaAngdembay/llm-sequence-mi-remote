#!/usr/bin/env python3
"""Validate extraction shards, attach verified labels, then publish atomically.

No model is run. Chunks are checked and symlinked without changing values.
Existing directories are never treated as success or overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import VERSION, file_sha256
from cert_label_contracts import TARGET, eval_index, load_stage_labels
from eval_cert_recovery import score_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=Path, nargs='+', required=True)
    ap.add_argument('--layer', type=int, default=26)
    ap.add_argument('--eval-jsonl', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True, help='typed ranking_labels_v1.parquet from labels_v2')
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise ValueError(f'refusing to overwrite {a.out_dir}')
    a.out_dir = a.out_dir.resolve()
    if len(set(s.resolve() for s in a.shards)) != len(a.shards):
        raise ValueError('Repeated shard directory')
    summaries = [json.loads((s / 'extract_summary.json').read_text()) for s in a.shards]
    fields = ['chunk_examples', 'layers', 'pool_unit', 'adapter_dir', 'model_name_or_path',
              'base_via_disable_adapter', 'batch_size', 'max_batch_tokens', 'ce_seq_chunk', 'split']
    for key in fields:
        if any(key not in s or s[key] != summaries[0][key] for s in summaries):
            raise ValueError(f'Shard extraction configuration mismatch: {key}')
    if summaries[0]['layers'] != [a.layer] or summaries[0]['pool_unit'] != 'token':
        raise ValueError('This merge requires a single token-level layer')
    per = summaries[0]['chunk_examples']
    if not isinstance(per, int) or per <= 0:
        raise ValueError('Invalid chunk size')
    frames = [pd.read_parquet(s / 'example_scores.parquet') for s in a.shards]
    for f, s in zip(frames, summaries):
        if s['n_examples'] != len(f) or f.example_idx.tolist() != list(range(s['start_index'], s['start_index'] + len(f))):
            raise ValueError('Shard row count/order disagrees with summary')
    scores = pd.concat(frames, ignore_index=True).sort_values('example_idx').reset_index(drop=True)
    if scores.example_idx.tolist() != list(range(len(scores))):
        raise ValueError('Score indices do not tile 0..N-1')
    labels = load_stage_labels(a.labels, scores, purpose='target')
    order = eval_index(a.eval_jsonl, labels)
    if scores.example_id.tolist() != order.example_id.tolist():
        raise ValueError('Shard indices disagree with canonical evaluation line order')
    for col in TARGET:
        scores[col] = labels[col].to_numpy()
    score_views(scores, 'nll')
    if not (scores.n_tokens > 0).all():
        raise ValueError('Empty token sequence')
    ntok = scores.set_index('example_idx').n_tokens
    manifests, sources = [], []
    for shard in a.shards:
        m = pd.read_csv(shard / 'chunk_manifest.csv')
        if set(m.layer) != {a.layer}:
            raise ValueError('Unexpected layer in chunk manifest')
        for r in m.itertuples():
            actual = (shard / f'layer_{a.layer}' / f'chunk_{int(r.chunk_id):05d}.pt').resolve()
            declared = Path(r.path)
            if not declared.is_absolute():
                declared = actual.parent / declared.name
            if declared.resolve() != actual or not actual.is_file():
                raise ValueError('Chunk manifest path disagrees with its owning shard')
            sources.append((int(r.chunk_id), actual))
        manifests.append(m)
    man = pd.concat(manifests, ignore_index=True).sort_values('chunk_id')
    expected_chunks = (len(scores) + per - 1) // per
    if man.chunk_id.tolist() != list(range(expected_chunks)):
        raise ValueError('Chunk IDs do not cover the complete evaluation domain')
    sources.sort()
    a.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.merge-', dir=a.out_dir.parent) as tmp:
        stage = Path(tmp) / 'complete'
        (stage / f'layer_{a.layer}').mkdir(parents=True)
        checks = []
        for (cid, src), r in zip(sources, man.itertuples()):
            obj = torch.load(src, map_location='cpu', weights_only=False)
            ex, pos, delta = np.asarray(obj['example_idx']), np.asarray(obj['position']), np.asarray(obj['delta'])
            if ex.ndim != 1 or pos.ndim != 1 or delta.ndim != 2 or len(pos) != len(ex) or len(delta) != len(ex) or not len(ex):
                raise ValueError(f'{src.name}: invalid array shape')
            if not np.issubdtype(ex.dtype, np.integer) or not np.issubdtype(pos.dtype, np.integer):
                raise ValueError('Token indices/positions must be integers')
            lo, hi = cid * per, min((cid + 1) * per, len(scores))
            u, starts, counts = np.unique(ex, return_index=True, return_counts=True)
            if not np.array_equal(u, np.arange(lo, hi)) or not (np.diff(ex) >= 0).all():
                raise ValueError(f'{src.name}: missing/unexpected examples or ungrouped rows')
            for e, s0, count in zip(u, starts, counts):
                if count != ntok[e] or not np.array_equal(pos[s0:s0 + count], np.arange(count)):
                    raise ValueError(f'{src.name}: example {e} token positions/count mismatch')
            if int(r.rows) != len(ex) or int(r.d_model) != delta.shape[1] or not np.isfinite(delta).all():
                raise ValueError(f'{src.name}: manifest shape mismatch or nonfinite values')
            (stage / f'layer_{a.layer}' / src.name).symlink_to(src)
            checks.append({'chunk_id': cid, 'rows': len(ex), 'examples': len(u), 'd_model': delta.shape[1],
                           'source': str(src), 'sha256': file_sha256(src)})
            del obj, delta, ex, pos
            print(f'validated chunk {cid + 1}/{expected_chunks}', flush=True)
        if len({c['d_model'] for c in checks}) != 1:
            raise ValueError('Hidden dimension changes between chunks')
        scores.to_parquet(stage / 'example_scores.parquet', index=False)
        man['path'] = [str(a.out_dir / f'layer_{a.layer}' / src.name) for _, src in sources]
        man.to_csv(stage / 'chunk_manifest.csv', index=False)
        summary = dict(summaries[0], n_examples=len(scores), shards=[dict(s, dir=str(d)) for s, d in zip(summaries, a.shards)],
                       cache_version='aquaman-l26-calendar-v2', label_version=VERSION)
        summary.pop('start_index', None)
        summary['batch_sizes_used'] = {key: sum(s['batch_sizes_used'].get(key, 0) for s in summaries)
                                       for key in sorted(set().union(*(s['batch_sizes_used'] for s in summaries)))}
        (stage / 'extract_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        report = {'examples': len(scores), 'chunks': len(checks), 'rows': sum(c['rows'] for c in checks),
                  'label_version': VERSION, 'positive_days': int(scores.y.sum()), 'chunk_checks': checks,
                  'eval_jsonl_sha256': file_sha256(a.eval_jsonl), 'labels_sha256': file_sha256(a.labels),
                  'provenance_scope': 'shard summaries agree; historical extractor did not record start-time model/code digests',
                  'source_files': {str(s / name): file_sha256(s / name) for s in a.shards
                                   for name in ('extract_summary.json', 'example_scores.parquet', 'chunk_manifest.csv')}}
        (stage / 'merge_report.json').write_text(json.dumps(report, indent=2) + '\n')
        done = {'status': 'complete', 'label_version': VERSION,
                'files': {p.name: file_sha256(p) for p in stage.iterdir() if p.is_file()}}
        (stage / 'COMPLETED.json').write_text(json.dumps(done, indent=2) + '\n')
        if a.out_dir.exists():
            raise ValueError('Output appeared while merging')
        stage.rename(a.out_dir)
    print(json.dumps({k: report[k] for k in ('examples', 'chunks', 'rows', 'positive_days')}))


if __name__ == '__main__':
    main()
