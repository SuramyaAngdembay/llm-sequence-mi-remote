#!/usr/bin/env python3
"""Merge chunk-aligned extraction shards into one extract directory (cache aquaman-l26-v1), with integrity checks.

Checks every chunk: example_idx inside the chunk's global range [512*id, 512*(id+1)), positions 0..n-1 per example,
and n equal to the score index's n_tokens. Score indices must tile 0..N-1 exactly. Chunk files are symlinked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=Path, nargs='+', required=True)
    ap.add_argument('--layer', type=int, default=26)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f'refusing to overwrite {a.out_dir}')
    summaries = [json.loads((s / 'extract_summary.json').read_text()) for s in a.shards]
    per = {s['chunk_examples'] for s in summaries}
    assert len(per) == 1, per
    per = per.pop()
    scores = pd.concat([pd.read_parquet(s / 'example_scores.parquet') for s in a.shards]).sort_values('example_idx')
    if scores.example_idx.tolist() != list(range(len(scores))):
        raise SystemExit('score indices do not tile 0..N-1')
    ntok = scores.set_index('example_idx').n_tokens
    man = pd.concat([pd.read_csv(s / 'chunk_manifest.csv') for s in a.shards]).sort_values('chunk_id')
    man = man[man.layer == a.layer]
    if man.chunk_id.tolist() != list(range(len(man))):
        raise SystemExit('chunk ids are not contiguous')
    (a.out_dir / f'layer_{a.layer}').mkdir(parents=True)
    checks = []
    for _, r in man.iterrows():
        src = Path(r.path); dst = a.out_dir / f'layer_{a.layer}' / src.name
        obj = torch.load(src, map_location='cpu', weights_only=False)
        ex = np.asarray(obj['example_idx']); pos = np.asarray(obj['position'])
        lo, hi = int(r.chunk_id) * per, (int(r.chunk_id) + 1) * per
        if ex.min() < lo or ex.max() >= hi:
            raise SystemExit(f'{src.name}: example_idx outside [{lo},{hi})')
        u, starts, counts = np.unique(ex, return_index=True, return_counts=True)
        if not (np.diff(ex) >= 0).all():
            raise SystemExit(f'{src.name}: rows not grouped by example')
        for e, s0, c in zip(u, starts, counts):
            if c != ntok[e] or not np.array_equal(pos[s0:s0 + c], np.arange(c)):
                raise SystemExit(f'{src.name}: example {e} has {c} rows / bad positions (n_tokens {ntok[e]})')
        if obj['delta'].shape[0] != len(ex) or not np.isfinite(np.asarray(obj['delta'], dtype=np.float32)).all():
            raise SystemExit(f'{src.name}: shape or non-finite values')
        dst.symlink_to(src)
        checks.append({'chunk_id': int(r.chunk_id), 'rows': int(len(ex)), 'examples': int(len(u))})
        del obj
    covered = sum(c['examples'] for c in checks)
    if covered != len(scores):
        raise SystemExit(f'chunks cover {covered} examples, score index has {len(scores)}')
    scores.to_parquet(a.out_dir / 'example_scores.parquet', index=False)
    man = man.assign(path=[str(a.out_dir / f'layer_{a.layer}' / Path(p).name) for p in man.path])
    man.to_csv(a.out_dir / 'chunk_manifest.csv', index=False)
    summary = dict(summaries[0], n_examples=int(len(scores)), shards=[dict(s, dir=str(d)) for s, d in zip(summaries, a.shards)],
                   cache_version='aquaman-l26-v1')
    summary.pop('start_index', None)
    (a.out_dir / 'extract_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    rep = {'examples': int(len(scores)), 'chunks': len(checks), 'rows': int(sum(c['rows'] for c in checks)),
           'integrity': 'every chunk checked: global ranges, per-example positions 0..n-1, n == n_tokens, finite values',
           'score_index_sha256': hashlib.sha256((a.out_dir / 'example_scores.parquet').read_bytes()).hexdigest()}
    (a.out_dir / 'merge_report.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
