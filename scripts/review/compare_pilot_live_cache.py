#!/usr/bin/env python3
"""Compare every fixed pilot live delta with the original extraction, on CPU."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import file_sha256


def main():
    ap = argparse.ArgumentParser()
    for key in ('extract-dir', 'live-state-dir', 'out-dir'):
        ap.add_argument('--'+key, type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists(): raise ValueError('Use a new comparison directory')
    a.out_dir.mkdir(parents=True)
    t0 = time.monotonic()
    live = json.loads((a.live_state_dir/'COMPLETED.json').read_text())
    if live.get('status') != 'complete': raise ValueError('Live capture is incomplete')
    needed = set(live['needed_examples'])
    extraction = json.loads((a.extract_dir/'extract_summary.json').read_text())
    chunk_size = int(extraction['chunk_examples'])
    rows = []
    for cid in sorted({i//chunk_size for i in needed}):
        path = a.extract_dir/f"layer_{live['layer']}"/f'chunk_{cid:05d}.pt'
        obj = torch.load(path, map_location='cpu', weights_only=False)
        idx = np.asarray(obj['example_idx'])
        for i in sorted(j for j in needed if j//chunk_size == cid):
            name = f'example_{i:05d}.npz'
            if live['files'][name] != file_sha256(a.live_state_dir/name): raise ValueError('Live-state hash changed')
            with np.load(a.live_state_dir/name) as z: fresh = z['delta']
            lo, hi = np.searchsorted(idx, i, side='left'), np.searchsorted(idx, i, side='right')
            old = np.asarray(obj['delta'][lo:hi], dtype=np.float32)
            if not np.array_equal(obj['position'][lo:hi], np.arange(len(fresh))) or old.shape != fresh.shape:
                raise ValueError(f'Cached/live token alignment failed: {i}')
            if not np.isfinite(old).all() or not np.isfinite(fresh).all(): raise ValueError('Nonfinite states')
            error = old.astype(np.float64)-fresh.astype(np.float64)
            denom = float(np.linalg.norm(fresh.astype(np.float64)))
            rows.append({'example_idx': i, 'n_tokens': len(fresh), 'exact': np.array_equal(old, fresh),
                         'max_abs': float(np.abs(error).max()), 'error_l2': float(np.linalg.norm(error)),
                         'fresh_l2': denom, 'relative_l2': float(np.linalg.norm(error)/denom) if denom else None,
                         'changed_coordinates': int(np.count_nonzero(error)), 'coordinates': int(error.size)})
            del old, fresh, error
        del obj
        print(f'compared chunk {cid}', flush=True)
    df = pd.DataFrame(rows).sort_values('example_idx')
    df.to_csv(a.out_dir/'per_example.csv', index=False)
    report = {'status': 'complete', 'examples': len(df), 'exact_examples': int(df.exact.sum()),
              'all_exact': bool(df.exact.all()), 'max_abs': float(df.max_abs.max()),
              'relative_l2_median': float(df.relative_l2.median()), 'relative_l2_max': float(df.relative_l2.max()),
              'source_extract_completion_sha256': file_sha256(a.extract_dir/'COMPLETED.json'),
              'live_manifest_sha256': file_sha256(a.live_state_dir/'COMPLETED.json'),
              'per_example_sha256': file_sha256(a.out_dir/'per_example.csv'), 'seconds': time.monotonic()-t0,
              'interpretation': 'Exact mismatch blocks using these cached deltas as the live batch-1 receiver state; it does not by itself invalidate dictionary training on the declared extraction distribution.'}
    (a.out_dir/'COMPLETED.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
