#!/usr/bin/env python3
"""Presample the dictionary's training rows on a CPU node, with the trainer's own functions.

Calls train_delta_sae_frontier.load_layer_vectors (labels override, 2M cap formula, seed + layer) and
subsample_rows exactly as its main() does, then writes the chosen rows in bounded chunks in the
trainer's input format. Run on that directory, the trainer finds len(rows) == max_rows, samples nothing further,
and trains on exactly these rows. Values are float16 in the cache and are written back as float16 losslessly
(checked). Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_delta_sae_frontier import load_layer_vectors, subsample_rows  # noqa: E402
from cert_label_contracts import load_stage_labels
from cert_calendar_labels import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract-dir', type=Path, required=True)
    ap.add_argument('--labels-override', type=Path, required=True)
    ap.add_argument('--layer', type=int, default=26)
    ap.add_argument('--max-rows', type=int, default=2_000_000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--chunk-rows', type=int, default=65536, help='bound serialization memory without changing sampled rows')
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f'refusing to overwrite {a.out_dir}')
    if a.chunk_rows <= 0:
        raise ValueError('chunk-rows must be positive')
    t0 = time.time()
    seed = a.seed + a.layer                                   # as the trainer's main()
    x, y, meta = load_layer_vectors(a.extract_dir, a.layer, max_rows=a.max_rows, seed=seed,
                                  labels_override=a.labels_override, sampling_policy=True)
    n_loaded, pos_loaded = len(x), int((y > 0).sum())
    x, y, meta = subsample_rows(x, y, meta, max_rows=a.max_rows, seed=seed)
    (a.out_dir / f'layer_{a.layer}').mkdir(parents=True)
    chunks = []
    for cid, start in enumerate(range(0, len(x), a.chunk_rows)):
        stop = min(start + a.chunk_rows, len(x))
        x16 = x[start:stop].astype(np.float16)
        if not np.array_equal(x16.astype(np.float32), x[start:stop]):
            raise ValueError('Rows are not exactly representable in float16')
        path = a.out_dir / f'layer_{a.layer}' / f'chunk_{cid:05d}.pt'
        torch.save({'delta': x16, 'example_idx': meta['example_idx'].iloc[start:stop].to_numpy(np.int64),
                    'position': meta['position'].iloc[start:stop].to_numpy(np.int64)}, path)
        chunks.append({'layer': a.layer, 'chunk_id': cid, 'path': str(path), 'rows': stop-start,
                       'd_model': x.shape[1], 'unit': 'token', 'sha256': file_sha256(path)})
        print(f'presample wrote chunk {cid + 1}: rows {start}:{stop}', flush=True)
    del x16
    scores = pd.read_parquet(a.extract_dir / 'example_scores.parquet').sort_values('example_idx').reset_index(drop=True)
    ov = load_stage_labels(a.labels_override, scores, purpose='sampling')
    for c in ('y', 'y_original', 'calendar_date', 'label_version', 'preserve_for_sampling'):
        scores[c] = ov[c].to_numpy()
    scores.to_parquet(a.out_dir / 'example_scores.parquet', index=False)
    pd.DataFrame(chunks).to_csv(a.out_dir / 'chunk_manifest.csv', index=False)
    src = json.loads((a.extract_dir / 'extract_summary.json').read_text())
    (a.out_dir / 'extract_summary.json').write_text(json.dumps(dict(src, layers=[a.layer], presampled=True), indent=1) + '\n')
    rep = {'source_extract_dir': str(a.extract_dir), 'labels_override': str(a.labels_override),
           'labels_override_sha256': hashlib.sha256(a.labels_override.read_bytes()).hexdigest(),
           'layer': a.layer, 'max_rows': a.max_rows, 'sampling_seed': seed, 'rows_loaded_before_cap': n_loaded,
           'positive_rows_loaded': pos_loaded, 'rows_written': len(x), 'positive_rows_written': int((y > 0).sum()),
           'examples_written': int(meta['example_idx'].nunique()),
           'chunks': chunks, 'sampling_preserved_rows_written': int(meta.preserve_for_sampling.sum()),
           'seconds': round(time.time() - t0, 1)}
    (a.out_dir / 'presample_report.json').write_text(json.dumps(rep, indent=1) + '\n')
    (a.out_dir / 'COMPLETED.json').write_text(json.dumps({'status': 'complete',
        'files': {p.name: file_sha256(p) for p in a.out_dir.iterdir() if p.is_file()},
        'chunks': {str(Path(c['path']).relative_to(a.out_dir)): c['sha256'] for c in chunks}}, indent=2) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
