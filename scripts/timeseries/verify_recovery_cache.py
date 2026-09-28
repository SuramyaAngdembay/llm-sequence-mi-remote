#!/usr/bin/env python3
"""Read-only CPU verification of saved time-series score row identity.

Print JSON to stdout. No model, cache, or historical gate is overwritten.
Rescore every row, verify context boundaries and saved standardisation, and
record hashes. CPU/GPU arithmetic is compared at rtol=atol=1e-5, declared here
before inspecting differences; retain the original cached scores for metrics.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_recon_detector import DayAutoencoder, build_windows


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(matrix_path, run_dir):
    start = time.monotonic()
    paths = [matrix_path, run_dir/'model.pt', run_dir/'channel_scores.npz', run_dir/'train_manifest.json']
    hashes = {str(p): sha(p) for p in paths}
    with np.load(matrix_path, allow_pickle=True) as z:
        matrix = {k: z[k] for k in z.files}
    with np.load(run_dir/'channel_scores.npz', allow_pickle=True) as z:
        cache = {k: z[k] for k in z.files}
    ck = torch.load(run_dir/'model.pt', map_location='cpu', weights_only=False)
    manifest = json.loads((run_dir/'train_manifest.json').read_text())
    uid, days = matrix['user_id'], matrix['day_index']
    boundaries = np.r_[True, uid[1:] != uid[:-1]]
    assert len(set(uid[boundaries])) == int(boundaries.sum()), 'Users are not contiguous'
    assert np.all((days[1:] > days[:-1]) | boundaries[1:]), 'Dates are not strictly increasing within user'
    for field in ['user_id', 'y', 'split']:
        assert np.array_equal(cache[field], matrix[field]), field
    assert np.array_equal(ck['channels'], matrix['channels'])
    assert np.array_equal(ck['channel_classes'], matrix['channel_classes'])
    X = matrix['X'].astype(np.float32)
    train = matrix['split'] == 'train'
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
    sd[sd < 1e-8] = 1.0
    assert np.array_equal(mu, ck['mu']) and np.array_equal(sd, ck['sd']), 'Training standardisation differs'
    assert np.isfinite(X).all() and np.isfinite(sd).all() and (sd > 0).all()
    objective = ck['objective']
    assert objective == manifest['objective']
    win = build_windows(uid, int(manifest['window']))
    history = np.ones(len(uid), bool)
    if objective == 'forecast':
        win = win[:, :-1]
        history = ~boundaries
    assert win.shape[1] == int(ck['ctx_len'])
    assert np.array_equal(history, cache['has_history'])
    # Check in chunks to keep peak memory bounded on the shared CPU host.
    for lo in range(0, len(uid), 4096):
        rows = np.arange(lo, min(lo+4096, len(uid)))
        assert np.all(uid[win[rows]] == uid[rows, None]), 'Cross-user context'
        if objective == 'forecast':
            assert np.all(win[rows][history[rows]] < rows[history[rows], None]), 'Target/future leakage'
    model = DayAutoencoder(int(ck['n_channels']), int(ck['ctx_len']), int(ck['hidden']), int(ck['latent']))
    model.load_state_dict(ck['state_dict'])
    model.eval()
    xs = torch.from_numpy((X-mu)/sd)
    classes = np.asarray(ck['channel_classes'])
    unique = sorted(set(classes))
    max_abs = dict.fromkeys(unique, 0.)
    max_scaled = dict.fromkeys(unique, 0.)
    for c in unique:
        assert int(cache['n_'+c]) == int((classes == c).sum())
        assert np.isfinite(cache['err_sum_'+c]).all()
    assert int(cache['n_total']) == len(classes)
    with torch.no_grad():
        for lo in range(0, len(uid), 4096):
            rows = np.arange(lo, min(lo+4096, len(uid)))
            se = (model(xs[win[rows]])-xs[rows]).square().numpy()
            for c in unique:
                actual = se[:, classes == c].sum(axis=1)
                expected = cache['err_sum_'+c][rows]
                assert np.isfinite(actual).all()
                error = np.abs(actual-expected)
                max_abs[c] = max(max_abs[c], float(error.max()))
                max_scaled[c] = max(max_scaled[c], float((error/(1e-5+1e-5*np.abs(expected))).max()))
    assert hashes == {str(p): sha(p) for p in paths}, 'An input changed during verification'
    passed = all(v <= 1. for v in max_scaled.values())
    return dict(status='pass' if passed else 'fail', rows_verified=len(uid), device='cpu',
        threads=torch.get_num_threads(), objective=objective, window=manifest['window'],
        excluded_no_history=int((~history).sum()), max_abs_error=max_abs,
        max_error_fraction_of_tolerance=max_scaled, rtol=1e-5, atol=1e-5,
        hashes=hashes, torch=torch.__version__, elapsed_seconds=time.monotonic()-start,
        scope='Full saved-matrix order and checkpoint/cache consistency; not a training rerun or a history-shuffle experiment')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    result = verify(a.matrix, a.run_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'pass' else 1)
