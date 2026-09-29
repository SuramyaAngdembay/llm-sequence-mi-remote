"""Measurements of proposed, cast and realized hidden-state interventions."""
from __future__ import annotations

import numpy as np
import torch


def apply_token_shift(hidden, shifts, attention_mask, *, strict=False, observer=None, offset=0):
    """Apply the historical dtype-cast/add path and optionally observe its result."""
    if len(shifts) != hidden.shape[0]:
        raise ValueError('Patch batch size mismatch')
    patch = torch.zeros_like(hidden)
    for bi, mat in enumerate(shifts):
        mat = np.asarray(mat)
        valid = int(attention_mask[bi].sum().item())
        if strict:
            if not torch.equal(attention_mask[bi].bool(), torch.arange(hidden.shape[1], device=hidden.device) < valid):
                raise ValueError('Strict patching requires right padding')
            if mat.shape != (valid, hidden.shape[2]) or not np.isfinite(mat).all():
                raise ValueError('Patch shape/finite-value gate failed')
        take = min(valid, int(mat.shape[0]))
        if take:
            patch[bi, :take] = torch.as_tensor(mat[:take], device=hidden.device, dtype=hidden.dtype)
    after = hidden + patch
    if observer is not None:
        for bi, mat in enumerate(shifts):
            valid = int(attention_mask[bi].sum().item())
            observer(offset + bi, hidden[bi, :valid], np.asarray(mat), patch[bi, :valid], after[bi, :valid])
    return after


def realized_token_stats(before, proposed, cast, after):
    """Return one row per token; zeros remain present for exact support checks."""
    before, after, cast = [x.detach().float().cpu().numpy() for x in (before, after, cast)]
    requested = np.asarray(proposed, dtype=np.float64)
    realized = after.astype(np.float64) - before.astype(np.float64)
    cast = cast.astype(np.float64)
    if requested.shape != realized.shape or not all(np.isfinite(x).all() for x in (before, after, cast, requested)):
        raise ValueError('Invalid realized-edit measurement')
    q = np.linalg.norm(requested, axis=1)
    r = np.linalg.norm(realized, axis=1)
    return {'requested_l2': q, 'cast_l2': np.linalg.norm(cast, axis=1), 'realized_l2': r,
            'realization_error_l2': np.linalg.norm(realized-requested, axis=1),
            'requested_nonzero_coordinates': np.count_nonzero(requested, axis=1),
            'realized_nonzero_coordinates': np.count_nonzero(realized, axis=1),
            'requested_dot_realized': np.sum(requested*realized, axis=1)}


def check_realized_controls(tokens, *, relative_tolerance=0.05):
    """Assess physical support/norm matching; do not use outcome scores."""
    if not np.isfinite(tokens.realized_l2).all() or (tokens.realized_l2 < 0).any():
        raise ValueError('Invalid realized norms')
    report = {'relative_tolerance': relative_tolerance, 'comparisons': {}, 'passed': True}
    for arm in ('S', 'C'):
        ref = tokens[tokens.condition == f'rpU_{arm}'].set_index(['receiver_idx', 'position'])
        if ref.empty:
            continue
        for control in (f'randI_{arm}', f'randD_{arm}'):
            other = tokens[tokens.condition == control].set_index(['receiver_idx', 'position'])
            if other.empty:
                raise ValueError(f'Missing declared control {control}')
            if not ref.index.is_unique or not other.index.is_unique or set(ref.index) != set(other.index):
                raise ValueError('Realized-edit comparison lacks identical token grids')
            other = other.loc[ref.index]
            a, b = ref.realized_l2.to_numpy(), other.realized_l2.to_numpy()
            mismatch = (a > 0) != (b > 0)
            active = a > 0
            err = np.abs(b[active]-a[active])/a[active]
            passed = bool(active.any()) and not mismatch.any() and not (err > relative_tolerance).any()
            report['comparisons'][control] = {'reference': f'rpU_{arm}', 'tokens': len(a),
                'support_mismatches': int(mismatch.sum()), 'active_reference_tokens': int(active.sum()),
                'norm_mismatch_tokens': int((err > relative_tolerance).sum()),
                'max_relative_norm_error': float(err.max()) if len(err) else 0., 'passed': passed}
            report['passed'] = report['passed'] and passed
    if not report['comparisons']:
        raise ValueError('No declared random-control comparison was measured')
    return report
