"""A separately versioned random-direction policy conditioned on BF16 feasibility.

Original passing draws are retained. Failed draws get at most 32 independent
dictionary candidates. No loss values or labels enter selection. This changes
the control distribution and must never be called the original random policy.
"""
from __future__ import annotations
import hashlib
import numpy as np

POLICY = 'dictionary-control-bf16-rejection-v1'


def realized_bfloat16(before, requested):
    import torch
    h = torch.from_numpy(np.asarray(before, dtype=np.float32)).to(torch.bfloat16)
    if not np.array_equal(h.float().numpy(), before):
        raise ValueError('Before-state is not exactly representable as BF16')
    q = torch.from_numpy(np.asarray(requested, dtype=np.float32)).to(torch.bfloat16)
    return (h + q).float().numpy().astype(np.float64) - h.float().numpy().astype(np.float64)


def match_dictionary_control(before, reference, original, directions, *, seed, receiver, condition,
                             tolerance=.05, max_candidates=32):
    arrays = [np.asarray(x, dtype=np.float32) for x in (before, reference, original, directions)]
    before, reference, original, directions = arrays
    if (before.ndim != 2 or reference.shape != before.shape or original.shape != before.shape
            or directions.ndim != 2 or directions.shape[1] != before.shape[1] or not len(directions)
            or not all(np.isfinite(x).all() for x in arrays)):
        raise ValueError('Invalid physical-control arrays')
    if tolerance != .05 or max_candidates != 32:
        raise ValueError('This version fixes the tolerance and candidate budget')
    if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-6, rtol=1e-6):
        raise ValueError('Dictionary directions must be unit length')
    if condition not in ('randD_S', 'randD_C'):
        raise ValueError('This policy applies only to dictionary controls')
    q = np.linalg.norm(reference.astype(np.float64), axis=1)
    old = np.linalg.norm(original.astype(np.float64), axis=1)
    if not np.array_equal(q > 0, old > 0) or not np.allclose(q, old, rtol=1e-6, atol=0.):
        raise ValueError('Original requested controls are not norm/support matched')
    target = np.linalg.norm(realized_bfloat16(before, reference), axis=1)
    actual = np.linalg.norm(realized_bfloat16(before, original), axis=1)
    # A zero reference edit at a requested-active token remains an unresolved
    # intervention design failure rather than accepting a vacuous zero control.
    if ((q > 0) & (target == 0)).any():
        raise ValueError('Reference edit vanished in BF16')
    bad = ((target > 0) != (actual > 0))
    active = target > 0
    bad[active] |= np.abs(actual[active]-target[active])/target[active] > tolerance
    result = original.copy()
    log = []
    salt = int(hashlib.sha256(POLICY.encode()).hexdigest()[:8], 16)
    tag = int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16)
    for pos in np.flatnonzero(bad):
        rng = np.random.default_rng([int(seed), int(receiver), int(pos), tag, salt])
        accepted = False
        for attempt in range(1, max_candidates + 1):
            index = int(rng.integers(0, len(directions)))
            # Keep the original requested norm calculation used by shift_random.
            candidate = directions[index] * np.linalg.norm(reference[pos])
            norm = float(np.linalg.norm(realized_bfloat16(before[pos:pos+1], candidate[None,:])[0]))
            error = abs(norm-target[pos])/target[pos]
            accepted = norm > 0 and error <= tolerance
            log.append(dict(position=int(pos), attempt=attempt, direction_pool_index=index,
                realized_norm=norm, target_realized_norm=float(target[pos]), relative_error=float(error), accepted=accepted))
            if accepted:
                result[pos] = candidate
                break
        if not accepted:
            raise ValueError(f'No feasible dictionary control in {max_candidates} attempts at position {pos}')
    if not np.array_equal(result[~bad], original[~bad]):
        raise AssertionError('An already passing draw changed')
    return result, dict(policy=POLICY, changed_positions=np.flatnonzero(bad).tolist(),
                        candidates=log, tolerance=tolerance, max_candidates=max_candidates,
                        selection_uses_losses_or_labels=False)
