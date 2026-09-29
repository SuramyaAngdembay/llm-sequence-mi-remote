"""Regression checks for the review's Stage B options (docs/review/STAGE_B_PROTOCOL_2026-09-28.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip('torch')

from train_delta_sae_frontier import override_labels          # noqa: E402
from reselect_token_sae_features import fold_view             # noqa: E402


def score_index():
    return pd.DataFrame({'example_idx': [0, 1, 2], 'example_id': ['A:1', 'A:2', 'B:1'], 'y': [1, 0, 0]}).set_index(
        'example_idx', drop=False)


def test_override_replaces_labels(tmp_path):
    p = tmp_path / 'ov.parquet'
    pd.DataFrame({'example_idx': [2, 0, 1], 'example_id': ['B:1', 'A:1', 'A:2'], 'y': [1, 0, 1]}).to_parquet(p)
    assert override_labels(score_index(), p).tolist() == [0, 1, 1]


@pytest.mark.parametrize('frame', [
    {'example_idx': [0, 1], 'example_id': ['A:1', 'A:2'], 'y': [0, 0]},                 # incomplete coverage
    {'example_idx': [0, 1, 2], 'example_id': ['A:1', 'A:9', 'B:1'], 'y': [0, 0, 0]},     # wrong identity
    {'example_idx': [0, 1, 2], 'example_id': ['A:1', 'A:2', 'B:1'], 'y': [0, 2, 0]},     # not binary
])
def test_override_refuses_bad_tables(tmp_path, frame):
    p = tmp_path / 'ov.parquet'
    pd.DataFrame(frame).to_parquet(p)
    with pytest.raises(ValueError):
        override_labels(score_index(), p)


def test_negatives_only_from_listed_users():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([1, 0, 1, 0, 0, 0])
    users = np.array(['disc', 'disc', 'conf', 'conf', 'benign', 'benign'])
    xs, ys, stats = fold_view(x, y, users, {'disc'}, None, {'benign'})
    # the discovery positive is kept; the discovery user's window-negative day and every confirmation row are not
    assert ys.tolist() == [1, 0, 0] and xs[:, 0].tolist() == [0, 8, 10]
    assert stats['n_positive_rows_kept'] == 1 and stats['n_benign_rows'] == 2
    # default behaviour unchanged
    _, ys_default, _ = fold_view(x, y, users, {'disc'})
    assert ys_default.tolist() == [1, 0, 0, 0, 0]


def test_pilot_alpha_edits():
    import importlib
    pis = importlib.import_module('pilot_intervention_specificity')
    Z = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    proto = np.array([4.0, 6.0])
    full = pis.edited_codes(Z, [0, 1], proto, 'union')
    assert np.array_equal(full, pis.edited_codes(Z, [0, 1], proto, 'union', 1.0))
    assert np.array_equal(pis.edited_codes(Z, [0, 1], proto, 'union', 0.0), Z)          # alpha 0 is a no-op
    half = pis.edited_codes(Z, [0, 1], proto, 'union', 0.5)
    assert np.allclose(half[:2], (Z[:2] + full[:2]) / 2) and np.array_equal(half[2], Z[2])  # untouched row stays
    own = pis.edited_codes(Z, [0, 1], proto, 'own', 0.5)
    assert own[0, 0] == 0.0 and own[0, 1] == 4.0 and own[1, 0] == 2.5                    # no writes into zeros
