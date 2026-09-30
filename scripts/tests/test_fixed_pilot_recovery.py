"""Protect user weighting, corrected class pairing, and eligibility sensitivity."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'review'))
from recover_fixed_pilots import analyze_effects, summarize


def test_user_weighting_not_receiver_weighting():
    # Three observations of A must not outweigh the single observation of B.
    frame = pd.DataFrame({'user': ['A', 'A', 'A', 'B'], 'effect': [0., 0., 0., 8.]})
    got = summarize(frame)
    assert got['estimate'] == 4.
    assert got['users'] == 2 and got['receivers'] == 4


def test_corrected_contrast_uses_only_shared_users_and_retains_full_cohort():
    frame = pd.DataFrame([
        ('A', 1, 5., True), ('A', 0, 2., True),
        ('B', 1, 8., False), ('B', 0, 3., False),
        ('C', 1, 100., True),  # No negative class: absent from paired contrast.
    ], columns=['user', 'y', 'effect', 'eligible'])
    frame['endpoint'] = 'example'
    got = analyze_effects(frame)
    contrast = got['example|all_fixed|attack_minus_window_negative']
    assert contrast['estimate'] == 4.
    assert contrast['users'] == 2 and contrast['receivers'] is None
    eligible = got['example|eligible_partner_subset|attack_minus_window_negative']
    assert eligible['estimate'] == 3. and eligible['users'] == 1
    assert eligible['ci95_boot'] is None
    assert got['example|all_fixed|all']['receivers'] == 5
    assert got['example|eligible_partner_subset|all']['receivers'] == 3


def test_nonfinite_effect_is_rejected():
    with pytest.raises(AssertionError):
        summarize(pd.DataFrame({'user': ['A', 'B'], 'effect': [1., float('inf')]}))
