"""Reject incomplete corrected-pilot runs before computing any effect estimate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cert_calendar_labels import VERSION, file_sha256

VIEWS = ('full', 'profile_only', 'behavior_only')
FILES = ('pilot1_rows.csv', 'pilot1_manifest.json', 'validity.json', 'edit_balance_rows.csv')
LIMITS = {'orig_equals_residual_plus_reconstruction_max_abs': 1e-4,
          'zero_edit_vs_unhooked_max_abs': 1e-5, 'alpha_zero_vs_unhooked_max_abs': 1e-5}


def validate_validity(validity, n_receivers):
    if (validity.get('token_rows_equal_tokenizer_length') is not True or validity.get('misaligned') != []
            or validity.get('n_receivers') != n_receivers or validity.get('tokenizer_padding_side') != 'right'):
        raise ValueError('Pilot token-alignment validity gate failed')
    for key, limit in LIMITS.items():
        value = validity.get(key)
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f'Pilot validity gate failed: {key}')


def validate_tables(rows, man, validity, balance):
    if man.get('label_version') != VERSION or man.get('contract') != 'corrected-pilot-v2':
        raise ValueError('Pilot lacks the corrected label/output contract')
    expected = pd.DataFrame(man['expected_receivers'])
    cols = ['receiver_idx', 'receiver_id', 'user', 'kind', 'pair', 'y']
    if expected.empty or not set(cols) <= set(expected) or expected[cols].isna().any().any():
        raise ValueError('Missing expected receiver identities')
    if expected.receiver_idx.duplicated().any() or expected.receiver_id.duplicated().any():
        raise ValueError('Duplicate planned receiver')
    if set(expected.kind) != {'attack', 'window_negative'} or not np.array_equal(expected.y, expected.kind.map({'attack': 1, 'window_negative': 0})):
        raise ValueError('Planned receiver labels disagree with their kind')
    for _, pair in expected.groupby('pair'):
        if len(pair) != 2 or pair.user.nunique() != 1 or set(pair.kind) != {'attack', 'window_negative'}:
            raise ValueError('Invalid planned receiver pair')
    if man.get('n_users') != expected.user.nunique() or man.get('n_pairs') != expected.pair.nunique():
        raise ValueError('Planned population counts disagree')
    conds = man['conditions']
    if not conds or conds[0] != 'zero' or len(set(conds)) != len(conds):
        raise ValueError('Conditions must be unique and start with zero')
    if not set(cols + ['condition']) <= set(rows) or rows[cols + ['condition']].isna().any().any():
        raise ValueError('Missing row identity')
    grid = {(i, c) for i in expected.receiver_idx for c in conds}
    keys = list(zip(rows.receiver_idx, rows.condition))
    if len(keys) != len(set(keys)) or set(keys) != grid:
        raise ValueError('Pilot rows do not equal the unique planned receiver-condition grid')
    ref = expected.set_index('receiver_idx').loc[rows.receiver_idx]
    for col in cols[1:]:
        if rows[col].tolist() != ref[col].tolist():
            raise ValueError(f'Pilot row identity/label mismatch: {col}')
    if 'label_version' not in rows or set(rows.label_version) != {VERSION}:
        raise ValueError('Pilot row label version mismatch')
    for v in VIEWS:
        numeric = [f'{k}_{v}' for k in ('base', 'patched', 'delta', 'n_targets')]
        if not set(numeric) <= set(rows) or not np.isfinite(rows[numeric].to_numpy(float)).all():
            raise ValueError(f'Missing or nonfinite pilot scores/counts: {v}')
        n = rows[f'n_targets_{v}']
        if not ((n > 0) & (n == np.floor(n))).all():
            raise ValueError('Invalid target count')
        if (rows.groupby('receiver_idx')[f'n_targets_{v}'].nunique() != 1).any():
            raise ValueError('Target counts changed across conditions')
        if (rows.groupby('receiver_idx')[f'base_{v}'].nunique() != 1).any():
            raise ValueError('Baseline changed across conditions')
        if not np.allclose(rows[f'delta_{v}'], rows[f'patched_{v}'] - rows[f'base_{v}'], atol=1e-12, rtol=1e-12):
            raise ValueError('Pilot delta arithmetic mismatch')
        zero = rows.loc[rows.condition == 'zero']
        if (zero[f'delta_{v}'] != 0).any() or not np.array_equal(zero[f'base_{v}'], zero[f'patched_{v}']):
            raise ValueError('Zero condition does not define the baseline')
    if not np.array_equal(rows.n_targets_full, rows.n_targets_profile_only + rows.n_targets_behavior_only):
        raise ValueError('Pilot target views do not partition the targets')
    for prefix in ('base', 'patched'):
        combined = (rows[f'{prefix}_profile_only'] * rows.n_targets_profile_only
                    + rows[f'{prefix}_behavior_only'] * rows.n_targets_behavior_only) / rows.n_targets_full
        if not np.allclose(rows[f'{prefix}_full'], combined, rtol=1e-10, atol=1e-10):
            raise ValueError('Pilot loss views do not reconstruct the full loss')
    if not {'receiver_idx', 'condition', 'kind', 'n_tokens', 'n_edited', 'norm_l2_total'} <= set(balance):
        raise ValueError('Missing edit-balance fields')
    bkeys = list(zip(balance.receiver_idx, balance.condition))
    if len(bkeys) != len(set(bkeys)) or set(bkeys) != {(i, c) for i in expected.receiver_idx for c in conds if c != 'zero'}:
        raise ValueError('Incomplete or duplicate edit-balance grid')
    bref = expected.set_index('receiver_idx').loc[balance.receiver_idx]
    if balance.kind.tolist() != bref.kind.tolist():
        raise ValueError('Edit-balance identity mismatch')
    nums = balance.select_dtypes(include='number')
    if not np.isfinite(nums.to_numpy(float)).all() or (balance.n_edited < 0).any() or (balance.n_edited > balance.n_tokens).any() or (balance.norm_l2_total < 0).any():
        raise ValueError('Invalid edit-balance measurement')
    validate_validity(validity, len(expected))


def load_validated_run(run_dir, *, require_completion=True):
    root = Path(run_dir)
    if require_completion:
        done = json.loads((root / 'COMPLETED.json').read_text())
        if done.get('status') != 'complete' or done.get('contract') != 'corrected-pilot-v2':
            raise ValueError('Pilot did not complete')
        if set(done.get('files', {})) != set(FILES):
            raise ValueError('Pilot completion manifest is incomplete')
        for name in FILES:
            if done['files'][name] != file_sha256(root / name):
                raise ValueError(f'Pilot completion checksum mismatch: {name}')
    rows = pd.read_csv(root / FILES[0])
    man = json.loads((root / FILES[1]).read_text())
    validity = json.loads((root / FILES[2]).read_text())
    balance = pd.read_csv(root / FILES[3])
    validate_tables(rows, man, validity, balance)
    return rows, man, validity, balance


def complete_run(run_dir):
    root = Path(run_dir)
    if (root / 'COMPLETED.json').exists():
        raise ValueError('Refusing to overwrite a completed pilot')
    load_validated_run(root, require_completion=False)
    done = {'status': 'complete', 'contract': 'corrected-pilot-v2',
            'files': {name: file_sha256(root / name) for name in FILES}}
    temp = root / 'COMPLETED.partial'
    temp.write_text(json.dumps(done, indent=2, allow_nan=False) + '\n')
    temp.rename(root / 'COMPLETED.json')
