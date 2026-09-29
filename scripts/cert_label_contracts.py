"""CPU-only identity and provenance checks for corrected CERT artifacts.

`y` always means the incident-calendar target. Sampling policy has its own
column; it must never be written into `y` in a score cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cert_calendar_labels import VERSION, calendar_dates, file_sha256, load_label_artifact, relabel_frame

IDENTITY = ['example_id', 'user_id', 'day_index', 'split']
TARGET = ['y_original', 'calendar_date', 'y', 'label_version']
BUNDLE_VERSION = 'cert-label-bundle-v2'


def unique_identity(frame):
    if not set(IDENTITY) <= set(frame) or frame.empty:
        raise ValueError('Missing or empty example identity')
    if frame[IDENTITY].isna().any().any() or frame.example_id.duplicated().any():
        raise ValueError('Null or duplicate example identity')
    dates = calendar_dates(frame.day_index).strftime('%Y-%m-%d')
    ids = frame.user_id.astype(str) + ':' + frame.day_index.astype('int64').astype(str)
    if ids.tolist() != frame.example_id.tolist():
        raise ValueError('example_id disagrees with user/day identity')
    if 'calendar_date' in frame and list(dates) != frame.calendar_date.tolist():
        raise ValueError('Calendar date disagrees with day_index')


def validate_canonical(metadata, labels):
    unique_identity(metadata)
    if not set(TARGET) <= set(metadata) or set(metadata.label_version) != {VERSION}:
        raise ValueError('Canonical metadata requires preserved labels and calendar version')
    original = metadata.drop(columns=['y', 'label_version', 'calendar_date']).rename(columns={'y_original': 'y'})
    rebuilt = relabel_frame(original, labels)
    if not np.array_equal(metadata.y.to_numpy(), rebuilt.y.to_numpy()):
        raise ValueError('Canonical metadata disagrees with verified incident labels')


def corrected_cache(cache, metadata, labels):
    """Relabel a historical cache after verifying identity and historical labels."""
    validate_canonical(metadata, labels)
    unique_identity(cache)
    if not set(cache.example_id) <= set(metadata.example_id):
        raise ValueError('Cache has unknown example IDs')
    ref = metadata.set_index('example_id').loc[cache.example_id]
    for col in IDENTITY[1:]:
        if cache[col].tolist() != ref[col].tolist():
            raise ValueError(f'Cache disagrees with canonical {col}')
    if cache.y.tolist() != ref.y_original.tolist():
        raise ValueError('Cache labels disagree with preserved original labels')
    return relabel_frame(cache, labels)


def eval_index(path, metadata):
    """Validate each JSONL identity and label; use its actual line number."""
    records = []
    with Path(path).open() as stream:
        for i, line in enumerate(stream):
            r = json.loads(line)
            records.append({k: r[k] for k in IDENTITY + ['y']} | {'example_idx': i,
                           'label_version': r.get('label_version')})
    frame = pd.DataFrame(records)
    unique_identity(frame)
    ref = metadata.set_index('example_id').loc[frame.example_id]
    if set(frame.example_id) != set(metadata.loc[metadata.split.isin(['val', 'eval']), 'example_id']):
        raise ValueError('Evaluation JSONL must cover exactly the canonical unseen domain')
    for col in IDENTITY[1:]:
        if frame[col].tolist() != ref[col].tolist():
            raise ValueError(f'Evaluation JSONL disagrees with canonical {col}')
    if frame.label_version.isna().all():
        expected = ref.y_original
    elif set(frame.label_version) == {VERSION}:
        expected = ref.y
    else:
        raise ValueError('Mixed or unknown JSONL label versions')
    if frame.y.tolist() != expected.tolist():
        raise ValueError('Evaluation JSONL labels disagree with its declared version')
    return frame[['example_idx', 'example_id']]


def load_stage_labels(path, score_index, *, purpose):
    """Read a typed label bundle and bind it to the score index's identities.

    The bundle includes a checksum-verified canonical sparse label artifact so
    changing both a label and its file checksum cannot bypass the target check.
    """
    path = Path(path)
    man = json.loads((path.parent / 'labels_manifest.json').read_text())
    if man.get('schema') != BUNDLE_VERSION or man.get('label_version') != VERSION:
        raise ValueError('Legacy/untyped label bundle refused; rebuild Stage B labels')
    if man.get('purposes', {}).get(path.name) != purpose:
        raise ValueError('Label artifact purpose mismatch')
    for name, digest in man['files'].items():
        if Path(name).name != name or file_sha256(path.parent / name) != digest:
            raise ValueError(f'Label bundle checksum mismatch: {name}')
    if path.name not in man['files']:
        raise ValueError('Label artifact is absent from bundle manifest')
    frame = pd.read_parquet(path)
    validate_canonical(frame, load_label_artifact(path.parent / 'incident_calendar_labels.parquet'))
    if 'example_idx' not in frame or frame.example_idx.duplicated().any():
        raise ValueError('Missing or duplicate canonical evaluation indices')
    indices = frame.example_idx.to_numpy(float)
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
        raise ValueError('Evaluation indices must be finite integers')
    score = score_index.reset_index(drop=True)
    if score.example_idx.duplicated().any() or set(score.example_idx) != set(frame.example_idx):
        raise ValueError('Label bundle must cover exactly the score index')
    out = frame.set_index('example_idx').loc[score.example_idx].reset_index()
    for col in IDENTITY:
        if col in score and out[col].tolist() != score[col].tolist():
            raise ValueError(f'Label bundle disagrees with score index {col}')
    if 'y' in score:
        expected_y = out.y_original
        if 'label_version' in score:
            if set(score.label_version) != {VERSION}:
                raise ValueError('Unknown score index label version')
            expected_y = out.y
            if 'y_original' not in score or score.y_original.tolist() != out.y_original.tolist():
                raise ValueError('Score index lost its original label provenance')
        if score.y.tolist() != expected_y.tolist():
            raise ValueError('Score index labels disagree with their declared version')
    if purpose == 'sampling':
        if 'preserve_for_sampling' not in out or not out.preserve_for_sampling.isin([0, 1]).all():
            raise ValueError('Invalid sampling flag')
        discovery = set((path.parent / 'discovery_users_v1.txt').read_text().splitlines())
        expected = ((out.y == 1) & out.user_id.isin(discovery)).astype(int)
        if not np.array_equal(out.preserve_for_sampling, expected):
            raise ValueError('Sampling flags disagree with the frozen discovery policy')
    return out


def align_eval_label_state(meta, scores):
    """Prevent a historical JSONL from overwriting corrected score-cache targets."""
    if 'label_version' not in scores:
        return meta
    unique_identity(scores)
    unique_identity(meta)
    if set(scores.label_version) != {VERSION} or not set(TARGET) <= set(scores):
        raise ValueError('Incomplete or unknown score-cache label provenance')
    for col in IDENTITY:
        if scores[col].tolist() != meta[col].tolist():
            raise ValueError(f'JSONL/score-cache metadata mismatch: {col}')
    expected = scores.y if 'label_version' in meta else scores.y_original
    if 'label_version' in meta and set(meta.label_version) != {VERSION}:
        raise ValueError('Unknown JSONL label version')
    if meta.y.tolist() != expected.tolist():
        raise ValueError('JSONL labels do not match score-cache provenance')
    out = meta.drop(columns=[c for c in TARGET if c != 'y' and c in meta]).copy()
    out['y'] = scores.y.to_numpy()
    return out
