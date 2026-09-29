#!/usr/bin/env python3
"""Stage B.C label inputs (docs/review/STAGE_B_PROTOCOL_2026-09-28.md sections 3-5).

Writes, into a new directory:
  discovery_users_v1.txt / confirmation_users_v1.txt  historical 30/30 plus the 10 newly positive insiders split
                                                       5/5 by sha256("cert-calendar-v1-split-2026-09-28|user") order
  negative_users_v1.txt                                held-out benign (val) users: the only negative-row source
  sampling_labels_v1.parquet                           corrected targets plus separate preserve_for_sampling flag
  ranking_labels_v1.parquet                            corrected calendar targets and complete identity/provenance
  labels_manifest.json                                 counts and hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import VERSION, file_sha256, load_label_artifact
from cert_label_contracts import BUNDLE_VERSION, IDENTITY, TARGET, corrected_cache, load_stage_labels

SALT = 'cert-calendar-v1-split-2026-09-28'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-index', type=Path, required=True, help='token_deltas/example_scores.parquet')
    ap.add_argument('--metadata', type=Path, required=True, help='recovery labels/r42/example_metadata.parquet')
    ap.add_argument('--labels', type=Path, required=True, help='verified incident_calendar_labels.parquet')
    ap.add_argument('--discovery', type=Path, required=True); ap.add_argument('--confirmation', type=Path, required=True)
    ap.add_argument('--insiders', type=Path, required=True, help='official r4.2 insider users, one per line')
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f'refusing to overwrite {a.out_dir}')
    source = pd.read_parquet(a.score_index)
    meta = pd.read_parquet(a.metadata)
    idx = corrected_cache(source, meta, load_label_artifact(a.labels))
    if set(idx.example_id) != set(meta.loc[meta.split.isin(['val', 'eval']), 'example_id']):
        raise ValueError('Stage B requires the complete unseen score index')
    idx = idx[['example_idx'] + IDENTITY + TARGET]
    if idx.example_idx.tolist() != list(range(len(idx))):
        raise ValueError('Stage B requires canonical sequential evaluation indices')
    insiders = {l.strip() for l in a.insiders.read_text().splitlines() if l.strip()}
    disc0 = {l.strip() for l in a.discovery.read_text().splitlines() if l.strip()}
    conf0 = {l.strip() for l in a.confirmation.read_text().splitlines() if l.strip()}
    pos_users = set(idx.user_id[idx.y == 1])
    assert pos_users == insiders & set(idx.user_id), 'corrected positive users are not the official insiders'
    new = sorted(pos_users - disc0 - conf0, key=lambda u: hashlib.sha256(f'{SALT}|{u}'.encode()).hexdigest())
    assert len(new) == 10 and not (disc0 & conf0)
    disc, conf = disc0 | set(new[:5]), conf0 | set(new[5:])
    neg = sorted(set(idx.user_id[idx.split == 'val']))
    assert not (set(neg) & insiders) and len(neg) == 79
    a.out_dir.parent.mkdir(parents=True, exist_ok=True)
    # A temporary directory is never a completed label bundle.
    stage = Path(tempfile.mkdtemp(prefix='.stage-labels-', dir=a.out_dir.parent))
    for name, users in [('discovery_users_v1.txt', disc), ('confirmation_users_v1.txt', conf), ('negative_users_v1.txt', neg)]:
        (stage / name).write_text('\n'.join(sorted(users)) + '\n')
    samp = idx.assign(preserve_for_sampling=((idx.y == 1) & idx.user_id.isin(disc)).astype('int8'))
    rank = idx.copy()
    samp.to_parquet(stage / 'sampling_labels_v1.parquet', index=False)
    rank.to_parquet(stage / 'ranking_labels_v1.parquet', index=False)
    shutil.copyfile(a.labels, stage / 'incident_calendar_labels.parquet')
    shutil.copyfile(a.labels.with_suffix('.manifest.json'), stage / 'incident_calendar_labels.manifest.json')
    man = {'schema': BUNDLE_VERSION, 'label_version': VERSION,
           'purposes': {'sampling_labels_v1.parquet': 'sampling', 'ranking_labels_v1.parquet': 'target'},
           'sampling_policy': 'preserve discovery-positive rows; y remains the calendar target',
           'salt': SALT, 'new_insiders_hash_order': new, 'new_to_discovery': new[:5], 'new_to_confirmation': new[5:],
           'n_discovery': len(disc), 'n_confirmation': len(conf), 'n_negative_users': len(neg),
           'sampling_preserved_days': int(samp.preserve_for_sampling.sum()), 'ranking_positive_days': int(rank.y.sum()),
           'discovery_positive_days': int(rank.y[idx.user_id.isin(disc)].sum()),
           'confirmation_positive_days': int(rank.y[idx.user_id.isin(conf)].sum()),
           'files': {p.name: file_sha256(p) for p in sorted(stage.iterdir())},
           'score_index_sha256': sha(a.score_index), 'metadata_sha256': sha(a.metadata)}
    (stage / 'labels_manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    load_stage_labels(stage / 'sampling_labels_v1.parquet', source, purpose='sampling')
    load_stage_labels(stage / 'ranking_labels_v1.parquet', source, purpose='target')
    if a.out_dir.exists():
        raise ValueError('Output appeared while building labels')
    stage.rename(a.out_dir)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
