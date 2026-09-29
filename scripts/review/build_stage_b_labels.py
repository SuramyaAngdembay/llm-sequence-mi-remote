#!/usr/bin/env python3
"""Stage B.C label inputs (docs/review/STAGE_B_PROTOCOL_2026-09-28.md sections 3-5).

Writes, into a new directory:
  discovery_users_v1.txt / confirmation_users_v1.txt  historical 30/30 plus the 10 newly positive insiders split
                                                       5/5 by sha256("cert-calendar-v1-split-2026-09-28|user") order
  negative_users_v1.txt                                held-out benign (val) users: the only negative-row source
  sampling_labels_v1.parquet                           example_idx, example_id, y = corrected positive AND discovery
                                                       user (the dictionary's positive-preservation flag)
  ranking_labels_v1.parquet                            example_idx, example_id, y = corrected calendar label
  labels_manifest.json                                 counts and hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SALT = 'cert-calendar-v1-split-2026-09-28'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-index', type=Path, required=True, help='token_deltas/example_scores.parquet')
    ap.add_argument('--metadata', type=Path, required=True, help='recovery labels/r42/example_metadata.parquet')
    ap.add_argument('--discovery', type=Path, required=True); ap.add_argument('--confirmation', type=Path, required=True)
    ap.add_argument('--insiders', type=Path, required=True, help='official r4.2 insider users, one per line')
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f'refusing to overwrite {a.out_dir}')
    idx = pd.read_parquet(a.score_index)[['example_idx', 'example_id', 'user_id', 'split']]
    meta = pd.read_parquet(a.metadata).set_index('example_id')
    if not set(idx.example_id) <= set(meta.index):
        raise SystemExit('score index has examples outside the metadata')
    idx['y_corr'] = meta.loc[idx.example_id, 'y'].to_numpy()
    insiders = {l.strip() for l in a.insiders.read_text().splitlines() if l.strip()}
    disc0 = {l.strip() for l in a.discovery.read_text().splitlines() if l.strip()}
    conf0 = {l.strip() for l in a.confirmation.read_text().splitlines() if l.strip()}
    pos_users = set(idx.user_id[idx.y_corr == 1])
    assert pos_users == insiders & set(idx.user_id), 'corrected positive users are not the official insiders'
    new = sorted(pos_users - disc0 - conf0, key=lambda u: hashlib.sha256(f'{SALT}|{u}'.encode()).hexdigest())
    assert len(new) == 10 and not (disc0 & conf0)
    disc, conf = disc0 | set(new[:5]), conf0 | set(new[5:])
    neg = sorted(set(idx.user_id[idx.split == 'val']))
    assert not (set(neg) & insiders) and len(neg) == 79
    a.out_dir.mkdir(parents=True)
    for name, users in [('discovery_users_v1.txt', disc), ('confirmation_users_v1.txt', conf), ('negative_users_v1.txt', neg)]:
        (a.out_dir / name).write_text('\n'.join(sorted(users)) + '\n')
    samp = idx.assign(y=((idx.y_corr == 1) & idx.user_id.isin(disc)).astype('int64'))[['example_idx', 'example_id', 'y']]
    rank = idx.assign(y=idx.y_corr.astype('int64'))[['example_idx', 'example_id', 'y']]
    samp.to_parquet(a.out_dir / 'sampling_labels_v1.parquet', index=False)
    rank.to_parquet(a.out_dir / 'ranking_labels_v1.parquet', index=False)
    man = {'salt': SALT, 'new_insiders_hash_order': new, 'new_to_discovery': new[:5], 'new_to_confirmation': new[5:],
           'n_discovery': len(disc), 'n_confirmation': len(conf), 'n_negative_users': len(neg),
           'sampling_positive_days': int(samp.y.sum()), 'ranking_positive_days': int(rank.y.sum()),
           'discovery_positive_days': int(rank.y[idx.user_id.isin(disc)].sum()),
           'confirmation_positive_days': int(rank.y[idx.user_id.isin(conf)].sum()),
           'files': {p.name: sha(p) for p in sorted(a.out_dir.iterdir())},
           'score_index_sha256': sha(a.score_index), 'metadata_sha256': sha(a.metadata)}
    (a.out_dir / 'labels_manifest.json').write_text(json.dumps(man, indent=1) + '\n')
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
