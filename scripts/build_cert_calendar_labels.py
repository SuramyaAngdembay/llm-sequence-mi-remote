#!/usr/bin/env python3
"""Build new, provenance-bound CERT calendar labels; never overwrite legacy labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cert_calendar_labels import (EVENT_VERSION, ORIGIN, VERSION, file_sha256,
    label_table, prepare_table, read_official_windows, relabel_frame, validate_label_table)
import pandas as pd


def build(archive: Path, dataset: str, out_dir: Path, metadata: Path | None = None):
    if out_dir.exists():
        raise FileExistsError(f'Recovery outputs must use a new directory: {out_dir}')
    windows, events, members, discrepancies = read_official_windows(archive, dataset)
    primary = label_table(windows)
    validate_label_table(primary)
    event_table = prepare_table(events, EVENT_VERSION)
    tables = [('incident_calendar_labels', primary, VERSION),
              ('observable_calendar_labels', event_table, EVENT_VERSION)]
    overlay = None
    if metadata is not None:
        original = pd.read_parquet(metadata)
        if original.duplicated(['user_id', 'day_index']).any():
            raise ValueError('Full metadata must have unique user-days')
        overlay = relabel_frame(original, primary)
        insiders = {r['user_id'] for r in windows}
        if set(overlay.loc[overlay.split.isin(['train', 'val']), 'user_id']) & insiders:
            raise ValueError('Official insider in saved adapter train/validation split')
        expected = (1362, 70) if dataset == '4.2' else (73, 5)
        observed = (int(overlay.y.sum()), int(overlay.loc[overlay.y > 0, 'user_id'].nunique()))
        if observed != expected:
            raise ValueError(f'Observed-domain gate failed: {observed}, expected {expected}')
    out_dir.mkdir(parents=True)
    report = {'dataset': dataset, 'origin': str(ORIGIN.date()), 'label_version': VERSION,
              'archive_sha256': file_sha256(archive), 'source_members': members,
              'windows': windows, 'master_detail_discrepancies': discrepancies,
              'interval_convention': '[start,end) intersected with [midnight,next midnight)',
              'date_semantics': 'recorded dataset calendar; no timezone conversion',
              'official_insider_users': sorted({r['user_id'] for r in windows})}
    for name, table, version in tables:
        path = out_dir / (name + '.parquet')
        table.to_parquet(path, index=False)
        manifest = dict(report, label_version=version, label_sha256=file_sha256(path), label_rows=len(table))
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if overlay is not None:
        overlay.to_parquet(out_dir / 'example_metadata.parquet', index=False)
        changes = overlay.loc[overlay.y != overlay.y_original]
        changes.to_csv(out_dir / 'changed_labels.csv', index=False)
        report['observed'] = {'rows': len(overlay), 'old_positive_days': int(overlay.y_original.sum()),
            'positive_days': int(overlay.y.sum()), 'positive_users': int(overlay.loc[overlay.y > 0, 'user_id'].nunique()),
            'changed_rows': len(changes), 'positive_to_negative': int(((overlay.y_original == 1) & (overlay.y == 0)).sum()),
            'negative_to_positive': int(((overlay.y_original == 0) & (overlay.y == 1)).sum()),
            'input_metadata_sha256': file_sha256(metadata),
            'output_metadata_sha256': file_sha256(out_dir / 'example_metadata.parquet'),
            'original_splits_preserved': original.split.equals(overlay.split)}
    (out_dir / 'label_build_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--answers-archive', type=Path, required=True)
    ap.add_argument('--dataset', choices=['4.2', '6.2'], required=True)
    ap.add_argument('--metadata', type=Path)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    result = build(args.answers_archive, args.dataset, args.out_dir, args.metadata)
    print(json.dumps(result.get('observed', {'label_version': VERSION}), indent=2))
