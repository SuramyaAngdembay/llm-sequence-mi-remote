"""Authoritative CERT incident-window labels in the LC-DAL calendar clock."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp('2009-12-28')
VERSION = 'cert-master-window-calendar-v1'
EVENT_VERSION = 'cert-named-user-observable-day-v1'
ARCHIVE_MD5 = 'c0c96eaa06915e417504f932756f58d8'
DT_FORMAT = '%m/%d/%Y %H:%M:%S'


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def window_dates(start: datetime, end: datetime) -> list[str]:
    if end <= start or start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError('Require increasing naive dataset-calendar timestamps')
    d = start.replace(hour=0, minute=0, second=0, microsecond=0)
    dates = []
    while d < end:
        if max(d, start) < min(d + timedelta(days=1), end):
            dates.append(d.date().isoformat())
        d += timedelta(days=1)
    return dates


def calendar_dates(day_index) -> pd.DatetimeIndex:
    x = np.asarray(day_index, dtype=float)
    if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
        raise ValueError('day_index must contain finite integers')
    return pd.DatetimeIndex(ORIGIN + pd.to_timedelta(x.astype(np.int64), unit='D'))


def validate_session_calendar(df: pd.DataFrame) -> None:
    if 'starttime' not in df:
        raise ValueError('Session timestamps are required to validate the day origin')
    actual = pd.DatetimeIndex(pd.to_datetime(df.starttime, unit='s', utc=True)).tz_localize(None).normalize()
    expected = calendar_dates(df.day_index)
    if actual.isna().any() or not actual.equals(expected):
        raise ValueError('Session day_index does not match starttime under the verified 2009-12-28 origin')


def read_official_windows(archive: Path, dataset: str):
    if dataset not in {'4.2', '6.2'}:
        raise ValueError('This audited pipeline supports CERT 4.2 and 6.2')
    if hashlib.md5(Path(archive).read_bytes()).hexdigest() != ARCHIVE_MD5:
        raise ValueError('Official answer archive checksum mismatch')
    windows, sources, discrepancies, event_rows = [], [], [], []
    with tarfile.open(archive, 'r:bz2') as tar:
        members = [m for m in tar.getmembers() if m.isfile()]

        def read_member(name):
            matches = [m for m in members if Path(m.name).name == name]
            if len(matches) != 1:
                raise ValueError(f'Expected one archive member for {name}, found {len(matches)}')
            raw = tar.extractfile(matches[0]).read()
            sources.append({'member': matches[0].name, 'sha256': hashlib.sha256(raw).hexdigest()})
            return raw.decode('utf-8-sig')

        master = list(csv.DictReader(io.StringIO(read_member('insiders.csv'))))
        for r in master:
            if r['dataset'] != dataset:
                continue
            events = []
            for line in csv.reader(io.StringIO(read_member(r['details']))):
                if len(line) >= 4 and line[3] == r['user']:
                    events.append(datetime.strptime(line[2], DT_FORMAT))
            if not events:
                raise ValueError(f'No named-user incident observables for {r["user"]}')
            repaired = False
            try:
                start = datetime.strptime(r['start'], DT_FORMAT)
            except ValueError:
                if (dataset, r['user'], r['start']) != ('6.2', 'CDE1846', '/21/2011 11:43:39'):
                    raise ValueError(f'Unapproved malformed master start: {r}')
                start = min(events)
                if start != datetime(2011, 2, 21, 11, 43, 39):
                    raise ValueError('CDE1846 detail evidence does not match the documented repair')
                repaired = True
            end = datetime.strptime(r['end'], DT_FORMAT)
            window_dates(start, end)  # Validate even if no session occurs in the window.
            windows.append({'user_id': r['user'], 'start': start.isoformat(), 'end': end.isoformat(),
                            'raw_start': r['start'], 'raw_end': r['end'], 'details': r['details'],
                            'scenario': r['scenario'], 'start_repaired': repaired})
            outside = [t for t in events if t < start or t > end]
            if outside:
                discrepancies.append({'user_id': r['user'], 'master_start': start.isoformat(),
                    'master_end': end.isoformat(), 'detail_start': min(events).isoformat(),
                    'detail_end': max(events).isoformat(), 'outside_event_count': len(outside),
                    'outside_dates': sorted({t.date().isoformat() for t in outside})})
            event_rows.extend({'user_id': r['user'], 'calendar_date': d, 'y': 1}
                              for d in sorted({t.date().isoformat() for t in events}))
    if len(windows) != (70 if dataset == '4.2' else 5):
        raise ValueError('Unexpected number of official incident windows')
    return windows, event_rows, sources, discrepancies


def label_table(windows) -> pd.DataFrame:
    rows = []
    for r in windows:
        for d in window_dates(datetime.fromisoformat(r['start']), datetime.fromisoformat(r['end'])):
            rows.append({'user_id': r['user_id'], 'calendar_date': d, 'y': 1})
    return prepare_table(rows, VERSION)


def prepare_table(rows, version) -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates(['user_id', 'calendar_date']).sort_values(['user_id', 'calendar_date'])
    df['day_index'] = (pd.to_datetime(df.calendar_date) - ORIGIN).dt.days.astype('int64')
    df['label_version'] = version
    return df.reset_index(drop=True)


def validate_label_table(labels: pd.DataFrame, version=VERSION) -> None:
    required = {'user_id', 'calendar_date', 'day_index', 'y', 'label_version'}
    if not required <= set(labels):
        raise ValueError('Unverified integer-origin labels refused; build_cert_calendar_labels.py must generate calendar labels first')
    if labels.empty or labels[['user_id', 'calendar_date']].isna().any().any():
        raise ValueError('Empty or null label keys')
    if labels.duplicated(['user_id', 'calendar_date']).any():
        raise ValueError('Duplicate canonical label keys')
    if set(labels.label_version) != {version} or set(labels.y) != {1}:
        raise ValueError('Unexpected label version or non-positive sparse label row')
    if list(calendar_dates(labels.day_index).strftime('%Y-%m-%d')) != labels.calendar_date.tolist():
        raise ValueError('Label dates and day indices disagree')


def load_label_artifact(path: Path, version=VERSION) -> pd.DataFrame:
    path = Path(path)
    manifest_path = path.with_suffix('.manifest.json')
    if not manifest_path.exists():
        raise ValueError(f'Missing verified calendar-label manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if manifest['label_version'] != version or manifest['origin'] != str(ORIGIN.date()):
        raise ValueError('Label manifest version/origin mismatch')
    if manifest['label_sha256'] != file_sha256(path):
        raise ValueError('Label artifact hash does not match its manifest')
    df = pd.read_parquet(path)
    validate_label_table(df, version)
    df.attrs['manifest'] = manifest
    return df


def relabel_frame(df: pd.DataFrame, labels: pd.DataFrame, *, version=VERSION) -> pd.DataFrame:
    validate_label_table(labels, version)
    out = df.copy()
    if out.user_id.isna().any():
        raise ValueError('Null user key')
    dates = calendar_dates(out.day_index).strftime('%Y-%m-%d')
    if 'calendar_date' in out and out.calendar_date.astype(str).tolist() != list(dates):
        raise ValueError('Example calendar_date and day_index disagree')
    if 'example_id' in out:
        expected = out.user_id.astype(str) + ':' + out.day_index.astype('int64').astype(str)
        if not expected.equals(out.example_id.astype(str)):
            raise ValueError('example_id disagrees with user/day identity')
    if 'y_original' in out or 'label_version' in out:
        raise ValueError('Refusing to relabel an already-versioned artifact')
    if 'y' in out:
        if not out.y.isin([0, 1]).all():
            raise ValueError('Input y must be binary')
        out = out.rename(columns={'y': 'y_original'})
    out['calendar_date'] = dates
    lookup = labels.set_index(['user_id', 'calendar_date']).y
    keys = pd.MultiIndex.from_arrays([out.user_id.astype(str), dates])
    out['y'] = lookup.reindex(keys, fill_value=0).to_numpy(dtype=np.int8)
    out['label_version'] = version
    return out
