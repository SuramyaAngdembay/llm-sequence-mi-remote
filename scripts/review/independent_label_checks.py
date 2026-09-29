#!/usr/bin/env python3
"""Independent check of the CERT calendar-label recovery (review of commit c31efcf).

Written without importing the recovery modules (cert_calendar_labels.py, eval_cert_recovery.py) so that
agreement is not circular. Date enumeration uses first/last-date arithmetic rather than the recovery's
day-by-day loop. Checks:
  1. official answer archive: MD5, SHA-256, members, the one malformed master timestamp;
  2. session clock: day index vs epoch day, recorded start hour vs epoch time of day, on every session row;
  3. event-level clock: official detail events fall inside the named user's session intervals under
     the UTC-naive reading, compared with shifted readings;
  4. complete positive key sets (user, calendar date) vs the recovery's label tables and metadata overlay;
  5. change ledger (old CTMC-joined labels vs corrected) and the named-user observable-day target.

  python independent_label_checks.py --recovery ROOT --sessions DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FMT = '%m/%d/%Y %H:%M:%S'
EPOCH_DAY_ORIGIN = (date(2009, 12, 28) - date(1970, 1, 1)).days       # 14606
EXPECTED_MD5 = 'c0c96eaa06915e417504f932756f58d8'                       # published on the KiltHub record


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def covered_dates(start: datetime, end: datetime) -> list[date]:
    """Calendar dates whose [00:00, 24:00) overlaps the half-open window [start, end)."""
    assert end > start
    last = end.date() if end.time() != datetime.min.time() else end.date() - timedelta(days=1)
    n = (last - start.date()).days
    return [start.date() + timedelta(days=k) for k in range(n + 1)]


def utc_naive_epoch(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


def read_archive(path: Path):
    raw = path.read_bytes()
    out = {'archive_md5': hashlib.md5(raw).hexdigest(), 'archive_sha256': sha256(raw), 'bytes': len(raw)}
    out['md5_matches_published'] = out['archive_md5'] == EXPECTED_MD5
    members = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:bz2') as tar:
        for m in tar.getmembers():
            if m.isfile():
                members[m.name] = tar.extractfile(m).read()
    return out, members


def member(members, basename):
    hits = [k for k in members if Path(k).name == basename]
    assert len(hits) == 1, (basename, hits)
    return hits[0], members[hits[0]].decode('utf-8-sig')


def official(members, dataset):
    name, text = member(members, 'insiders.csv')
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r['dataset'] == dataset]
    windows, malformed, events, discrepancies = [], [], defaultdict(list), []
    for r in rows:
        dname, dtext = member(members, r['details'])
        ev = []
        for line in csv.reader(io.StringIO(dtext)):
            if len(line) >= 4 and line[3] == r['user']:
                ev.append((line[0], datetime.strptime(line[2], FMT)))
        events[r['user']] = ev
        parsed = {}
        for k in ('start', 'end'):
            try:
                parsed[k] = datetime.strptime(r[k], FMT)
            except ValueError:
                malformed.append({'user': r['user'], 'field': k, 'raw': r[k]})
                parsed[k] = None
        if parsed['start'] is None:                                  # documented repair: earliest named-user event
            parsed['start'] = min(t for _, t in ev)
        times = [t for _, t in ev]
        outside = [t for t in times if not (parsed['start'] <= t <= parsed['end'])]
        if outside:
            discrepancies.append({'user': r['user'], 'n_outside': len(outside),
                                  'outside_dates': sorted({t.date().isoformat() for t in outside})})
        windows.append({'user': r['user'], 'scenario': r['scenario'], 'start': parsed['start'], 'end': parsed['end'],
                        'raw_start': r['start'], 'raw_end': r['end'], 'details_member': dname,
                        'details_sha256': sha256(members[dname])})
    return windows, malformed, events, discrepancies


def load_sessions(path: Path, user_map: Path):
    cols = ['starttime', 'endtime', 'user', 'day', 'pc', 'isafterhour', 'isweekend', 'ses_start', 'ses_end', 'insider']
    df = pd.read_csv(path, names=cols, header=None)
    um = pd.read_csv(user_map)
    df['user_id'] = df.user.map(dict(zip(um.user_code, um.user_id)))
    assert df.user_id.notna().all()
    return df


def clock_checks(s: pd.DataFrame):
    epoch_day = np.floor(s.starttime.to_numpy() / 86400).astype(np.int64)
    tod_h = (s.starttime.to_numpy() - epoch_day * 86400) / 3600
    diff_min = (tod_h - s.ses_start.to_numpy()) * 60
    hours = Counter(np.floor(s.ses_start).astype(int))
    return {'rows': int(len(s)),
            'origin_epoch_day_minus_day_index': {str(date(1970, 1, 1) + timedelta(days=int(k))): int(v)
                                                 for k, v in Counter(epoch_day - s.day.to_numpy()).items()},
            'utc_time_of_day_minus_ses_start_minutes': {'max_abs': float(np.abs(diff_min).max()),
                                                        'n_over_1min': int((np.abs(diff_min) > 1).sum())},
            'ses_start_hour_histogram': {int(h): int(n) for h, n in sorted(hours.items())},
            'starttime_nonintegral': int((s.starttime % 1 != 0).sum())}


def event_containment(s: pd.DataFrame, events: dict):
    """Share of official named-user events inside one of that user's session intervals, by clock offset."""
    by_user = {u: g[['starttime', 'endtime']].to_numpy() for u, g in s.groupby('user_id')}
    res = {}
    for off_h in [-8, -7, -6, -5, -4, -1, 0, 1, 4, 5, 6, 7, 8]:
        inside = total = 0; kinds = Counter(); kinds_in = Counter()
        for u, ev in events.items():
            iv = by_user.get(u)
            for kind, t in ev:
                e = utc_naive_epoch(t) + off_h * 3600
                hit = iv is not None and bool(((iv[:, 0] <= e) & (e <= iv[:, 1])).any())
                total += 1; inside += hit; kinds[kind] += 1; kinds_in[kind] += hit
        res[str(off_h)] = {'events': total, 'inside_a_session': inside, 'share': inside / total,
                           'share_by_kind': {k: kinds_in[k] / kinds[k] for k in sorted(kinds)}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--sessions', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    R = a.recovery
    arch, members = read_archive(R / 'source/answers.tar.bz2')
    report = {'archive': arch, 'datasets': {}}
    for tag, ds, sess, umap in [('r42', '4.2', 'r42_session_slim.csv.gz', 'sessionr4.2_user_map.csv'),
                                ('r62', '6.2', 'r62_session_slim.csv.gz', 'sessionr6.2_user_map.csv')]:
        windows, malformed, events, discrepancies = official(members, ds)
        s = load_sessions(a.sessions / sess, a.sessions / umap)
        rep = {'n_windows': len(windows), 'malformed_master_fields': malformed,
               'master_detail_discrepancies': discrepancies, 'clock': clock_checks(s)}
        rep['event_containment_by_offset_hours'] = event_containment(s, events)

        # complete positive key sets
        window_keys = {(w['user'], d.isoformat()) for w in windows for d in covered_dates(w['start'], w['end'])}
        observable_keys = {(u, t.date().isoformat()) for u, ev in events.items() for _, t in ev}
        meta = pd.read_parquet(R / f'labels/{tag}/example_metadata.parquet')
        orig = pd.read_parquet(R / f'inputs/{tag}/metadata.parquet')
        cal = (pd.Timestamp('2009-12-28') + pd.to_timedelta(meta.day_index, unit='D')).dt.strftime('%Y-%m-%d')
        mkeys = list(zip(meta.user_id.astype(str), cal))
        mine_y = np.array([k in window_keys for k in mkeys], dtype=int)
        sess_days = set(zip(s.user_id, s.day))
        rep['metadata'] = {
            'rows': int(len(meta)), 'unique_user_day': int(meta[['user_id', 'day_index']].drop_duplicates().shape[0]),
            'metadata_keys_equal_session_user_days': set(zip(meta.user_id, meta.day_index)) == sess_days,
            'original_metadata_rows_identical_order': bool(orig.example_id.equals(meta.example_id)),
            'original_y_preserved': bool((orig.y.to_numpy() == meta.y_original.to_numpy()).all()),
            'original_split_preserved': bool(orig.split.equals(meta.split)),
            'recovery_calendar_date_matches_mine': bool((meta.calendar_date.astype(str).to_numpy() == cal.to_numpy()).all()),
            'my_positive_days': int(mine_y.sum()),
            'my_positive_users': int(meta.user_id[mine_y == 1].nunique()),
            'recovery_y_equals_mine_every_row': bool((meta.y.to_numpy() == mine_y).all()),
            'changed_rows': int((mine_y != meta.y_original.to_numpy()).sum()),
            'positive_to_negative': int(((meta.y_original.to_numpy() == 1) & (mine_y == 0)).sum()),
            'negative_to_positive': int(((meta.y_original.to_numpy() == 0) & (mine_y == 1)).sum()),
            'window_days_total_calendar': len(window_keys),
            'window_days_without_any_session': len({k for k in window_keys} - set(mkeys)),
            'positive_users_by_split': meta.assign(m=mine_y).query('m == 1').groupby('split').user_id.nunique().to_dict(),
            'official_insiders_in_train_or_val': sorted(set(meta.user_id[meta.split.isin(['train', 'val'])]) & {w['user'] for w in windows}),
        }
        # recovery's sparse label tables vs my enumeration (complete key sets, not totals)
        lab = pd.read_parquet(R / f'labels/{tag}/incident_calendar_labels.parquet')
        obs = pd.read_parquet(R / f'labels/{tag}/observable_calendar_labels.parquet')
        rk = set(zip(lab.user_id, lab.calendar_date.astype(str))); ok = set(zip(obs.user_id, obs.calendar_date.astype(str)))
        rep['label_tables'] = {'incident_rows': len(lab), 'incident_equal_mine': rk == window_keys,
                               'incident_only_recovery': sorted(rk - window_keys)[:10], 'incident_only_mine': sorted(window_keys - rk)[:10],
                               'observable_rows': len(ok), 'observable_equal_mine': ok == observable_keys}
        unseen = meta.split.isin(['val', 'eval']).to_numpy()
        obs_y = np.array([k in observable_keys for k in mkeys], dtype=int)
        rep['observable_target'] = {'observed_positive_days_all': int(obs_y.sum()), 'observed_positive_days_unseen': int(obs_y[unseen].sum()),
                                    'observed_positive_users_unseen': int(meta.user_id[(obs_y == 1) & unseen].nunique()),
                                    'differs_from_window_target_unseen': int((obs_y[unseen] != mine_y[unseen]).sum())}
        # the old CTMC-joined labels as a key set, to confirm what changed is a clock difference
        ledger = pd.read_csv(R / f'labels/{tag}/changed_labels.csv')
        rep['recovery_change_ledger_rows'] = int(len(ledger))
        rep['recovery_change_ledger_matches_mine'] = set(ledger.example_id) == set(meta.example_id[mine_y != meta.y_original.to_numpy()])
        report['datasets'][tag] = rep
        print(tag, json.dumps({k: rep['metadata'][k] for k in ['my_positive_days', 'my_positive_users', 'changed_rows',
                                                              'positive_to_negative', 'negative_to_positive',
                                                              'recovery_y_equals_mine_every_row']}), flush=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
