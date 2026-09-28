"""Regression tests for the date-origin failure and provenance gates."""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import (ORIGIN, VERSION, calendar_dates, file_sha256,
    label_table, load_label_artifact, relabel_frame, validate_label_table,
    validate_session_calendar, window_dates)
from build_session_jsonl import build_examples
from build_session_jsonl_fast import build_examples_fast
from eval_metrics_core import dedup_by_id
from eval_cert_recovery import attach_metadata, bootstrap_fold_table, evaluate_population, score_views, strict_dedup


def labels(user='AAM0658', start='2010-10-23T01:34:19', end='2010-10-29T05:23:28'):
    return label_table([{'user_id': user, 'start': start, 'end': end}])


def test_early_morning_is_calendar_day_not_previous_ctmc_day():
    lab = labels()
    assert lab.calendar_date.tolist() == [f'2010-10-{d:02d}' for d in range(23, 30)]
    assert lab.day_index.tolist() == list(range(299, 306))
    frame = pd.DataFrame({'user_id': ['AAM0658'] * 2, 'day_index': [298, 305],
                          'example_id': ['AAM0658:298', 'AAM0658:305'], 'y': [1, 0], 'split': ['eval'] * 2})
    out = relabel_frame(frame, lab)
    assert out.y.tolist() == [0, 1]
    assert out.y_original.tolist() == [1, 0]
    assert out.split.equals(frame.split)
    assert frame.y.tolist() == [1, 0]  # Input remains untouched.


def test_half_open_midnight_boundary():
    assert window_dates(datetime(2010, 1, 2, 23), datetime(2010, 1, 3)) == ['2010-01-02']
    assert window_dates(datetime(2010, 1, 2), datetime(2010, 1, 3, 0, 0, 1)) == ['2010-01-02', '2010-01-03']
    with pytest.raises(ValueError):
        window_dates(datetime(2010, 1, 2), datetime(2010, 1, 2))


@pytest.mark.parametrize('days', [[227.1], [np.nan], [np.inf]])
def test_invalid_indices_are_rejected(days):
    with pytest.raises(ValueError):
        calendar_dates(days)


def test_timestamp_origin_gate():
    df = pd.DataFrame({'day_index': [227], 'starttime': [pd.Timestamp('2010-08-12T07:30:00Z').timestamp()]})
    validate_session_calendar(df)
    df['day_index'] = 222
    with pytest.raises(ValueError, match='origin'):
        validate_session_calendar(df)


def test_legacy_labels_are_rejected_even_with_plausible_integers():
    with pytest.raises(ValueError, match='integer-origin'):
        validate_label_table(pd.DataFrame({'user_id': ['BBS0039'], 'day_index': [222], 'y': [1]}))


def test_manifest_hash_is_required_and_checked(tmp_path):
    p = tmp_path / 'labels.parquet'
    labels().to_parquet(p, index=False)
    with pytest.raises(ValueError, match='manifest'):
        load_label_artifact(p)
    manifest = {'label_version': VERSION, 'origin': str(ORIGIN.date()), 'label_sha256': file_sha256(p)}
    p.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    assert len(load_label_artifact(p)) == 7
    manifest['label_sha256'] = 'wrong'
    p.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hash'):
        load_label_artifact(p)


def test_identity_and_double_relabel_gates():
    frame = pd.DataFrame({'user_id': ['AAM0658'], 'day_index': [299], 'example_id': ['AAM0658:294'], 'y': [0]})
    with pytest.raises(ValueError, match='identity'):
        relabel_frame(frame, labels())
    frame['example_id'] = 'AAM0658:299'
    out = relabel_frame(frame, labels())
    with pytest.raises(ValueError, match='already-versioned'):
        relabel_frame(out, labels())


def test_both_builders_use_calendar_labels_and_preserve_split_and_text():
    df = pd.DataFrame({'user_id': ['BBS0039', 'BBS0039', 'NORMAL'],
        'day_index': [222, 227, 227], 'sessionid': [1, 2, 3],
        'starttime': [pd.Timestamp(d, tz='UTC').timestamp() for d in ['2010-08-07 08:00', '2010-08-12 08:00', '2010-08-12 09:00']],
        'week': [31, 32, 32], 'n_usb': [0, 1, 0], 'file_n-to_usb1': [0, 1, 0]})
    lab = labels('BBS0039', '2010-08-12T07:00:00', '2010-08-12T23:00:00')
    slow, ms = build_examples(df, lab, 0.1, 24)
    fast, mf = build_examples_fast(df, lab, 0.1, 24)
    assert slow == fast
    pd.testing.assert_frame_equal(ms, mf)
    assert [r['y'] for r in slow] == [0, 1, 0]
    assert [r['split'] for r in slow[:2]] == ['eval', 'eval']
    assert [r['calendar_date'] for r in slow] == ['2010-08-07','2010-08-12','2010-08-12']
    assert all(r['label_version']==VERSION for r in slow)
    assert all('file_n-to_usb1' not in r['text'] for r in slow)  # No incidental serializer change.


def test_duplicate_score_difference_is_not_silently_tolerated():
    with pytest.raises(ValueError, match='disagree'):
        dedup_by_id(['x', 'x'], {'score': np.array([1.0, 1.000000001])})


def test_recovery_folds_match_hand_calculated_user_ranking(tmp_path):
    # Two insiders; their largest scores are on negative days. Each fold must
    # keep those days and exactly the same two negative-only users.
    df = pd.DataFrame({'user_id': ['m1','m1','m2','m2','b1','b2'],
                       'y': [1,0,1,0,0,0]})
    s = np.array([.1,.9,.2,.8,.4,.5])
    rows, ff = evaluate_population(df, {'full':s,'same':s.copy()},'toy','all','y',tmp_path)
    assert all(r['user_roc']==1 for r in rows)
    assert all(r['within_user_roc']==0 for r in rows)
    assert all(r['heldout_user_rank_midpoint']==1 for r in ff)
    assert all(r['n_rows']==4 and r['n_users']==3 for r in ff)
    summaries, contrasts = bootstrap_fold_table(pd.DataFrame(ff),draws=100)
    assert all(r['estimate']==0 and r['ci_lo']==0 and r['ci_hi']==0 for r in contrasts)


def test_nonfinite_cache_fails_instead_of_silent_drop():
    d = pd.DataFrame({'adapted_nll':[np.nan], 'base_nll':[1.], 'delta_nll':[np.nan]})
    with pytest.raises(ValueError,match='Nonfinite'):
        score_views(d,'nll')


def test_cross_cache_split_conflict_is_rejected():
    m = pd.DataFrame({'example_id':['x:1'],'user_id':['x'],'day_index':[1],
                      'split':['eval'],'y_original':[0],'y':[1]})
    d = m.rename(columns={'y_original':'y_old'}).drop(columns='y').rename(columns={'y_old':'y'})
    d['split'] = 'train'
    with pytest.raises(ValueError,match='split'):
        attach_metadata(d,m)


def test_class_partition_and_target_counts_are_enforced():
    from token_class_decomposition import CERT_CLASSES
    d = pd.DataFrame({f'n_{c}':[1] for c in CERT_CLASSES})
    for c in CERT_CLASSES:d[f'loss_sum_{c}']=2.
    d['n_targets']=6;d['n_tokens']=7;d['loss_sum_total']=12.;d['adapted_nll']=2.
    views, checks = score_views(d,'classes')
    assert all(v[0]==2 for v in views.values())
    d['n_targets']=7
    with pytest.raises(ValueError,match='partition'):
        score_views(d,'classes')


def test_jsonl_relabel_preserves_model_input_and_original_exposure(tmp_path):
    from relabel_cert_jsonl import rewrite
    lab = labels()
    lp = tmp_path/'labels.parquet';lab.to_parquet(lp,index=False)
    lp.with_suffix('.manifest.json').write_text(json.dumps({'label_version':VERSION,
        'origin':str(ORIGIN.date()),'label_sha256':file_sha256(lp)}))
    original={'example_id':'AAM0658:305','user_id':'AAM0658','day_index':305,
              'split':'eval','y':0,'text':'DAY week=43\nPSY O=2\nSES idx=0','context':{'week':43}}
    src=tmp_path/'old.jsonl';src.write_text(json.dumps(original)+'\n')
    meta=relabel_frame(pd.DataFrame([{k:v for k,v in original.items() if k not in ['text','context']}]),lab)
    mp=tmp_path/'metadata.parquet';meta.to_parquet(mp,index=False)
    dest=tmp_path/'corrected.jsonl'
    result=rewrite(src,dest,lp,mp)
    output=json.loads(dest.read_text())
    assert output['y']==1 and output['y_original']==0
    assert output['text']==original['text'] and output['split']==original['split']
    assert json.loads(src.read_text())==original
    assert result['changed_labels']==1
    with pytest.raises(FileExistsError):rewrite(src,dest,lp,mp)


def test_independent_metric_arithmetic_includes_ties():
    from audit_cert_recovery_results import independent_metrics
    # One positive/negative tie at the top; the other negative is below it.
    result=independent_metrics([1,0,0],[2.,2.,1.])
    assert result['day_roc']==0.75
    assert result['day_ap']==0.5


def test_unrecognized_answer_archive_is_rejected(tmp_path):
    from cert_calendar_labels import read_official_windows
    archive=tmp_path/'answers.tar.bz2';archive.write_bytes(b'not the audited source')
    with pytest.raises(ValueError,match='checksum mismatch'):
        read_official_windows(archive,'6.2')
