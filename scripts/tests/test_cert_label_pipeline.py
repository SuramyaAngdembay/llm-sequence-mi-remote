"""Regression cases for label reintroduction and silent pilot-row loss."""
import json
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'review'))
from cert_calendar_labels import ORIGIN, VERSION, file_sha256, label_table, relabel_frame
from cert_label_contracts import BUNDLE_VERSION, align_eval_label_state, corrected_cache, eval_index, load_stage_labels, validate_canonical
from cert_pilot_contracts import LIMITS, complete_run, load_validated_run, validate_tables
from eval_cert_recovery import score_views
from eval_r42_classes_complete import prepare_complete_cache, validate_bridge
from token_class_decomposition import CERT_CLASSES


def label_fixture():
    labels = label_table([{'user_id': 'BBS0039', 'start': '2010-08-12T01:00:00', 'end': '2010-08-12T23:00:00'}])
    raw = pd.DataFrame({'example_idx': [0, 1, 2], 'example_id': ['BBS0039:226', 'BBS0039:227', 'BEN0001:227'],
                        'user_id': ['BBS0039', 'BBS0039', 'BEN0001'], 'day_index': [226, 227, 227],
                        'split': ['eval', 'eval', 'val'], 'y': [1, 0, 0]})
    return raw, relabel_frame(raw, labels), labels


def class_cache(raw):
    out = raw.copy()
    for c in CERT_CLASSES:
        out[f'n_{c}'] = 1
        out[f'loss_sum_{c}'] = np.arange(1, len(out) + 1, dtype=float)
    out['n_tokens'] = 7
    out['n_targets'] = 6
    out['adapted_nll'] = out.loss_sum_DAY
    out['loss_sum_total'] = 6 * out.adapted_nll
    return out


def write_bundle(root):
    raw, canonical, labels = label_fixture()
    p = root / 'incident_calendar_labels.parquet'
    labels.to_parquet(p, index=False)
    p.with_suffix('.manifest.json').write_text(json.dumps({'label_version': VERSION, 'origin': str(ORIGIN.date()),
                                                          'label_sha256': file_sha256(p)}))
    (root / 'discovery_users_v1.txt').write_text('BBS0039\n')
    canonical.to_parquet(root / 'ranking_labels_v1.parquet', index=False)
    canonical.assign(preserve_for_sampling=canonical.y).to_parquet(root / 'sampling_labels_v1.parquet', index=False)
    man = {'schema': BUNDLE_VERSION, 'label_version': VERSION,
           'purposes': {'ranking_labels_v1.parquet': 'target', 'sampling_labels_v1.parquet': 'sampling'},
           'files': {p.name: file_sha256(p) for p in root.iterdir()}}
    (root / 'labels_manifest.json').write_text(json.dumps(man))
    return raw, canonical, labels


def test_completion_writer_roundtrip_uses_corrected_labels_and_canonical_indices(tmp_path):
    raw, meta, labels = label_fixture()
    scored = class_cache(raw)
    old, ext = scored.iloc[[0, 2]], scored.iloc[[1]].assign(example_idx=0)
    result = prepare_complete_cache(old, ext, meta, labels, raw[['example_idx', 'example_id']])
    result.to_parquet(tmp_path / 'complete.parquet', index=False)
    saved = pd.read_parquet(tmp_path / 'complete.parquet')
    assert saved.y.tolist() == [0, 1, 0]
    assert saved.y_original.tolist() == [1, 0, 0]
    assert saved.example_idx.tolist() == [0, 1, 2]
    assert saved.source_example_idx.tolist() == [0, 0, 2]
    assert set(saved.label_version) == {VERSION}
    assert saved.calendar_date.tolist() == ['2010-08-11', '2010-08-12', '2010-08-12']
    pd.testing.assert_frame_equal(old, scored.iloc[[0, 2]])


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'wrong_history', 'wrong_user', 'wrong_split', 'bad_partition', 'nan_count', 'nan_loss'])
def test_completion_rejects_corrupt_input(problem):
    raw, meta, labels = label_fixture()
    scored = class_cache(raw)
    old, ext = scored.iloc[[0, 2]].copy(), scored.iloc[[1]].copy()
    if problem == 'duplicate': old = pd.concat([old, old.iloc[[0]]])
    if problem == 'missing': old = old.iloc[:1]
    if problem == 'wrong_history': ext['y'] = 1
    if problem == 'wrong_user': ext['user_id'] = 'WRONG'
    if problem == 'wrong_split': ext['split'] = 'train'
    if problem == 'bad_partition': ext['loss_sum_total'] += 1
    if problem == 'nan_count': ext['n_targets'] = np.nan
    if problem == 'nan_loss': ext['loss_sum_total'] = np.nan
    with pytest.raises(ValueError):
        prepare_complete_cache(old, ext, meta, labels, raw[['example_idx', 'example_id']])


def test_metadata_cannot_reintroduce_a_historical_label():
    raw, meta, labels = label_fixture()
    meta['y'] = raw.y
    with pytest.raises(ValueError, match='disagrees'):
        validate_canonical(meta, labels)


def test_loading_historical_jsonl_cannot_overwrite_corrected_scores():
    raw, canonical, _ = label_fixture()
    assert align_eval_label_state(raw, canonical).y.tolist() == [0, 1, 0]
    assert align_eval_label_state(canonical, canonical).y.tolist() == [0, 1, 0]
    wrong = raw.copy(); wrong.loc[0, 'y'] = 0
    with pytest.raises(ValueError, match='provenance'):
        align_eval_label_state(wrong, canonical)


def test_eval_indices_come_from_jsonl_and_wrong_labels_fail(tmp_path):
    raw, meta, _ = label_fixture()
    path = tmp_path / 'eval.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in raw.iloc[::-1].to_dict('records')) + '\n')
    got = eval_index(path, meta)
    assert got.example_id.tolist() == raw.example_id.tolist()[::-1]
    assert got.example_idx.tolist() == [0, 1, 2]
    raw['label_version'] = VERSION
    path.write_text('\n'.join(json.dumps(r) for r in raw.to_dict('records')) + '\n')
    with pytest.raises(ValueError, match='labels disagree'):
        eval_index(path, meta)


def test_bundle_purpose_hash_and_target_checks(tmp_path):
    raw, canonical, _ = write_bundle(tmp_path)
    rank = tmp_path / 'ranking_labels_v1.parquet'
    sample = tmp_path / 'sampling_labels_v1.parquet'
    assert load_stage_labels(rank, raw, purpose='target').y.tolist() == [0, 1, 0]
    assert load_stage_labels(sample, raw, purpose='sampling').y.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match='purpose'):
        load_stage_labels(sample, raw, purpose='target')
    canonical['y'] = raw.y
    canonical.to_parquet(rank, index=False)
    with pytest.raises(ValueError, match='checksum'):
        load_stage_labels(rank, raw, purpose='target')
    man = json.loads((tmp_path / 'labels_manifest.json').read_text())
    man['files'][rank.name] = file_sha256(rank)
    (tmp_path / 'labels_manifest.json').write_text(json.dumps(man))
    with pytest.raises(ValueError, match='disagrees'):
        load_stage_labels(rank, raw, purpose='target')


def pilot_fixture():
    expected = [{'receiver_idx': i, 'receiver_id': f'{u}:{i}', 'user': u,
                 'kind': 'attack' if i % 2 == 0 else 'window_negative', 'pair': i // 2, 'y': int(i % 2 == 0)}
                for i, u in enumerate(['BBS0039', 'BBS0039', 'AAM0658', 'AAM0658'])]
    conds = ['zero', 'rpU_S', 'randI_S', 'randD_S', 'rpU_C', 'randI_C', 'rpO_S', 'orig_S', 'recon_S', 'rpU_S@0.5', 'rpU_C@0.5']
    man = {'contract': 'corrected-pilot-v2', 'label_version': VERSION, 'expected_receivers': expected,
           'conditions': conds, 'n_users': 2, 'n_pairs': 2}
    rows, balance = [], []
    for r in expected:
        for c in conds:
            d = (1 if r['user'] == 'BBS0039' else -1) if c == 'rpU_S' else 0
            row = dict(r, condition=c, label_version=VERSION)
            for v, n in [('full', 10), ('profile_only', 2), ('behavior_only', 8)]:
                row.update({f'base_{v}': 2., f'patched_{v}': 2. + d, f'delta_{v}': float(d), f'n_targets_{v}': n})
            rows.append(row)
            if c != 'zero':
                balance.append({'receiver_idx': r['receiver_idx'], 'condition': c, 'kind': r['kind'],
                                'n_tokens': 11, 'n_edited': 1, 'norm_l2_total': 1.})
    validity = {'token_rows_equal_tokenizer_length': True, 'misaligned': [], 'n_receivers': 4,
                'tokenizer_padding_side': 'right'} | {key: 0. for key in LIMITS}
    return pd.DataFrame(rows), man, validity, pd.DataFrame(balance)


def write_pilot(root):
    rows, man, val, balance = pilot_fixture()
    rows.to_csv(root / 'pilot1_rows.csv', index=False)
    (root / 'pilot1_manifest.json').write_text(json.dumps(man))
    (root / 'validity.json').write_text(json.dumps(val))
    balance.to_csv(root / 'edit_balance_rows.csv', index=False)
    return rows, man, val, balance


def test_pilot_completion_required_and_bound_to_bytes(tmp_path):
    write_pilot(tmp_path)
    with pytest.raises(FileNotFoundError): load_validated_run(tmp_path)
    complete_run(tmp_path)
    assert len(load_validated_run(tmp_path)[0]) == 44
    with (tmp_path / 'pilot1_rows.csv').open('a') as f: f.write('\n')
    with pytest.raises(ValueError, match='checksum'): load_validated_run(tmp_path)


@pytest.mark.parametrize('engineering,matched', [(False,True),(True,True),(False,False)])
def test_v3_complete_diagnostics_cannot_become_causal_results(tmp_path, engineering, matched):
    from intervention_audit import check_realized_controls
    rows, man, val, bal=write_pilot(tmp_path)
    keep=['zero','rpU_S','randI_S','randD_S']
    rows=rows[rows.condition.isin(keep)]; bal=bal[bal.condition.isin(keep)]
    man.update(contract='corrected-pilot-v3',conditions=keep,batch_size=1,full_logits_max_gib=0.,
               live_state_manifest_sha256='a'*64,engineering_only=engineering,realized_control_relative_tolerance=.05)
    parts=[]
    for r in rows.itertuples():
        for p in range(11):
            n=0. if r.condition=='zero' else 1.
            if not matched and r.condition=='randI_S': n*=2
            parts.append(dict(condition=r.condition,receiver_idx=r.receiver_idx,position=p,
                requested_l2=n,cast_l2=n,realized_l2=n,realization_error_l2=0.,
                requested_nonzero_coordinates=int(n>0),realized_nonzero_coordinates=int(n>0),requested_dot_realized=n*n))
    tok=pd.DataFrame(parts)
    report=check_realized_controls(tok)
    report.update(live_pre_edit_states_exact=True,checked_receiver_conditions=len(rows),engineering_only=engineering)
    tok.to_parquet(tmp_path/'realized_token_edits.parquet',index=False)
    (tmp_path/'implementation_validation.json').write_text(json.dumps(report))
    (tmp_path/'pilot1_manifest.json').write_text(json.dumps(man))
    rows.to_csv(tmp_path/'pilot1_rows.csv',index=False); bal.to_csv(tmp_path/'edit_balance_rows.csv',index=False)
    complete_run(tmp_path)
    load_validated_run(tmp_path,require_causal_validity=False)
    if engineering or not matched:
        with pytest.raises(ValueError,match='Diagnostic'): load_validated_run(tmp_path)
    else: load_validated_run(tmp_path)
    tok=tok.iloc[:-1]
    tok.to_parquet(tmp_path/'realized_token_edits.parquet',index=False)
    with pytest.raises(ValueError,match='grid'):
        load_validated_run(tmp_path,require_completion=False,require_causal_validity=False)


@pytest.mark.parametrize('problem', ['missing_row', 'missing_user', 'duplicate', 'nan', 'wrong_label', 'wrong_user',
                                  'failed_gate', 'nan_gate', 'bad_baseline', 'bad_count', 'missing_balance', 'wrong_plan'])
def test_pilot_rejects_invalid_results(problem):
    rows, man, val, balance = pilot_fixture()
    if problem == 'missing_row': rows = rows.drop(rows.index[-1])
    if problem == 'missing_user': rows = rows[rows.user != 'AAM0658']
    if problem == 'duplicate': rows = pd.concat([rows, rows.iloc[[1]]])
    if problem == 'nan': rows.loc[1, 'delta_behavior_only'] = np.nan
    if problem == 'wrong_label': rows.loc[1, 'y'] = 0
    if problem == 'wrong_user': rows.loc[1, 'user'] = 'OTHER'
    if problem == 'failed_gate': val['zero_edit_vs_unhooked_max_abs'] = .1
    if problem == 'nan_gate': val['zero_edit_vs_unhooked_max_abs'] = np.nan
    if problem == 'bad_baseline': rows.loc[1, 'base_full'] += 1
    if problem == 'bad_count': rows.loc[1, 'n_targets_behavior_only'] += 1
    if problem == 'missing_balance': balance = balance.iloc[:-1]
    if problem == 'wrong_plan': man['expected_receivers'][0]['y'] = 0
    with pytest.raises(ValueError): validate_tables(rows, man, val, balance)


def test_sampling_does_not_overwrite_targets(tmp_path):
    pytest.importorskip('torch')
    from train_delta_sae_frontier import override_labels, subsample_rows
    raw, canonical, _ = write_bundle(tmp_path)
    assert override_labels(raw.set_index('example_idx', drop=False), tmp_path / 'ranking_labels_v1.parquet').tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match='purpose'):
        override_labels(raw.set_index('example_idx', drop=False), tmp_path / 'sampling_labels_v1.parquet')
    # Preserve a declared sampling row independently of its target. This helper
    # does no eligibility inference; the typed bundle enforces policy upstream.
    meta = canonical.assign(preserve_for_sampling=[1, 0, 0])
    x, y, got = subsample_rows(np.eye(3), canonical.y.to_numpy(), meta, max_rows=1, seed=42)
    assert got.example_id.tolist() == ['BBS0039:226'] and y.tolist() == [0]


def test_presampling_and_dictionary_only_cpu_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    root = Path(__file__).resolve().parents[2]
    labels_dir = tmp_path / 'labels'
    labels_dir.mkdir()
    raw, canonical, _ = write_bundle(labels_dir)
    # Treat the positive user as confirmation: its true target must survive even
    # though none of its rows has positive-preservation priority.
    (labels_dir / 'discovery_users_v1.txt').write_text('BEN0001\n')
    canonical.assign(preserve_for_sampling=0).to_parquet(labels_dir / 'sampling_labels_v1.parquet', index=False)
    p = labels_dir / 'labels_manifest.json'
    man = json.loads(p.read_text())
    man['files'] = {name: file_sha256(labels_dir / name) for name in man['files']}
    p.write_text(json.dumps(man))
    extract = tmp_path / 'extract'
    (extract / 'layer_26').mkdir(parents=True)
    raw.to_parquet(extract / 'example_scores.parquet', index=False)
    chunk = extract / 'layer_26/chunk_00000.pt'
    torch.save({'delta': np.arange(12, dtype=np.float16).reshape(6, 2),
                'example_idx': np.repeat(np.arange(3), 2), 'position': np.tile([0, 1], 3)}, chunk)
    pd.DataFrame([{'layer': 26, 'rows': 6, 'path': str(chunk)}]).to_csv(extract / 'chunk_manifest.csv', index=False)
    (extract / 'extract_summary.json').write_text(json.dumps({'layers': [26], 'pool_unit': 'token'}))
    sampled = tmp_path / 'presampled'
    cmd = [sys.executable, str(root / 'scripts/review/presample_sae_rows.py'), '--extract-dir', str(extract),
           '--labels-override', str(labels_dir / 'sampling_labels_v1.parquet'), '--max-rows', '6',
           '--chunk-rows', '2', '--out-dir', str(sampled)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    scores = pd.read_parquet(sampled / 'example_scores.parquet')
    assert scores.y.tolist() == [0, 1, 0] and scores.preserve_for_sampling.tolist() == [0, 0, 0]
    assert len(list((sampled / 'layer_26').glob('chunk_*.pt'))) == 3
    assert (sampled / 'COMPLETED.json').is_file()
    config = tmp_path / 'config.yaml'
    config.write_text('sweep:\n  latent_multipliers: [2]\n  topk: [1]\n')
    trained = tmp_path / 'trained'
    cmd = [sys.executable, str(root / 'scripts/train_delta_sae_frontier.py'), '--config', str(config),
           '--extract-dir', str(sampled), '--out-dir', str(trained), '--device', 'cpu', '--epochs', '1',
           '--batch-size', '2', '--max-rows', '6', '--torch-seed', '42', '--dictionary-only']
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    output = trained / 'layer_26/m02_k01'
    assert (output / 'delta_sae_model.pt').is_file()
    assert (output / 'RANKING_NOT_COMPUTED.json').is_file()
    assert not (output / 'delta_sae_top_features.csv').exists()


def scenario_archive(path):
    data = b'dataset,user,scenario\n4.2,BBS0039,3\n4.2,AAM0658,1\n'
    with tarfile.open(path, 'w:bz2') as tar:
        member = tarfile.TarInfo('answers/insiders.csv')
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize('engineering,matched', [(False,False),(False,True),(True,False)])
def test_explicit_diagnostic_keeps_global_failure_and_existing_estimates(tmp_path,engineering,matched):
    from intervention_audit import check_realized_controls
    root=Path(__file__).resolve().parents[2]
    rows,man,val,bal=write_pilot(tmp_path)
    rows=pd.concat([rows,rows[rows.condition=='randI_C'].assign(condition='randD_C')],ignore_index=True)
    bal=pd.concat([bal,bal[bal.condition=='randI_C'].assign(condition='randD_C')],ignore_index=True)
    man['conditions'].append('randD_C')
    man.update(contract='corrected-pilot-v3',batch_size=1,full_logits_max_gib=0.,
        live_state_manifest_sha256='a'*64,engineering_only=engineering,realized_control_relative_tolerance=.05)
    parts=[]
    for r in rows.itertuples():
        for position in range(11):
            n=0. if r.condition=='zero' else 1.
            if not matched and r.receiver_idx==1 and r.condition=='randD_S' and position==2: n=1.1
            parts.append(dict(condition=r.condition,receiver_idx=r.receiver_idx,position=position,
                requested_l2=n,cast_l2=n,realized_l2=n,realization_error_l2=0.,
                requested_nonzero_coordinates=int(n>0),realized_nonzero_coordinates=int(n>0),requested_dot_realized=n*n))
    tokens=pd.DataFrame(parts)
    gate=check_realized_controls(tokens)
    gate.update(live_pre_edit_states_exact=True,checked_receiver_conditions=len(rows),engineering_only=engineering)
    tokens.to_parquet(tmp_path/'realized_token_edits.parquet',index=False)
    (tmp_path/'implementation_validation.json').write_text(json.dumps(gate))
    answers=tmp_path/'answers.tar.bz2'; scenario_archive(answers)
    man['answers_sha256']=file_sha256(answers)
    (tmp_path/'pilot1_manifest.json').write_text(json.dumps(man))
    rows.to_csv(tmp_path/'pilot1_rows.csv',index=False); bal.to_csv(tmp_path/'edit_balance_rows.csv',index=False)
    complete_run(tmp_path)
    common=[sys.executable,str(root/'scripts/review/analyze_corrected_pilot.py'),'--run-dir',str(tmp_path),
            '--answers',str(answers)]
    default=tmp_path/'default.json'
    result=subprocess.run(common+['--out',str(default)],capture_output=True,text=True)
    assert (result.returncode==0)==(matched and not engineering)
    out=tmp_path/'diagnostic.json'
    result=subprocess.run(common+['--out',str(out),'--implementation-diagnostic'],capture_output=True,text=True)
    if engineering:
        assert result.returncode!=0 and not out.exists()
    else:
        assert result.returncode==0,result.stderr
        diagnostic=json.loads(out.read_text())
        assert not diagnostic['scientific_claim_cleared']
        assert diagnostic['implementation_validation']['passed']==matched
        audit=diagnostic['implementation_dependency_audit']
        assert audit['rows_excluded']==0 and audit['original_global_gate_passed']==matched
        assert diagnostic['endpoints']['primary rpU_S - randI_S (attack)']['estimate']==0.
        if matched:
            normal=json.loads(default.read_text())
            for field in ('endpoints','secondary','conditions','heterogeneity'):
                assert diagnostic[field]==normal[field]
        # Even explicitly requested diagnostics must reject a modified score file.
        rows.iloc[:-1].to_csv(tmp_path/'pilot1_rows.csv',index=False)
        bad=tmp_path/'corrupt.json'
        result=subprocess.run(common+['--out',str(bad),'--implementation-diagnostic'],capture_output=True,text=True)
        assert result.returncode!=0 and not bad.exists()


def test_analyzer_cli_rejects_lost_control_and_handles_zero_dose(tmp_path):
    root = Path(__file__).resolve().parents[2]
    write_pilot(tmp_path)
    answers = tmp_path / 'answers.tar.bz2'
    scenario_archive(answers)
    p = tmp_path / 'pilot1_manifest.json'
    man = json.loads(p.read_text())
    man['answers_sha256'] = file_sha256(answers)
    p.write_text(json.dumps(man))
    complete_run(tmp_path)
    out = tmp_path / 'analysis.json'
    cmd = [sys.executable, str(root / 'scripts/review/analyze_corrected_pilot.py'),
           '--run-dir', str(tmp_path), '--answers', str(answers), '--out', str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    assert result['endpoints']['primary rpU_S - randI_S (attack)']['estimate'] == 0.
    assert result['secondary']['dose ratio mean(rpU_S@0.5) / mean(rpU_S)']['estimate'] is None
    rows = pd.read_csv(tmp_path / 'pilot1_rows.csv')
    rows = rows[~((rows.receiver_idx == 2) & (rows.condition == 'randI_S'))]
    rows.to_csv(tmp_path / 'pilot1_rows.csv', index=False)
    out.unlink()
    bad = subprocess.run(cmd, capture_output=True, text=True)
    assert bad.returncode != 0 and not out.exists()


def test_complete_cache_cli_publishes_only_valid_labeled_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    raw, _, labels = label_fixture()
    raw.loc[0, ['example_id', 'user_id']] = ['AAM0658:226', 'AAM0658']
    extra = dict(raw.iloc[0], example_idx=3, example_id='AAM0658:299', user_id='AAM0658', day_index=299, y=1)
    raw = pd.concat([raw, pd.DataFrame([extra])], ignore_index=True)
    labels = pd.concat([labels, label_table([{'user_id': 'AAM0658', 'start': '2010-10-23T01:00:00', 'end': '2010-10-23T23:00:00'}])], ignore_index=True)
    meta = relabel_frame(raw, labels)
    scored = class_cache(raw)
    old, ext = scored.iloc[[0, 2, 3]], scored.iloc[[1]].assign(example_idx=0)
    bridge = old.iloc[:2]
    paths = {}
    for name, frame in [('old-cache', old), ('extension-cache', ext), ('bridge-cache', bridge), ('metadata', meta), ('labels', labels)]:
        paths[name] = tmp_path / (name + '.parquet')
        frame.to_parquet(paths[name], index=False)
    for name, frame in [('eval-jsonl', raw), ('bridge-jsonl', raw.iloc[[0, 2]]), ('extension-jsonl', raw.iloc[[1]])]:
        paths[name] = tmp_path / (name + '.jsonl')
        paths[name].write_text('\n'.join(json.dumps(r) for r in frame.to_dict('records')) + '\n')
    paths['answers'] = tmp_path / 'answers.tar.bz2'
    scenario_archive(paths['answers'])
    paths['labels'].with_suffix('.manifest.json').write_text(json.dumps({
        'dataset': '4.2', 'label_version': VERSION, 'origin': str(ORIGIN.date()),
        'label_sha256': file_sha256(paths['labels']), 'archive_sha256': file_sha256(paths['answers'])}))
    template = {k: 'fixture' for k in ['adapter_digest', 'tokenizer_digest', 'model_name_or_path', 'max_seq_len',
                                      'length_sorted', 'with_base', 'classes', 'views', 'scoring_convention', 'gpu', 'torch']}
    template.update(batch_size=1, schema='cert')
    for name, n in [('old-manifest', 3), ('bridge-manifest', 2), ('extension-manifest', 1)]:
        paths[name] = tmp_path / (name + '.json')
        paths[name].write_text(json.dumps(dict(template, n_examples=n)))
    paths['bridge-gate'] = tmp_path / 'bridge_gate.json'
    paths['bridge-gate'].write_text(json.dumps({'bit_exact': True}))
    paths['input-manifest'] = tmp_path / 'inputs.json'
    inp = {key: file_sha256(paths[name]) for key, name in [('class_cache_sha256', 'old-cache'),
           ('original_eval_sha256', 'eval-jsonl'), ('bridge_sha256', 'bridge-jsonl'), ('extension_sha256', 'extension-jsonl')]}
    inp.update(bridge_rows=2, extension_rows=1)
    paths['input-manifest'].write_text(json.dumps(inp))
    out = tmp_path / 'complete'
    cmd = [sys.executable, str(root / 'scripts/review/eval_r42_classes_complete.py'), '--out-dir', str(out)]
    for key, path in paths.items(): cmd += ['--' + key, str(path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    saved = pd.read_parquet(out / 'class_scores_complete_calendar_v1.parquet')
    assert saved.y.tolist() == [0, 1, 0, 1] and saved.y_original.tolist() == [1, 0, 0, 1]
    assert saved.example_idx.tolist() == [0, 1, 2, 3] and (out / 'COMPLETED.json').exists()
    # A true-looking boolean cannot authorize merging different bridge scores.
    bad = bridge.copy(); bad['adapted_nll'] += 1
    bad.to_parquet(paths['bridge-cache'], index=False)
    cmd[cmd.index(str(out))] = str(tmp_path / 'bad-output')
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and not (tmp_path / 'bad-output').exists()


def test_merge_publishes_corrected_complete_cache_and_refuses_bad_chunk(tmp_path):
    torch = pytest.importorskip('torch')
    root = Path(__file__).resolve().parents[2]
    labdir = tmp_path / 'labels'; labdir.mkdir()
    raw, canonical, _ = write_bundle(labdir)
    raw = raw.assign(n_tokens=2, base_nll=2., adapted_nll=1., delta_nll=-1.)
    ev = tmp_path / 'eval.jsonl'
    ev.write_text('\n'.join(json.dumps(r) for r in raw.to_dict('records')) + '\n')
    shards = []
    for cid, take in enumerate(([0, 1], [2])):
        shard = tmp_path / f'shard_{cid}'; shards.append(shard)
        (shard / 'layer_26').mkdir(parents=True)
        frame = raw.iloc[take]
        frame.to_parquet(shard / 'example_scores.parquet', index=False)
        chunk = shard / f'layer_26/chunk_{cid:05d}.pt'
        torch.save({'delta': np.ones((2*len(take), 2), dtype=np.float16), 'example_idx': np.repeat(take, 2),
                    'position': np.tile([0, 1], len(take))}, chunk)
        pd.DataFrame([{'layer': 26, 'chunk_id': cid, 'path': str(chunk), 'rows': 2*len(take),
                       'd_model': 2, 'unit': 'token'}]).to_csv(shard / 'chunk_manifest.csv', index=False)
        summary = dict(n_examples=len(take), start_index=take[0], chunk_examples=2, layers=[26], pool_unit='token',
                       adapter_dir='fixture', model_name_or_path='fixture', base_via_disable_adapter=True,
                       batch_size=1, max_batch_tokens=10, ce_seq_chunk=2, split='eval', batch_sizes_used={'1': len(take)})
        (shard / 'extract_summary.json').write_text(json.dumps(summary))
    out = tmp_path / 'merged'
    cmd = [sys.executable, str(root / 'scripts/review/merge_delta_shards.py'), '--shards', *map(str, shards),
           '--labels', str(labdir / 'ranking_labels_v1.parquet'), '--eval-jsonl', str(ev), '--out-dir', str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    scores = pd.read_parquet(out / 'example_scores.parquet')
    assert scores.y.tolist() == [0, 1, 0] and scores.y_original.tolist() == [1, 0, 0]
    assert (out / 'COMPLETED.json').is_file()
    assert (out / 'layer_26/chunk_00000.pt').is_symlink()
    report = json.loads((out / 'merge_report.json').read_text())
    assert report['rows'] == 6 and report['examples'] == 3
    # A later malformed chunk must not leave a publishable partial directory.
    chunk = shards[1] / 'layer_26/chunk_00001.pt'
    torch.save({'delta': np.ones((2, 2), dtype=np.float16), 'example_idx': np.array([2, 2]),
                'position': np.array([0, 0])}, chunk)
    cmd[-1] = str(tmp_path / 'bad_merge')
    bad = subprocess.run(cmd, capture_output=True, text=True)
    assert bad.returncode != 0 and not (tmp_path / 'bad_merge').exists()
