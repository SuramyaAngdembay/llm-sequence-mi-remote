"""Prepared CPU ranking must reproduce the original feature evaluation exactly."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sae_core import TopKSAE
from reselect_token_sae_features import load_prepared_rows
from test_cert_label_pipeline import write_bundle


def test_preparation_and_ranking_cli_preserve_features_exactly(tmp_path):
    root=Path(__file__).resolve().parents[2]
    labels=tmp_path/'labels'; labels.mkdir()
    raw,canonical,_=write_bundle(labels)
    negatives=tmp_path/'negative.txt'; negatives.write_text('BEN0001\n')
    extract=tmp_path/'extract'; (extract/'layer_26').mkdir(parents=True)
    raw.to_parquet(extract/'example_scores.parquet',index=False)
    (extract/'COMPLETED.json').write_text('{"status":"complete"}')
    rng=np.random.default_rng(8)
    for chunk in range(2):
        idx=np.repeat(np.arange(3),40); pos=np.tile(np.arange(40)+chunk*40,3)
        torch.save({'delta':rng.normal(size=(120,4)).astype(np.float16),'example_idx':idx,'position':pos},
                   extract/'layer_26'/f'chunk_{chunk:05d}.pt')
    source=tmp_path/'source'; cfg=source/'layer_26/m02_k02'; cfg.mkdir(parents=True)
    torch.manual_seed(3); model=TopKSAE(4,8,2)
    torch.save({'state_dict':model.state_dict(),'d_in':4,'d_latent':8,'k':2,
                'x_mean':np.array([[.1,.2,-.3,.4]],np.float32),'x_std':np.array([[.3,.5,1.2,2.]],np.float32)},
               cfg/'delta_sae_model.pt')
    common=[sys.executable,str(root/'scripts/reselect_token_sae_features.py'),'--extract-dir',str(extract),
            '--data-dir',str(tmp_path),'--frontier-dir',str(source),'--layer','26','--latent-mult','2','--k','2',
            '--discovery-user-file',str(labels/'discovery_users_v1.txt'),'--negative-user-file',str(negatives),
            '--labels-override',str(labels/'ranking_labels_v1.parquet'),'--device','cpu','--batch-size','17',
            '--min-positive-users','1','--seed','42']
    direct=tmp_path/'direct'; prep=tmp_path/'prepared'; replay=tmp_path/'replayed'
    def run(args): subprocess.run(common+args,check=True,capture_output=True,text=True)
    run(['--out-frontier-dir',str(direct)])
    run(['--out-frontier-dir',str(tmp_path/'unused'),'--prepare-only-dir',str(prep)])
    assert not (tmp_path/'unused').exists()
    run(['--out-frontier-dir',str(replay),'--prepared-rows-dir',str(prep)])
    rel=Path('layer_26/m02_k02')
    pd.testing.assert_frame_equal(pd.read_csv(direct/rel/'delta_sae_top_features.csv'),
                                  pd.read_csv(replay/rel/'delta_sae_top_features.csv'),check_exact=True)
    a=json.loads((direct/rel/'reselect_summary.json').read_text())
    b=json.loads((replay/rel/'reselect_summary.json').read_text())
    for key in ('feature_sets','row_stats','eval_stats','positive_rows_by_discovery_user','feature_set_criteria'):
        assert a[key]==b[key]
    done=json.loads((prep/'COMPLETED.json').read_text())
    with pytest.raises(ValueError,match='contract'): load_prepared_rows(prep,{'changed':True})
    with (prep/'y.npy').open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError,match='checksum'): load_prepared_rows(prep,done['binding'])
