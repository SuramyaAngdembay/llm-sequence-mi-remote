import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot_endpoint_audit import endpoint_implementation_audit


def fixture():
    conds=['zero','rpU_S','randI_S','randD_S','rpU_C','randI_C','randD_C',
           'orig_S','recon_S','rpO_S','rpU_S@0.5','rpU_C@0.5']
    rows=pd.DataFrame([dict(receiver_idx=i,kind=k,condition=c) for i,k in [(1,'attack'),(2,'window_negative')]
                       for c in conds])
    tokens=pd.DataFrame([dict(receiver_idx=r.receiver_idx,condition=r.condition,position=j,
                             realized_l2=0. if j==0 else 1.) for r in rows.itertuples() for j in range(2)])
    return rows,tokens


def test_window_negative_dictionary_failure_does_not_enter_attack_equation():
    rows,tokens=fixture()
    tokens.loc[(tokens.receiver_idx==2)&(tokens.condition=='randD_S')&(tokens.position==1),'realized_l2']=1.1
    a=endpoint_implementation_audit(rows,tokens)
    assert not a['original_global_gate_passed'] and not a['scientific_claim_cleared']
    assert len(a['failures'])==1 and a['rows_excluded']==0
    assert all(v['local_matching_prerequisites_pass'] for v in a['endpoints'].values())
    assert all(not v['failed_random_control_rows_used'] for v in a['endpoints'].values())


@pytest.mark.parametrize('condition,key',[('randD_S','secondary_dictionary_control'),('randI_S','primary')])
def test_attack_failure_follows_its_endpoint_dependency(condition,key):
    rows,tokens=fixture()
    tokens.loc[(tokens.receiver_idx==1)&(tokens.condition==condition)&(tokens.position==1),'realized_l2']=1.1
    a=endpoint_implementation_audit(rows,tokens)
    assert not a['endpoints'][key]['local_matching_prerequisites_pass']
    assert a['endpoints'][key]['failed_random_control_rows_used']
    assert a['endpoints']['co_primary']['local_matching_prerequisites_pass']


def test_missing_token_and_ambiguous_identity_rejected():
    rows,tokens=fixture()
    with pytest.raises(ValueError,match='grid'):
        endpoint_implementation_audit(rows,tokens.iloc[1:][~((tokens.iloc[1:].condition=='randI_S')&(tokens.iloc[1:].position==1))])
    bad=pd.concat([rows,pd.DataFrame([dict(receiver_idx=1,kind='window_negative',condition='zero')])])
    with pytest.raises(ValueError,match='Ambiguous'):
        endpoint_implementation_audit(bad,tokens)


def test_diagnostic_does_not_claim_original_protocol_passed_even_with_local_success():
    rows,tokens=fixture()
    a=endpoint_implementation_audit(rows,tokens)
    assert a['original_global_gate_passed']
    assert not a['original_protocol_reclassified'] and not a['scientific_claim_cleared']
