import sys
import json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
pytest.importorskip('torch')
from physical_control_policy import match_dictionary_control, realized_bfloat16


def arrays():
    h=np.array([[64.,0.],[0.,0.]],dtype=np.float32)
    ref=np.array([[0.,.1],[0.,0.]],dtype=np.float32)
    old=np.array([[.1,0.],[0.,0.]],dtype=np.float32)
    dirs=np.array([[0.,1.]],dtype=np.float32)
    return h,ref,old,dirs


def test_replaces_only_failed_draw_deterministically_and_preserves_zero_support():
    a=arrays(); kw=dict(seed=42,receiver=3,condition='randD_S')
    result,log=match_dictionary_control(*a,**kw)
    assert json.loads(json.dumps(log))==log
    again,log2=match_dictionary_control(*a,**kw)
    np.testing.assert_array_equal(result,again); assert log==log2
    assert log['changed_positions']==[0]
    np.testing.assert_array_equal(result[1],a[2][1])
    np.testing.assert_array_equal(realized_bfloat16(a[0],result),realized_bfloat16(a[0],a[1]))
    unchanged,record=match_dictionary_control(a[0],a[1],result,a[3],**kw)
    np.testing.assert_array_equal(unchanged,result); assert record['changed_positions']==[]


def test_finite_candidate_budget_fails_without_weakening_threshold():
    h,ref,old,dirs=arrays()
    with pytest.raises(ValueError,match='No feasible'):
        match_dictionary_control(h,ref,old,np.array([[1.,0.]],np.float32),seed=42,receiver=3,condition='randD_S')
    with pytest.raises(ValueError,match='fixes'):
        match_dictionary_control(h,ref,old,dirs,seed=42,receiver=3,condition='randD_S',tolerance=.1)


def test_rejects_vanished_reference_and_changed_requested_norm():
    h,ref,old,dirs=arrays()
    with pytest.raises(ValueError,match='vanished'):
        match_dictionary_control(h,old,old,dirs,seed=42,receiver=3,condition='randD_S')
    with pytest.raises(ValueError,match='not norm'):
        match_dictionary_control(h,ref,old*2,dirs,seed=42,receiver=3,condition='randD_S')
