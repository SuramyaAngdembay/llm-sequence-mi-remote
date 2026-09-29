import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
torch=pytest.importorskip('torch')
from intervention_audit import apply_token_shift, realized_token_stats, check_realized_controls


def test_bfloat16_rounding_is_measured_after_addition():
    h=torch.tensor([[[1., 1.], [1., 1.]]],dtype=torch.bfloat16)
    shift=np.array([[0.001,0.001],[0.1,0.1]],dtype=np.float32)
    seen=[]
    def observe(i,b,q,c,a): seen.append(realized_token_stats(b,q,c,a))
    got=apply_token_shift(h,[shift],torch.ones((1,2)),strict=True,observer=observe)
    assert torch.equal(got,h+torch.tensor(shift,dtype=torch.bfloat16))
    assert seen[0]['requested_l2'][0]>0 and seen[0]['realized_l2'][0]==0
    assert seen[0]['cast_l2'][0]>0 and seen[0]['realized_l2'][1]>0


def test_zero_and_padding():
    h=torch.randn((1,3,4),dtype=torch.bfloat16)
    q=np.zeros((2,4),np.float32)
    assert torch.equal(apply_token_shift(h,[q],torch.tensor([[1,1,0]]),strict=True),h)
    with pytest.raises(ValueError): apply_token_shift(h,[q],torch.tensor([[0,1,1]]),strict=True)
    with pytest.raises(ValueError): apply_token_shift(h,[q[:1]],torch.tensor([[1,1,0]]),strict=True)
    q[0,0]=np.nan
    with pytest.raises(ValueError): apply_token_shift(h,[q],torch.tensor([[1,1,0]]),strict=True)


def test_realized_matching_gate_detects_support_and_dose_failure():
    rows=[]
    for c,vals in [('rpU_S',[0.,1.,2.]),('randI_S',[0.,1.,2.]),('randD_S',[0.,1.,2.])]:
        rows.extend(dict(condition=c,receiver_idx=3,position=i,realized_l2=v) for i,v in enumerate(vals))
    df=pd.DataFrame(rows)
    assert check_realized_controls(df)['passed']
    df.loc[(df.condition=='randI_S')&(df.position==0),'realized_l2']=.1
    df.loc[(df.condition=='randD_S')&(df.position==2),'realized_l2']=3.
    r=check_realized_controls(df)
    assert not r['passed'] and r['comparisons']['randI_S']['support_mismatches']==1
    assert r['comparisons']['randD_S']['norm_mismatch_tokens']==1


def test_control_gate_rejects_missing_or_nonfinite_controls():
    df=pd.DataFrame([dict(condition=c,receiver_idx=1,position=0,realized_l2=1.)
                     for c in ('rpU_S','randI_S','randD_S')])
    with pytest.raises(ValueError,match='Missing'): check_realized_controls(df.iloc[:2])
    df.loc[2,'realized_l2']=np.nan
    with pytest.raises(ValueError,match='Invalid'): check_realized_controls(df)


def test_scorer_no_hook_zero_and_observer_offset():
    """Exercise the real Qwen hook and chunked loss engine with a tiny CPU LM."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from eval_token_delta_sae_causal import score_with_token_patches
    torch.manual_seed(1)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=8, attention_dropout=0.)).eval()
    class Tok:
        def __call__(self,texts,**kwargs):
            ids=torch.tensor([[1,2,3,4] for _ in texts])
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    kw=dict(layer=1,max_seq_len=16,batch_size=1,full_logits_max_gib=0.,max_logit_elements=64,
            strict_patch_shapes=True)
    texts=['a','b']; patches=[np.zeros((4,16),np.float32) for _ in texts]
    plain=score_with_token_patches(model,Tok(),texts,None,**kw)
    seen=[]
    def observer(j,b,q,c,a):
        seen.append(j)
        assert torch.equal(b,a)
    zero=score_with_token_patches(model,Tok(),texts,patches,patch_observer=observer,**kw)
    np.testing.assert_array_equal(plain,zero)
    assert seen==[0,1]
    patches[1]=patches[1][:2]
    with pytest.raises(ValueError,match='shape'):
        score_with_token_patches(model,Tok(),texts,patches,**kw)
    assert not model.model.layers[0]._forward_hooks


def test_ranking_early_filter_preserves_original_random_sample(tmp_path):
    from reselect_token_sae_features import load_rows_base, fold_view
    folder=tmp_path/'layer_26'; folder.mkdir()
    users={0:'discovery',1:'confirmation',2:'negative',3:'insider_negative'}
    labels={0:1,1:1,2:0,3:0}
    for cid in range(3):
        idx=np.tile(np.arange(4),20)
        delta=np.arange(160,dtype=np.float16).reshape(80,2)+cid
        torch.save({'delta':delta,'example_idx':idx},folder/f'chunk_{cid:05d}.pt')
    kw=dict(user_by_example=users,y_by_example=labels,benign_sample_prob=.25,seed=42)
    x,y,u=load_rows_base(tmp_path,26,**kw)
    want,wy,_=fold_view(x,y,u,{'discovery'},negative_users={'negative'})
    stats={}
    a,b,c=load_rows_base(tmp_path,26,eligible_positive_users={'discovery'},
                        eligible_negative_users={'negative'},load_stats=stats,**kw)
    np.testing.assert_array_equal(a,want); np.testing.assert_array_equal(b,wy)
    assert set(c)=={'discovery','negative'} and stats['positive_rows_dropped']==60


def test_pilot_view_aggregation_preserves_float32_class_sum_identity():
    from pilot_intervention_specificity import view_means
    from token_class_decomposition import CERT_CLASSES, CERT_VIEWS
    rng=np.random.default_rng(8)
    sums=rng.uniform(0,10000,(40,len(CERT_CLASSES))).astype(np.float32)
    counts=rng.integers(1,400,(40,len(CERT_CLASSES))).astype(np.float32)
    names=list(CERT_CLASSES)
    views=view_means(sums,counts,names)
    n={v:counts[:,[names.index(c) for c in CERT_VIEWS[v]]].astype(np.float64).sum(1)
       for v in ('full','profile_only','behavior_only')}
    recombined=(views['profile_only']*n['profile_only']+views['behavior_only']*n['behavior_only'])/n['full']
    np.testing.assert_allclose(views['full'],recombined,rtol=1e-12,atol=1e-12)
