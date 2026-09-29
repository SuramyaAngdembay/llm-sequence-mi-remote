#!/usr/bin/env python3
"""Verify stored live states and zero-edit identity through the real scorer."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import file_sha256
from eval_token_delta_sae_causal import load_eval_examples, score_with_token_patches
from pilot_live_states import validate_live_manifest, load_live_arrays, check_live_before
from intervention_audit import realized_token_stats
from remote_common import load_yaml


def main():
    ap=argparse.ArgumentParser()
    for key in ('config','data-dir','adapter-dir','extract-dir','labels','populations','live-state-dir','cohort','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists(): raise ValueError('Refusing to overwrite validation')
    t0=time.monotonic()
    cfg=load_yaml(a.config)
    cohort=json.loads(a.cohort.read_text())
    if cohort['populations_sha256']!=file_sha256(a.populations): raise ValueError('Cohort/populations mismatch')
    pops=json.loads(a.populations.read_text())
    needed=[i for j in cohort['pair_indices'] for i in pops['receiver_pairs'][j]]
    live=validate_live_manifest(a.live_state_dir,populations=a.populations,labels=a.labels,
          adapter_dir=a.adapter_dir,config=a.config,layer=26,eval_jsonl=a.data_dir/'eval.jsonl')
    load_live_arrays(a.live_state_dir,live,needed)  # Validate checksums/shapes before loading model.
    scores=pd.read_parquet(a.extract_dir/'example_scores.parquet').sort_values('example_idx').reset_index(drop=True)
    meta=load_eval_examples(a.data_dir,scores)
    texts=[meta.loc[i,'text'] for i in needed]
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(a.adapter_dir,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    q=cfg['quantization']
    model=AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'],device_map='auto',torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=bool(q['load_in_4bit']),
            bnb_4bit_quant_type=q['bnb_4bit_quant_type'],bnb_4bit_compute_dtype=getattr(torch,q['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=bool(q['bnb_4bit_use_double_quant'])))
    model=PeftModel.from_pretrained(model,a.adapter_dir); model.eval(); model.config.use_cache=False
    kw=dict(layer=26,max_seq_len=int(cfg['training']['max_seq_len']),batch_size=1,loss_batch_size=1,
            full_logits_max_gib=0.,max_logit_elements=4194304,class_schema='cert',strict_patch_shapes=True)
    checked=[]
    def observe(j,before,proposed,cast,after):
        check_live_before(a.live_state_dir,needed[j],before)
        stats=realized_token_stats(before,proposed,cast,after)
        if np.any(stats['realized_l2']): raise ValueError('Zero patch changed state')
        checked.append(needed[j])
    zero=[]
    for i in needed:
        with np.load(a.live_state_dir/f'example_{i:05d}.npz') as z:
            if not np.array_equal(tok(meta.loc[i,'text'],truncation=True,max_length=kw['max_seq_len'])['input_ids'],z['input_ids']):
                raise ValueError('Live input IDs changed')
            zero.append(np.zeros_like(z['delta']))
    plain=score_with_token_patches(model,tok,texts,None,**kw)
    patched=score_with_token_patches(model,tok,texts,zero,patch_observer=observe,**kw)
    errors=[float(np.max(np.abs(x-y))) for x,y in zip(plain[:3],patched[:3])]
    if checked!=needed or not np.isfinite(errors).all() or max(errors)>1e-5:
        raise ValueError(f'Live/zero scoring validation failed: {errors}')
    report={'status':'complete','passed':True,'examples':len(needed),'receiver_indices':needed,
            'live_pre_edit_states_exact':True,'zero_realized_edits_exact':True,
            'unhooked_vs_zero_max_abs_by_output':dict(zip(['mean_nll','class_loss_sums','class_counts'],errors)),
            'live_manifest_sha256':file_sha256(a.live_state_dir/'COMPLETED.json'),
            'cohort_sha256':file_sha256(a.cohort),'seconds':time.monotonic()-t0,
            'code_sha256':{p.name:file_sha256(p) for p in [Path(__file__),Path(__file__).resolve().parents[1]/'eval_token_delta_sae_causal.py']}}
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(json.dumps(report))


if __name__=='__main__': main()
