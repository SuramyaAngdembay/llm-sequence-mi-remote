#!/usr/bin/env python3
"""Independently reconstruct the frozen sample from token counts and label roles.

Does not call the trainer's sampling functions. The previously validated merged
cache has contiguous example/token order, so counts define its original row grid.
Checks chosen (example,position) pairs and target labels without reading all of
the original 89 GiB delta cache again.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import file_sha256
from cert_label_contracts import load_stage_labels


def main():
    ap=argparse.ArgumentParser()
    for key in ('extract-dir','sample-dir','labels','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists(): raise ValueError('Refusing to overwrite independent audit')
    done=json.loads((a.sample_dir/'COMPLETED.json').read_text())
    if done.get('status')!='complete': raise ValueError('Incomplete sample')
    scores=pd.read_parquet(a.extract_dir/'example_scores.parquet').sort_values('example_idx').reset_index(drop=True)
    if not np.array_equal(scores.example_idx, np.arange(len(scores))): raise ValueError('Source index is not contiguous')
    labels=load_stage_labels(a.labels,scores,purpose='sampling')
    counts=scores.n_tokens.to_numpy(np.int64)
    offsets=np.r_[0,np.cumsum(counts)]
    priority=np.repeat(labels.preserve_for_sampling.to_numpy(bool),counts)
    cap=2000000; seed=68
    probability=min(1.,cap*1.1/len(priority))
    keep=priority | (np.random.default_rng(seed).random(len(priority))<probability)
    rows=np.flatnonzero(keep)
    p=priority[rows]
    if len(rows)>cap:
        rng=np.random.default_rng(seed)
        positive=rows[p]; ordinary=rows[~p]
        if len(positive)>=cap:
            rows=np.sort(rng.choice(positive,size=cap,replace=False))
        else:
            rows=np.sort(np.r_[positive,rng.choice(ordinary,size=cap-len(positive),replace=False)])
    expected_example=np.searchsorted(offsets[1:],rows,side='right')
    expected_position=rows-offsets[expected_example]
    seen_examples=[]; seen_positions=[]
    for path in sorted((a.sample_dir/'layer_26').glob('chunk_*.pt')):
        key=str(path.relative_to(a.sample_dir))
        if done['chunks'].get(key)!=file_sha256(path): raise ValueError('Sample chunk changed')
        obj=torch.load(path,map_location='cpu',weights_only=False)
        seen_examples.append(np.asarray(obj['example_idx']).copy())
        seen_positions.append(np.asarray(obj['position']).copy())
        del obj
    got_e=np.concatenate(seen_examples); got_p=np.concatenate(seen_positions)
    if not np.array_equal(got_e,expected_example) or not np.array_equal(got_p,expected_position):
        raise ValueError('Sample differs from independent sampling reconstruction')
    output_scores=pd.read_parquet(a.sample_dir/'example_scores.parquet').sort_values('example_idx').reset_index(drop=True)
    if not np.array_equal(output_scores.y,labels.y) or not np.array_equal(output_scores.preserve_for_sampling,labels.preserve_for_sampling):
        raise ValueError('Targets or sampling roles changed in output index')
    report={'status':'complete','passed':True,'rows':len(rows),'priority_rows':int(priority[rows].sum()),
            'true_positive_rows':int(labels.y.to_numpy()[expected_example].sum()),
            'sampling_seed':seed,'first_stage_probability':probability,
            'ordered_example_position_pairs_exact':True,'targets_separate_from_sampling_roles':True,
            'source_completion_sha256':file_sha256(a.extract_dir/'COMPLETED.json'),
            'sample_completion_sha256':file_sha256(a.sample_dir/'COMPLETED.json'),
            'labels_sha256':file_sha256(a.labels),'audit_code_sha256':file_sha256(Path(__file__))}
    a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))


if __name__=='__main__': main()
