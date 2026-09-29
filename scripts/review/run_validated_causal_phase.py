#!/usr/bin/env python3
"""Bounded, fail-closed execution of the already specified corrected pilot.

Uses prepared inputs in a new phase directory; never resumes partial GPU work.
The original Stage C driver remains held. A failed physical matching gate stops
before the full pilot, while preserving the complete engineering diagnostics.
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cert_calendar_labels import file_sha256


def read(p): return json.loads(Path(p).read_text())
def write(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def gpu_free():
    uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(row.split(',')[0].strip()==uuid for row in rows.splitlines()):
        raise RuntimeError('Aquaman GPU 0 is occupied; no job launched')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--phase-dir',type=Path,required=True)
    a=ap.parse_args(); w=a.work_dir; p=a.phase_dir
    repo=Path(__file__).resolve().parents[2]
    plan_path=p/'execution_plan_v1.json'
    if plan_path.exists(): raise ValueError('This execution plan already exists; do not resume implicitly')
    live=p/'live_states_b1_v2'; sample=p/'presampled_l26'; x=w/'aquaman_l26_v1/merged_calendar_v2'
    labels=w/'labels_v2'; frontier=p/'frontier_v1'; selected=p/'frontier_selected_v1'
    for directory in (frontier,selected,p/'engineering_pilot_v1',p/'full_pilot_v1'):
        if directory.exists(): raise ValueError(f'Refusing existing output: {directory}')
    verification=read(p/'verify_live_scoring_v1.json')
    comparison=read(p/'cache_comparison_v1/COMPLETED.json')
    if (verification.get('passed') is not True or verification.get('examples')!=16
            or comparison.get('status')!='complete' or comparison.get('examples')!=356
            or any(m['live_manifest_sha256']!=file_sha256(live/'COMPLETED.json') for m in [verification,comparison])):
        raise ValueError('State-validation prerequisites did not pass')
    # Charge 60s conservatively for the failed capture, whose last error log was
    # at 12.3s. Other state checks carry their measured wall times plus startup.
    used=60.+read(live/'COMPLETED.json')['seconds']+verification['seconds']+10.
    plan={'started_at_epoch':time.time(),'gpu_wall_cap_seconds':8100,'state_validation_charged_seconds':used,
          'code_directory':str(repo),'code_sha256':{str(f.relative_to(repo)):file_sha256(f)
          for folder in ('scripts','configs') for f in sorted((repo/folder).rglob('*'))
          if f.is_file() and f.suffix in ('.py','.yaml')},
          'input_sha256':{str(f):file_sha256(f) for f in [live/'COMPLETED.json',p/'verify_live_scoring_v1.json',
              p/'cache_comparison_v1/COMPLETED.json',p/'engineering_cohort_v1.json',w/'pilot_populations_v1.json',
              labels/'labels_manifest.json',labels/'ranking_labels_v1.parquet',labels/'sampling_labels_v1.parquet']},
          'caps':{'dictionary':4500,'ranking':1200,'engineering':600,'full':3600},
          'full_budget_rule':'at most 60 minutes, remaining total envelope, and sufficient predicted runtime',
          'steps':[],'status':'waiting_for_preparation'}
    write(plan_path,plan)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HOME='/data/suramya/insider_mi/hf_cache',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false',
             PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')

    def step(name, seconds, args):
        nonlocal used
        if used+seconds>8100: raise RuntimeError('Remaining GPU budget is too small for this step')
        gpu_free()
        command=[sys.executable]+list(map(str,args))
        rec={'name':name,'command':command,'started_at_epoch':time.time(),'wall_cap_seconds':seconds,'status':'running'}
        plan['steps'].append(rec); plan['status']=name; write(plan_path,plan)
        start=time.monotonic()
        with (p/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rec['pid']=child.pid; write(plan_path,plan)
            try: code=child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
                code=124
        elapsed=time.monotonic()-start; used+=elapsed
        rec.update(status='complete' if code==0 else 'failed',returncode=code,seconds=elapsed)
        plan['gpu_wall_seconds_charged']=used; write(plan_path,plan)
        print(f'{name}: code={code}, seconds={elapsed:.1f}',flush=True)
        if code: raise RuntimeError(f'{name} failed with exit {code}; dependent jobs remain held')

    try:
        deadline=read(p/'presample_launch.json')['launched_at_epoch']+2700
        while not (sample/'COMPLETED.json').exists():
            if time.time()>deadline: raise RuntimeError('CPU preparation exceeded its existing 45-minute bound')
            time.sleep(10)
        done=read(sample/'COMPLETED.json')
        if done.get('status')!='complete': raise ValueError('Incomplete dictionary sample')
        for name,digest in {**done['files'],**done['chunks']}.items():
            if file_sha256(sample/name)!=digest: raise ValueError(f'Sampled input checksum failed: {name}')
        if read(sample/'presample_report.json')['rows_written']!=2000000: raise ValueError('Unexpected dictionary size')
        plan['input_sha256'][str(sample/'COMPLETED.json')]=file_sha256(sample/'COMPLETED.json'); write(plan_path,plan)
        mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
        if mem['MemAvailable']<72*1024**2: raise RuntimeError('Less than 72 GiB RAM available for the frozen dictionary fit')
        step('dictionary',4500,['scripts/train_delta_sae_frontier.py','--config','configs/token_delta_sae_frontier.yaml',
             '--extract-dir',sample,'--out-dir',frontier,'--device','cuda','--batch-size',4096,'--epochs',20,
             '--layers',26,'--max-rows',2000000,'--latent-multipliers',2,'--topk',4,'--seed',42,'--torch-seed',42,'--dictionary-only'])
        import torch
        import numpy as np
        model_path=frontier/'layer_26/m02_k04/delta_sae_model.pt'
        bundle=torch.load(model_path,map_location='cpu',weights_only=False)
        if (bundle['d_in'],bundle['d_latent'],bundle['k'])!=(4096,8192,4): raise ValueError('Wrong SAE architecture')
        if not all(torch.isfinite(v).all() for v in bundle['state_dict'].values()): raise ValueError('Nonfinite SAE weights')
        if not all(np.isfinite(bundle[k]).all() for k in ('x_mean','x_std')): raise ValueError('Nonfinite SAE normalization')
        del bundle
        write(frontier/'COMPLETED.json',{'status':'complete','model_sha256':file_sha256(model_path),
              'sample_manifest_sha256':file_sha256(sample/'COMPLETED.json'),'execution_plan':str(plan_path)})
        step('ranking',1200,['scripts/reselect_token_sae_features.py','--extract-dir',x,'--data-dir',w/'data_r42',
             '--frontier-dir',frontier,'--out-frontier-dir',selected,'--layer',26,'--latent-mult',2,'--k',4,
             '--discovery-user-file',labels/'discovery_users_v1.txt','--negative-user-file',labels/'negative_users_v1.txt',
             '--labels-override',labels/'ranking_labels_v1.parquet','--benign-sample-prob',.25,'--batch-size',8192,
             '--device','cuda','--seed',42,'--active-control-min-frac',.002])
        ranking=read(selected/'layer_26/m02_k04/reselect_summary.json')
        features=ranking['feature_sets']
        if len(features['top5'])!=5 or len(features['control5_active'])!=5: raise ValueError('Missing frozen feature sets')
        plan['feature_sets']=features; write(plan_path,plan)
        common=['scripts/pilot_intervention_specificity.py','--config','configs/qwen3_8b_qlora_session_targeted.yaml',
            '--data-dir',w/'data_r42','--adapter-dir','/home/suramya/cert-data/r42_adapter','--extract-dir',x,
            '--live-state-dir',live,'--frontier-dir',selected,'--populations',w/'pilot_populations_v1.json',
            '--selected',','.join(map(str,features['top5'])),'--control',','.join(map(str,features['control5_active'])),
            '--exclude-users','','--labels',labels/'ranking_labels_v1.parquet','--insider-file',w/'official_insiders_r42.txt',
            '--batch-size',1,'--loss-batch-size',1,'--seed',20260928]
        conditions='zero,rpU_S,randI_S,randD_S,rpU_C,randI_C,randD_C,recon_S,recon_C,orig_S,orig_C,rpO_S,rpO_C,rpU_S@0.5,rpU_C@0.5'
        cohort=read(p/'engineering_cohort_v1.json')
        step('engineering',600,common+['--out-dir',p/'engineering_pilot_v1','--conditions',conditions,
              '--pair-indices',','.join(map(str,cohort['pair_indices'])),'--engineering-only'])
        from cert_pilot_contracts import load_validated_run
        load_validated_run(p/'engineering_pilot_v1',require_causal_validity=False)
        gate=read(p/'engineering_pilot_v1/implementation_validation.json')
        if not gate['passed']:
            plan.update(status='stopped_at_realized_control_gate',full_pilot_launched=False)
            write(plan_path,plan); print('Physical control matching failed; full pilot remains held.',flush=True); return
        import pandas as pd
        sizes=pd.read_parquet(x/'example_scores.parquet').set_index('example_idx').n_tokens
        all_receivers=[i for pair in read(w/'pilot_populations_v1.json')['receiver_pairs'] for i in pair]
        small_receivers=[i for pair in cohort['pairs'] for i in pair['receivers']]
        ratio=float(sizes.loc[all_receivers].sum()/sizes.loc[small_receivers].sum())
        estimated=90.+1.2*max(0.,plan['steps'][-1]['seconds']-60.)*ratio
        cap=min(3600,int(8100-used)-15)
        plan.update(full_predicted_seconds=estimated,full_available_seconds=cap); write(plan_path,plan)
        if estimated>cap:
            plan.update(status='stopped_at_compute_bound',full_pilot_launched=False); write(plan_path,plan)
            print('Engineering gates passed, but full pilot exceeds remaining envelope.',flush=True); return
        step('full',cap,common+['--out-dir',p/'full_pilot_v1','--conditions',conditions])
        load_validated_run(p/'full_pilot_v1')
        subprocess.run([sys.executable,'scripts/review/analyze_corrected_pilot.py','--run-dir',str(p/'full_pilot_v1'),
                        '--answers',str(w/'answers.tar.bz2'),'--out',str(p/'full_pilot_analysis_v1.json')],cwd=repo,check=True)
        plan.update(status='complete',full_pilot_launched=True); write(plan_path,plan)
    except BaseException as exc:
        plan.update(status='failed',failure=str(exc),gpu_wall_seconds_charged=used); write(plan_path,plan)
        raise


if __name__=='__main__': main()
