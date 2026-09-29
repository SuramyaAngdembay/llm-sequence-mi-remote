#!/usr/bin/env python3
"""Capture batch-1 base/adapted layer states for the fixed corrected pilot.

This is an implementation check, not an intervention outcome. Float32 storage
preserves the model's BF16 states exactly. Delta subtraction retains the original
extractor's hidden-dtype arithmetic. No full vocabulary logits are allocated.
"""
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
from cert_calendar_labels import VERSION, file_sha256
from cert_label_contracts import load_stage_labels
from eval_token_delta_sae_causal import get_layer_module, get_base_causal_lm, load_eval_examples
from remote_common import load_yaml


def main():
    ap = argparse.ArgumentParser()
    for key in ('config', 'data-dir', 'adapter-dir', 'extract-dir', 'labels', 'populations', 'out-dir'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--layer', type=int, default=26)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise ValueError('Use a new live-state directory')
    t0 = time.monotonic()
    cfg = load_yaml(a.config)
    scores = pd.read_parquet(a.extract_dir / 'example_scores.parquet').sort_values('example_idx').reset_index(drop=True)
    labels = load_stage_labels(a.labels, scores, purpose='target')
    meta = load_eval_examples(a.data_dir, scores)
    pops = json.loads(a.populations.read_text())
    if pops['eval_jsonl_sha256'] != file_sha256(a.data_dir / 'eval.jsonl'):
        raise ValueError('Population/data hash mismatch')
    needed = sorted({i for pair in pops['receiver_pairs'] for i in pair} |
                    {i for v in pops['donor_policy_by_dept'].values() for i in v})
    if len(needed) != len(set(needed)) or any(i not in scores.index for i in needed):
        raise ValueError('Invalid state population')
    a.out_dir.mkdir(parents=True)
    manifest = {'contract': 'pilot-live-states-v1', 'label_version': VERSION, 'batch_size': 1,
                'layer': a.layer, 'needed_examples': needed, 'max_seq_len': int(cfg['training']['max_seq_len']),
                'delta_arithmetic': '(adapted_hidden - base_hidden) in hidden dtype, then float32',
                'eval_jsonl_sha256': file_sha256(a.data_dir/'eval.jsonl'),
                'populations_sha256': file_sha256(a.populations), 'labels_sha256': file_sha256(a.labels),
                'adapter_sha256': file_sha256(a.adapter_dir/'adapter_model.safetensors'),
                'config_sha256': file_sha256(a.config), 'files': {}, 'examples': [],
                'code_sha256': {str(p.relative_to(Path(__file__).resolve().parents[2])): file_sha256(p)
                    for p in [Path(__file__).resolve(), Path(__file__).resolve().parents[1]/'eval_token_delta_sae_causal.py']},
                'adapter_support_sha256': {p.name: file_sha256(p) for p in sorted(a.adapter_dir.iterdir())
                    if p.is_file() and p.suffix == '.json'}}
    (a.out_dir/'manifest_before_capture.json').write_text(json.dumps(manifest, indent=2)+'\n')
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.adapter_dir, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.padding_side != 'right': raise ValueError('Right padding required')
    q = cfg['quantization']
    base = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'], device_map='auto',
        torch_dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(
            load_in_4bit=bool(q['load_in_4bit']), bnb_4bit_quant_type=q['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, q['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=bool(q['bnb_4bit_use_double_quant'])))
    model = PeftModel.from_pretrained(base, a.adapter_dir)
    model.eval(); model.config.use_cache = False
    devices = {str(p.device) for p in model.parameters()}
    if len(devices) != 1 or not next(iter(devices)).startswith('cuda:'):
        raise ValueError('State capture requires the entire model on one GPU')
    backbone = get_base_causal_lm(model).model
    capture = {}
    def hook(_module, _args, output):
        capture['h'] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    handle = get_layer_module(model, a.layer).register_forward_hook(hook)
    import transformers, peft, bitsandbytes
    manifest.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    transformers=transformers.__version__, peft=peft.__version__, bitsandbytes=bitsandbytes.__version__,
                    attention_implementation=model.config._attn_implementation,
                    model_name=cfg['model_name_or_path'], dtype='bfloat16')
    try:
        with torch.no_grad():
            for n, i in enumerate(needed):
                batch = tok(meta.loc[i, 'text'], return_tensors='pt', truncation=True,
                            max_length=manifest['max_seq_len'])
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with model.disable_adapter():
                    base_out = backbone(**batch, return_dict=True, use_cache=False)
                h_base = capture.pop('h')
                del base_out
                adapted_out = backbone(**batch, return_dict=True, use_cache=False, output_hidden_states=(n == 0))
                h_adapted = capture.pop('h')
                if n == 0:
                    same = torch.equal(h_adapted, adapted_out.hidden_states[a.layer])
                    manifest['hook_matches_hidden_state_index_exactly'] = same
                    if not same: raise ValueError('Hook indexes a different hidden state than extraction')
                del adapted_out
                delta = (h_adapted-h_base).float()[0].cpu().numpy()
                hb, ha = h_base.float()[0].cpu().numpy(), h_adapted.float()[0].cpu().numpy()
                ids = batch['input_ids'][0].cpu().numpy()
                if len(ids) != int(scores.loc[i, 'n_tokens']) or not all(np.isfinite(x).all() for x in (hb, ha, delta)):
                    raise ValueError('Live token alignment/finite-value failure')
                name = f'example_{i:05d}.npz'
                np.savez(a.out_dir/name, base_hidden=hb, adapted_hidden=ha, delta=delta, input_ids=ids)
                manifest['files'][name] = file_sha256(a.out_dir/name)
                manifest['examples'].append({'example_idx': i, 'example_id': meta.loc[i, 'example_id'],
                    'user_id': meta.loc[i, 'user_id'], 'y': int(labels.loc[i, 'y']), 'n_tokens': len(ids)})
                del h_base, h_adapted, hb, ha, delta, batch
                if (n+1) % 20 == 0: print(f'captured {n+1}/{len(needed)} examples', flush=True)
    finally:
        handle.remove()
    manifest['seconds'] = time.monotonic()-t0
    (a.out_dir/'COMPLETED.json').write_text(json.dumps(dict(manifest, status='complete'), indent=2)+'\n')
    print(json.dumps({'examples': len(needed), 'seconds': manifest['seconds'], 'layer_index_exact': True}), flush=True)


if __name__ == '__main__': main()
