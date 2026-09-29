"""Read verified, batch-1 layer states for corrected intervention pilots."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cert_calendar_labels import VERSION, file_sha256


def validate_live_manifest(root, *, populations, labels, adapter_dir, config, layer, eval_jsonl):
    root = Path(root)
    m = json.loads((root/'COMPLETED.json').read_text())
    if (m.get('contract') != 'pilot-live-states-v1' or m.get('status') != 'complete'
            or m.get('label_version') != VERSION or m.get('batch_size') != 1
            or m.get('layer') != layer or m.get('hook_matches_hidden_state_index_exactly') is not True):
        raise ValueError('Live-state capture did not pass the required gates')
    for key, p in [('populations', populations), ('labels', labels), ('config', config),
                   ('adapter', Path(adapter_dir)/'adapter_model.safetensors'), ('eval_jsonl', eval_jsonl)]:
        if m.get(key+'_sha256') != file_sha256(p):
            raise ValueError(f'Live-state input mismatch: {key}')
    for name, digest in m.get('adapter_support_sha256', {}).items():
        if Path(name).name != name or digest != file_sha256(Path(adapter_dir)/name):
            raise ValueError('Live-state tokenizer/adapter config changed')
    return m


def load_live_arrays(root, m, needed):
    """Verify each selected file once; retain only deltas in memory."""
    parts, indices = [], []
    records = {r['example_idx']: r for r in m['examples']}
    for i in sorted(needed):
        name = f'example_{i:05d}.npz'
        path = Path(root)/name
        if i not in records or m['files'].get(name) != file_sha256(path):
            raise ValueError(f'Missing or modified live state: {i}')
        with np.load(path) as z:
            d = z['delta']
            if (d.ndim != 2 or len(d) != records[i]['n_tokens'] or not np.isfinite(d).all()
                    or z['adapted_hidden'].shape != d.shape or z['base_hidden'].shape != d.shape
                    or len(z['input_ids']) != len(d)):
                raise ValueError(f'Invalid live state: {i}')
            parts.append(d)
            indices.append(np.full(len(d), i, np.int64))
    return np.concatenate(parts), np.concatenate(indices)


def check_live_before(root, example_idx, before):
    with np.load(Path(root)/f'example_{example_idx:05d}.npz') as z:
        expected = z['adapted_hidden']
    actual = before.detach().float().cpu().numpy()
    if not np.array_equal(actual, expected):
        err = float(np.max(np.abs(actual-expected))) if actual.shape == expected.shape else None
        raise ValueError(f'Live pre-edit state differs for example {example_idx}; max_abs={err}')
