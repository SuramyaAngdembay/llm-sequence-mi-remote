#!/usr/bin/env python3
"""Write a corrected JSONL copy while preserving text, order and training exposure."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cert_calendar_labels import VERSION, calendar_dates, file_sha256, load_label_artifact


def rewrite(source: Path, destination: Path, labels_path: Path, metadata_path: Path):
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError('Choose a new corrected JSONL output')
    labels = load_label_artifact(labels_path)
    metadata = pd.read_parquet(metadata_path)
    if metadata.duplicated('example_id').any() or set(metadata.label_version) != {VERSION}:
        raise ValueError('Canonical metadata must have unique versioned identities')
    dates = calendar_dates(metadata.day_index).strftime('%Y-%m-%d')
    if list(dates) != metadata.calendar_date.tolist():
        raise ValueError('Metadata day/date inconsistency')
    positives = set(zip(labels.user_id, labels.calendar_date))
    expected_y = [int((u,d) in positives) for u,d in zip(metadata.user_id,metadata.calendar_date)]
    if expected_y != metadata.y.tolist():
        raise ValueError('Canonical metadata disagrees with label artifact')
    meta = metadata.set_index('example_id')[['user_id','day_index','calendar_date','split','y_original','y']].to_dict('index')
    # A length-delimited digest covers all original fields other than y; JSON byte
    # spacing may change, but serialized text and every other value must not.
    before, after = hashlib.sha256(), hashlib.sha256()
    seen = set(); n = changed = 0
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial = destination.with_suffix(destination.suffix+'.partial')
    if partial.exists():
        raise FileExistsError('Partial output exists; inspect it before choosing a new path')
    try:
        with source.open() as src, partial.open('x') as out:
            for line in src:
                r = json.loads(line)
                if 'label_version' in r or 'y_original' in r:
                    raise ValueError('Already-versioned record')
                eid = r['example_id']
                if eid in seen:
                    raise ValueError('Duplicate example ID in original JSONL')
                seen.add(eid)
                if eid not in meta:
                    raise ValueError('JSONL example absent from canonical metadata')
                m = meta[eid]
                for c in ['user_id','day_index','split']:
                    if r[c] != m[c]:
                        raise ValueError(f'JSONL disagrees with canonical {c}: {eid}')
                if r['y'] != m['y_original']:
                    raise ValueError('JSONL original label differs from preserved history')
                original_fields = {k:v for k,v in r.items() if k!='y'}
                blob = json.dumps(original_fields,sort_keys=True,ensure_ascii=True).encode()
                before.update(len(blob).to_bytes(8,'big'));before.update(blob)
                r.update(y=int(m['y']),y_original=int(m['y_original']),calendar_date=m['calendar_date'],label_version=VERSION)
                restored = {k:v for k,v in r.items() if k not in ['y','y_original','calendar_date','label_version']}
                blob = json.dumps(restored,sort_keys=True,ensure_ascii=True).encode()
                after.update(len(blob).to_bytes(8,'big'));after.update(blob)
                out.write(json.dumps(r,ensure_ascii=True)+'\n')
                n+=1;changed+=int(m['y']!=m['y_original'])
        if before.digest()!=after.digest():
            raise ValueError('Non-label fields changed')
        partial.rename(destination)
    except Exception:
        # Retain any partial file as a failure artifact; it is never a valid dataset.
        raise
    result={'source':str(source),'source_sha256':file_sha256(source),
            'output':str(destination),'output_sha256':file_sha256(destination),
            'label_sha256':file_sha256(labels_path),'metadata_sha256':file_sha256(metadata_path),
            'label_version':VERSION,'rows':n,'changed_labels':changed,
            'nonlabel_values_and_order_sha256':before.hexdigest(),
            'text_and_original_splits_preserved':True}
    destination.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(rewrite(a.source,a.destination,a.labels,a.metadata),indent=2))
