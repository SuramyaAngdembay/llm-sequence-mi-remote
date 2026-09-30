"""Strict selectors for the accepted, scoped result registry."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / 'data/recovery_registry.json').read_text())


def select(kind, **query):
    rows = [r for r in REGISTRY['cached'][kind]
            if all(r.get(k) == v for k, v in query.items())]
    if len(rows) != 1:
        raise ValueError(f'Expected one result for {kind} {query}; got {len(rows)}')
    return rows[0]


def fold(run, view, metric='day_roc', population='unseen_users'):
    return select('fold_summary', run=run, view=view, metric=metric, population=population)


def value(run, view, metric='day_roc', population='unseen_users'):
    return fold(run, view, metric, population)['estimate']
