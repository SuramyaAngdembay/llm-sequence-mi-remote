#!/usr/bin/env python3
"""Check all current editions against the 46-family ledger and shared values."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper'


def expand(path, evidence_root, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError('Recursive input: ' + str(path))
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        rows.append((path, lineno, line))
        for name in re.findall(r'\\input\{([^}]+)\}', line):
            name = name.replace(r'\evidenceroot', str(evidence_root))
            # TeX resolves inputs against the invocation directory, not the included file.
            target = Path(name)
            if not target.is_absolute():
                target = ROOT_CWD / target
            rows += expand(target, evidence_root, stack + (path,))
    return rows


def main():
    global ROOT_CWD
    registry = json.loads((PAPER / 'data/recovery_registry.json').read_text())
    claims = {r['id']: r for r in registry['claims']}
    definitions = set(re.findall(r'csname rv@([^\\]+)\\endcsname',
                                (PAPER / 'tables/recovery_values.tex').read_text()))
    mapping = {cid: {'claim': row['claim'], 'disposition': row['current_disposition'],
                     'historical_location': row['manuscript_location'], 'editions': {}}
               for cid, row in claims.items()}
    for name in ('main.tex', 'companion/readable_full.tex', 'companion/plain_summary.tex'):
        source = PAPER / name
        ROOT_CWD = source.parent
        rows = expand(source, PAPER)
        known = set()
        for path, line, text in rows:
            if text.startswith('% claims:'):
                for cid in re.findall(r'C\d{2}', text):
                    assert cid in claims
                    known.add(cid)
                    mapping[cid]['editions'].setdefault(name, []).append({
                        'file': str(path.relative_to(ROOT)), 'line': line})
            for key in re.findall(r'\\rv\{([^}]+)\}', text):
                assert key in definitions, (name, 'undefined result', key)
            if r'\includegraphics' in text:
                assert 'recovery_score_mitigation.pdf' in text, (name, 'unreviewed figure')
        missing = set(claims) - known
        if missing:
            raise ValueError(f'{name} missing claim families: {sorted(missing)}')
    (PAPER / 'data/claim_manuscript_map.json').write_text(json.dumps(mapping, indent=2) + '\n')
    lines = ['# Current claim-to-manuscript map', '',
             'All 46 families are explicitly covered in each edition. Coverage includes limitations and deferred work; it does not mean every claim is supported.', '',
             '| Claim | Current disposition | Main / readable location |', '|---|---|---|']
    for cid, row in mapping.items():
        loc = row['editions']['main.tex'][0]
        lines.append(f"| {cid}: {row['claim']} | {row['disposition']} | `{loc['file']}:{loc['line']}` |")
    (PAPER / 'CLAIM_MAP.md').write_text('\n'.join(lines) + '\n')
    print('All 46 claim families mapped in each edition; result macros and figures resolved.')


if __name__ == '__main__':
    main()
