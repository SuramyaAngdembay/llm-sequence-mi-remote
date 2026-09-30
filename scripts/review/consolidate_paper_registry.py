#!/usr/bin/env python3
"""Consolidate already-reviewed results without rerunning their evaluations."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--paper', type=Path, required=True)
    a = p.parse_args()
    base = a.evidence_root
    paths = {
        'cached': base / 'mi-paper-recovery-2026-09-28/evaluation_v1/results.json',
        'c32': base / 'mi-paper-label-fix-2026-09-29/c32_complete/c32_complete_results.json',
        'pilot': base / 'mi-paper-causal-validation-2026-09-29/full_pilot_implementation_diagnostic_v1.json',
        'plan': base / 'mi-paper-causal-validation-2026-09-29/current_recovery_plan.json',
        'baselines': base / 'mi-paper-recovery-closeout-2026-09-29/saved_baseline_fold_audit.json',
        'fixed': base / 'mi-paper-recovery-closeout-2026-09-29/fixed_pilots/fixed_pilot_recovery.json',
        'fixed_checks': base / 'mi-paper-recovery-closeout-2026-09-29/fixed_pilots/independent_checks.json',
        'review': base / 'mi-paper-independent-review-20260928T183122/independent_label_checks.json',
    }
    # The independent review filename is located from its evidence root, never fabricated.
    if not paths['review'].exists():
        candidates = list((base / 'mi-paper-independent-review-20260928T183122').rglob('independent_label_checks.json'))
        if len(candidates) != 1:
            raise RuntimeError('Cannot uniquely locate independent label check')
        paths['review'] = candidates[0]
    data = {k: json.loads(v.read_text()) for k, v in paths.items()}
    claims = data['plan']
    assert len(claims) == 46 and len({c['id'] for c in claims}) == 46
    verified = {'C01', 'C02', 'C05', 'C06', 'C07', 'C08', 'C09', 'C28', 'C31', 'C32', 'C33', 'C34'}
    for c in claims:
        c['current_disposition'] = ('verified_with_scope_limits' if c['id'] in verified else
                                    'diagnostic_only' if c['id'] == 'C35' else
                                    'fixed_input_only' if c['id'] in {'C10', 'C36', 'C39'} else
                                    'provenance_qualified' if c['id'] in {'C03', 'C04'} else
                                    'conditional_only' if c['id'] in {'C19', 'C38', 'C43'} else
                                    'outside_label_repair' if c['id'] in {'C40', 'C41', 'C42'} else
                                    'manuscript_reconciliation' if c['id'] == 'C44' else
                                    'pending_dependency_repair')
        c['paper_claim_cleared'] = False  # Clearance belongs to a precise result, never an entire family.
        if c['id'] == 'C10':
            c['closeout_update'] = data['fixed_checks']['C10']
            c['completed_scope'] = 'Historical fixed-score reanalysis retained; donor manifests now audited against corrected receiver labels and official insiders (723 r42 / 60 r62 eligible pairs).'
            c['remaining_scope'] = 'Reconstruct corrected donor policy and score missing or changed pairs; fixed eligible subset is not full-population recovery.'
            c['next_action'] = 'Freeze full corrected receiver/donor assignments and a timing slice only if retaining a population input-swap claim.'
        if c['id'] in {'C36', 'C39'}:
            c['closeout_update'] = {k: v for k, v in data['fixed']['analyses'][c['id']].items() if k != 'results'}
            c['completed_scope'] = 'CPU corrected-class and fixed-cohort response summaries, partner audit, eligible-subset sensitivity, and independent arithmetic verification completed.'
            c['remaining_scope'] = 'Corrected population matching, feature/dictionary dependencies and any new changed-partner responses remain pending; no mediation or attack-specific mechanism cleared.'
            c['next_action'] = 'Use verified fixed-cohort summaries with explicit limits; reconstruct a corrected matched population before any causal class claim.'
        if c['id'] == 'C46':
            c['closeout_update'] = {k: {x: y for x, y in v.items() if x != 'rows'} for k, v in data['baselines']['datasets'].items()}
            c['completed_scope'] = 'All 128 actual saved bundles audited, hashed, and checked against official insiders: 94/120 r42 and 8/8 r62 training-contaminated; all r42 test groups affected.'
            c['remaining_scope'] = 'Repair required folds, refit affected models/scalers, and repeat label/model-dependent selection and interventions. Evaluate 26 clean-training r42 bundles for restricted exact reuse.'
            c['next_action'] = 'Freeze retained comparator protocol and time one corrected fold before scheduling necessary refits.'
        if c['id'] == 'C44':
            c['current_disposition'] = 'synchronized_recovery_editions'
            c['completed_scope'] = 'All three recovery editions use shared evidence, generated values and an explicit 46-family map; all three PDF builds verified.'
            c['remaining_scope'] = 'Future evidence must update the registry and all builds; unsupported historical claims are not reinstated.'
            c['next_action'] = 'Maintain the synchronized recovery editions; update only when new evidence clears a precise result scope.'
        if c['id'] in {'C27', 'C45', 'C46'}:
            c['publication_limit'] = 'No corrected model-performance ordering or architecture-specific mechanistic comparison.'
    cached = data['cached']
    registry = {
        'schema': 'cert-paper-evidence-v1', 'date': '2026-09-29',
        'label_version': 'cert-master-window-calendar-v1',
        'sources': {k: {'path': str(v), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in paths.items()},
        'claims': claims,
        'populations': {k: v['metadata'] for k, v in data['review']['datasets'].items()},
        'cached': {key: [r for r in cached[key] if r.get('label_state') == 'y']
                   for key in ('fold_summary', 'pooled', 'paired_contrasts', 'mean_gap', 'fixed_sample')},
        'c32': data['c32'],
        'pilot_diagnostic': {key: data['pilot'][key] for key in ('endpoints', 'secondary', 'analysis_class',
                             'scientific_claim_cleared', 'scope', 'n_users', 'n_attack_receivers',
                             'n_window_negative_receivers', 'implementation_validation')},
        'fixed_pilots': data['fixed'], 'fixed_checks': data['fixed_checks'],
        'baseline_audit_summary': {k: {x: y for x, y in v.items() if x != 'rows'}
                                   for k, v in data['baselines']['datasets'].items()},
        'interpretation_limits': [
            'Score mitigation does not remove profile context or establish internal profile independence.',
            'Main and factorial adapters are separate model families; never merge their metrics.',
            'Fold means and pooled metrics have different weighting and prevalence.',
            'Unseen validation users did not enter adaptation, but are not a newly untouched test cohort.',
            'Pilot original global control gate failed; diagnostics do not pass it retroactively.',
            'Historical causal and comparison-model claims remain pending unless their exact dependencies are cleared.',
        ],
    }
    dest = a.paper / 'data'; dest.mkdir(exist_ok=True)
    (dest / 'recovery_registry.json').write_text(json.dumps(registry, indent=2, allow_nan=False) + '\n')
    print('Wrote registry:', len(claims), 'claim families; source hashes:', len(paths))


if __name__ == '__main__':
    main()
