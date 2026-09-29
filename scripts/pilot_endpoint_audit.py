"""Outcome-blind dependencies for diagnostics after a pilot's global gate fails.

Passing an endpoint's local checks does not retrospectively pass the original
global protocol. This module never consumes model loss values.
"""
from __future__ import annotations

import numpy as np
from intervention_audit import check_realized_controls


def endpoint_implementation_audit(rows, tokens):
    identity = rows[['receiver_idx', 'kind']].drop_duplicates()
    if identity.receiver_idx.duplicated().any() or set(identity.kind) != {'attack', 'window_negative'}:
        raise ValueError('Ambiguous or incomplete receiver day types')
    comparisons = {}
    failures = []
    for kind, ids in identity.groupby('kind'):
        subset = tokens[tokens.receiver_idx.isin(ids.receiver_idx)]
        for arm in ('S', 'C'):
            ref = subset[subset.condition == f'rpU_{arm}'].set_index(['receiver_idx', 'position'])
            if ref.empty or not ref.index.is_unique:
                raise ValueError('Missing or duplicate reference token grid')
            for condition in (f'randI_{arm}', f'randD_{arm}'):
                other = subset[subset.condition == condition].set_index(['receiver_idx', 'position'])
                if not other.index.is_unique or set(ref.index) != set(other.index):
                    raise ValueError('Random-control grid differs from reference')
                other = other.loc[ref.index]
                a, b = ref.realized_l2.to_numpy(), other.realized_l2.to_numpy()
                if not np.isfinite(np.r_[a, b]).all() or (np.r_[a, b] < 0).any():
                    raise ValueError('Invalid realized norms')
                active = a > 0
                support_bad = (a > 0) != (b > 0)
                errors = np.zeros(len(a))
                errors[active] = np.abs(b[active]-a[active])/a[active]
                norm_bad = active & (errors > .05)
                passed = bool(active.any()) and not (support_bad | norm_bad).any()
                comparisons[f'{condition} | {kind}'] = dict(reference=f'rpU_{arm}',
                    active_reference_tokens=int(active.sum()), support_mismatches=int(support_bad.sum()),
                    norm_mismatches=int(norm_bad.sum()), max_relative_error=float(errors.max()), passed=passed)
                for n in np.flatnonzero(support_bad | norm_bad):
                    receiver, position = ref.index[n]
                    failures.append(dict(condition=condition, kind=kind, receiver_idx=int(receiver),
                        position=int(position), support_mismatch=bool(support_bad[n]),
                        relative_norm_error=float(errors[n]) if active[n] else None))
    # These are the dependencies of the frozen equations, not a new choice of
    # equations, cohorts, conditions, seeds or matched-dose tolerance.
    definitions = {
        'primary': (['zero', 'rpU_S', 'randI_S'], ['attack'], ['randI_S | attack']),
        'co_primary': (['zero', 'rpU_S', 'rpU_C'], ['attack'], []),
        'specificity': (['zero', 'rpU_S', 'rpU_C'], ['attack', 'window_negative'], []),
        'secondary_dictionary_control': (['zero', 'rpU_S', 'randD_S'], ['attack'], ['randD_S | attack']),
        'secondary_control_isotropic': (['zero', 'rpU_C', 'randI_C'], ['attack'], ['randI_C | attack']),
        'secondary_own_support': (['zero', 'rpU_S', 'rpO_S'], ['attack'], []),
        'secondary_reconstruction_interaction': (['zero', 'orig_S', 'rpU_S', 'recon_S'], ['attack'], []),
        'secondary_half_dose': (['zero', 'rpU_S', 'rpU_C', 'rpU_S@0.5', 'rpU_C@0.5'], ['attack'], []),
    }
    endpoints = {}
    available = set(rows.condition)
    for name, (conditions, kinds, required_matches) in definitions.items():
        if not set(conditions) <= available:
            raise ValueError(f'Missing conditions for {name}')
        affected = [v for v in failures if v['condition'] in conditions and v['kind'] in kinds]
        endpoints[name] = dict(conditions=conditions, day_types=kinds, required_matching=required_matches,
            local_matching_prerequisites_pass=all(comparisons[k]['passed'] for k in required_matches),
            failed_random_control_rows_used=affected,
            interpretation='Policy contrast; not assumed equal-dose across selected/control feature sets.'
            if not required_matches else 'Uses realized support/norm checks on its stated random-control comparison.')
    return dict(analysis_class='post-run implementation diagnostic', original_global_gate_passed=check_realized_controls(tokens)['passed'],
        original_protocol_reclassified=False, scientific_claim_cleared=False,
        tolerance=.05, rows_excluded=0, comparisons=comparisons, failures=failures, endpoints=endpoints,
        scope='Local checks describe dependencies only. They do not override the original all-controls gate, establish mediation, or make an exploratory population a fresh holdout.')
