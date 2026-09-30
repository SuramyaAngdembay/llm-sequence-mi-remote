#!/usr/bin/env python3
"""Separate arithmetic checks and donor audit for the CPU recovery.

Uses csv + explicit user aggregation; does not import the recovery analyzer.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--historical-h6", type=Path, required=True)
    a = p.parse_args()
    rec = json.loads((a.root / "fixed_pilots/fixed_pilot_recovery.json").read_text())
    max_error, checks = 0., 0
    raw_effect_error, raw_effect_checks = 0., 0
    for name in ("C36", "C39"):
        rows = list(csv.DictReader((a.root / f"fixed_pilots/{name}_effects.csv").open()))
        raw_name, condition = ('pilot2_rows.csv', 'variant') if name == 'C36' else ('pilot_rows.csv', 'condition')
        raw = {(r['receiver_id'], r[condition]): r for r in csv.DictReader((a.root / 'inputs' / raw_name).open())}
        definitions = {'rescue_dec': ('R', 'dec'), 'rescue_I2_sel': ('R', 'I2_sel'),
                       'rescue_I1_sel': ('R', 'I1_sel'), 'rescue_I2_ctl': ('R', 'I2_ctl'),
                       'rescue_rand_I2': ('R', 'rand_I2'), 'necessity_noise_I2': ('noise_I2', 'O'),
                       'necessity_rand_noise': ('rand_noise', 'O'),
                       'I2_sel_minus_I2_ctl': ('I2_ctl', 'I2_sel'),
                       'I2_sel_minus_rand_I2': ('rand_I2', 'I2_sel'),
                       'I2_sel_minus_I1_sel': ('I1_sel', 'I2_sel'),
                       'noise_I2_minus_rand_noise': ('noise_I2', 'rand_noise'),
                       'input_effect_R_minus_O': ('R', 'O')}
        for row in rows:
            rid = row['receiver_id']
            if name == 'C36':
                view, variant, model = row['endpoint'].split('|')
                c1, c0 = ('swap_train', 'swap_same') if variant.startswith('familiarity_') else (variant, 'orig')
                models = [('adapted', 1), ('base', -1)] if model == 'adapted_minus_base' else [(model, 1)]
                expected = sum(sign * (float(raw[rid, c1][m+'_'+view]) - float(raw[rid, c0][m+'_'+view]))
                               for m, sign in models)
            else:
                c1, c0 = definitions[row['endpoint']]
                expected = float(raw[rid, c1]['behavior_only']) - float(raw[rid, c0]['behavior_only'])
            raw_effect_error = max(raw_effect_error, abs(expected - float(row['effect'])))
            raw_effect_checks += 1
        for key, saved in rec["analyses"][name]["results"].items():
            endpoint, scope, group = key.rsplit("|", 2)
            selected = [r for r in rows if r["endpoint"] == endpoint and
                        (scope == "all_fixed" or r["eligible"] == "True")]
            users = defaultdict(list)
            if group == "attack_minus_window_negative":
                groups = defaultdict(lambda: defaultdict(list))
                for r in selected:
                    groups[r["user"]][r["y"]].append(float(r["effect"]))
                for user, v in groups.items():
                    if '0' in v and '1' in v:
                        users[user] = [sum(v['1']) / len(v['1']) - sum(v['0']) / len(v['0'])]
            else:
                for r in selected:
                    if group == "all" or int(r["y"]) == int(group == "attack"):
                        users[r["user"]].append(float(r["effect"]))
            x = np.array([sum(v) / len(v) for _, v in sorted(users.items())])
            assert len(x) == saved["users"]
            if not len(x):
                assert saved["estimate"] is None
                continue
            vals, targets = [x.mean()], [saved["estimate"]]
            if len(x) > 1:
                half = t.ppf(.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                rng = np.random.default_rng(42)
                boots = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
                vals.extend([x.mean() - half, x.mean() + half, *np.quantile(boots, [.025, .975])])
                targets.extend(saved["ci90_t"] + saved["ci95_boot"])
            max_error = max(max_error, max(abs(u - v) for u, v in zip(vals, targets)))
            checks += len(vals)
    old = json.loads(a.historical_h6.read_text())
    old_errors = []
    for new, previous in [("rescue_dec", "rescue dec (H6 edit)"),
                          ("rescue_I2_sel", "rescue I2_sel"), ("rescue_I1_sel", "rescue I1_sel")]:
        fresh = rec["analyses"]["C39"]["results"][new + "|all_fixed|all"]
        prior = old["pooled_results"][previous]
        old_errors.extend(abs(u - v) for u, v in zip([fresh["estimate"], *fresh["ci90_t"]],
                                                     [prior["estimate"], *prior["ci90_t"]]))
    assert max_error < 1e-12 and max(old_errors) < 1e-12 and raw_effect_error < 1e-12
    official = json.loads((a.root / "official_insiders.json").read_text())
    swap_audits = {}
    for dataset in ("r42", "r62"):
        path = a.root / f"inputs/{dataset}_swap_manifest.json"
        donors = json.loads(path.read_text())
        meta = pd.read_parquet(a.recovery / f"labels/{dataset}/example_metadata.parquet").set_index("example_id")
        ids = [r["example_id"] for r in donors]
        assert len(ids) == len(set(ids)) and set(ids) <= set(meta.index)
        assert set(r["donor_example_id"] for r in donors) <= set(meta.index)
        audit = []
        for r in donors:
            receiver, donor = meta.loc[r["example_id"]], meta.loc[r["donor_example_id"]]
            audit.append(dict(r, receiver_y=int(receiver.y), donor_y=int(donor.y),
                              donor_official_insider=donor.user_id in official[dataset],
                              same_user=receiver.user_id == donor.user_id))
        df = pd.DataFrame(audit)
        df["corrected_eligible"] = ((df.receiver_y == 1) & ~df.donor_official_insider & ~df.same_user)
        df.to_csv(a.root / f"fixed_pilots/C10_{dataset}_donor_audit.csv", index=False)
        swap_audits[dataset] = {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "historical_receivers": len(df), "corrected_positive_receivers": int(df.receiver_y.sum()),
                               "donor_official_insider": int(df.donor_official_insider.sum()),
                               "donor_positive_days": int(df.donor_y.sum()), "same_user": int(df.same_user.sum()),
                               "eligible_fixed_pairs": int(df.corrected_eligible.sum()),
                               "missing_positive_receivers": int(meta.y.sum() - df.receiver_y.sum()),
                               "scope": "Manifest eligibility only; previous fixed-score reanalysis retained, full corrected population not scored."}
    report = {"recomputed_numbers": checks, "max_arithmetic_error": max_error,
              "raw_receiver_effect_checks": raw_effect_checks, "raw_receiver_effect_max_error": raw_effect_error,
              "historical_h6_pooled_max_error": max(old_errors), "C10": swap_audits}
    (a.root / "fixed_pilots/independent_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
