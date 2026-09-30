#!/usr/bin/env python3
"""CPU-only, post-hoc relabeling of frozen Pilot 2 and H6 responses.

Never refits, reselects, changes a loss, or promotes a historical experiment to
a corrected population experiment. Eligible-subset summaries are sensitivities.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def summarize(frame):
    means = frame.groupby("user", sort=True)["effect"].mean()
    x = means.to_numpy(float)
    if len(x) < 2:
        return {"estimate": float(x.mean()) if len(x) else None,
                "users": len(x), "receivers": len(frame), "ci95_boot": None, "ci90_t": None}
    assert np.isfinite(x).all()
    rng = np.random.default_rng(42)
    boot = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    half = t.ppf(.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"estimate": float(x.mean()), "users": len(x), "receivers": len(frame),
            "ci95_boot": np.quantile(boot, [.025, .975]).tolist(),
            "ci90_t": [float(x.mean() - half), float(x.mean() + half)]}


def analyze_effects(effects):
    results = {}
    for endpoint, df in effects.groupby("endpoint", sort=True):
        for scope, sub in [("all_fixed", df), ("eligible_partner_subset", df[df.eligible])]:
            for group, selected in [("all", sub), ("attack", sub[sub.y == 1]),
                                    ("window_negative", sub[sub.y == 0])]:
                results[f"{endpoint}|{scope}|{group}"] = summarize(selected)
            a = sub[sub.y == 1].groupby("user").effect.mean()
            b = sub[sub.y == 0].groupby("user").effect.mean()
            common = a.index.intersection(b.index)
            paired = pd.DataFrame({"user": common.to_numpy(),
                                   "effect": (a[common] - b[common]).to_numpy()})
            s = summarize(paired)
            s["receivers"] = None
            s["scope_note"] = "Difference of corrected class means within shared users; historical day matching is not restored."
            results[f"{endpoint}|{scope}|attack_minus_window_negative"] = s
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--insiders", type=Path, required=True)
    ap.add_argument("--pilot2-manifest", type=Path, required=True)
    ap.add_argument("--h6-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    official = set(json.loads(args.insiders.read_text())["r42"])
    meta = pd.read_parquet(args.metadata).set_index("example_id")
    assert meta.index.is_unique
    pops_file = args.inputs / "pilot_populations.json"
    pops = json.loads(pops_file.read_text())
    manifest = json.loads(args.pilot2_manifest.read_text())
    assert hashlib.sha256(pops_file.read_bytes()).hexdigest() == manifest["populations_sha256"]
    h6_manifest = json.loads(args.h6_manifest.read_text())
    h6_partners = {x["id"]: x for x in h6_manifest["receivers"]}
    sources = [args.metadata, args.insiders, pops_file, args.pilot2_manifest, args.h6_manifest]
    output = {"label_version": "cert-master-window-calendar-v1", "post_hoc": True,
              "scope": "Frozen historical inputs, models, dictionaries and losses; no corrected population or mediation claim.",
              "analyses": {}}
    for name, filename, condition in [("C36", "pilot2_rows.csv", "variant"),
                                       ("C39", "pilot_rows.csv", "condition")]:
        path = args.inputs / filename
        sources.append(path)
        df = pd.read_csv(path)
        assert not df.duplicated(["receiver_id", condition]).any()
        receivers = df[["receiver_id", "user", "kind"]].drop_duplicates().set_index("receiver_id")
        assert receivers.index.is_unique
        assert receivers.index.isin(meta.index).all()
        assert (receivers.user == meta.loc[receivers.index, "user_id"]).all()
        receivers["y"] = meta.loc[receivers.index, "y"]
        effects, partners = [], []
        if name == "C36":
            for rid, r in receivers.iterrows():
                frozen = df[df.receiver_id == rid].set_index(condition)
                assert "orig" in frozen.index
                by_user = pops["swap_partners_by_user"][r.user]
                train = pops["train_partners_by_user"].get(r.user)
                partner_map = {"swap_same": by_user["same_dept_user"],
                               "swap_other": by_user["other_dept_user"],
                               "swap_train": train["user"] if train else None}
                for variant in ("swap_same", "swap_other", "swap_train"):
                    if variant not in frozen.index:
                        continue
                    partner = partner_map[variant]
                    assert partner is not None and partner != r.user
                    eligible = partner not in official
                    partners.append({"receiver_id": rid, "user": r.user, "variant": variant,
                                     "partner": partner, "eligible": eligible})
                    for view in ("behavior_only", "behavior_ses_only", "full"):
                        changes = {m: float(frozen.loc[variant, f"{m}_{view}"] -
                                            frozen.loc["orig", f"{m}_{view}"])
                                   for m in ("adapted", "base")}
                        changes["adapted_minus_base"] = changes["adapted"] - changes["base"]
                        for model, effect in changes.items():
                            effects.append({"receiver_id": rid, "user": r.user, "y": int(r.y),
                                            "eligible": eligible, "effect": effect,
                                            "endpoint": f"{view}|{variant}|{model}"})
                if {"swap_train", "swap_same"} <= set(frozen.index):
                    eligible = all(partner_map[v] not in official for v in ("swap_train", "swap_same"))
                    for view in ("behavior_only", "behavior_ses_only", "full"):
                        effect = sum(sign * float(frozen.loc["swap_train", f"{m}_{view}"] -
                                                frozen.loc["swap_same", f"{m}_{view}"])
                                     for m, sign in [("adapted", 1), ("base", -1)])
                        effects.append({"receiver_id": rid, "user": r.user, "y": int(r.y),
                                        "eligible": eligible, "effect": effect,
                                        "endpoint": f"{view}|familiarity_train_minus_same|adapted_minus_base"})
        else:
            definitions = {
                "rescue_dec": ("R", "dec"), "rescue_I2_sel": ("R", "I2_sel"),
                "rescue_I1_sel": ("R", "I1_sel"), "rescue_I2_ctl": ("R", "I2_ctl"),
                "rescue_rand_I2": ("R", "rand_I2"), "necessity_noise_I2": ("noise_I2", "O"),
                "necessity_rand_noise": ("rand_noise", "O"),
                "I2_sel_minus_I2_ctl": ("I2_ctl", "I2_sel"),
                "I2_sel_minus_rand_I2": ("rand_I2", "I2_sel"),
                "I2_sel_minus_I1_sel": ("I1_sel", "I2_sel"),
                "noise_I2_minus_rand_noise": ("noise_I2", "rand_noise"),
                "input_effect_R_minus_O": ("R", "O")}
            assert len(df) == 1320 and len(receivers) == 120
            for rid, r in receivers.iterrows():
                frozen = df[df.receiver_id == rid].set_index(condition).behavior_only
                assert frozen["zero_O"] == frozen["O"] and frozen["zero_R"] == frozen["R"]
                p = h6_partners[rid]
                assert p['user'] == r.user and p['kind'] == r.kind
                eligible = p["partner_R"] not in official and p["partner_T"] not in official
                partners.append({"receiver_id": rid, "user": r.user,
                                 "partner_R": p["partner_R"], "partner_T": p["partner_T"],
                                 "eligible": eligible})
                for endpoint, (a, b) in definitions.items():
                    effects.append({"receiver_id": rid, "user": r.user, "y": int(r.y),
                                    "eligible": eligible, "effect": float(frozen[a] - frozen[b]),
                                    "endpoint": endpoint})
        effect_df = pd.DataFrame(effects)
        assert np.isfinite(effect_df.effect).all()
        partner_df = pd.DataFrame(partners)
        flips = int(((receivers.kind == "malicious").astype(int) != receivers.y).sum())
        output["analyses"][name] = {
            "rows": len(df), "receivers": len(receivers), "users": receivers.user.nunique(),
            "label_flips": flips,
            "class_counts": {str(k): int(v) for k, v in receivers.y.value_counts().items()},
            "cross_tab": pd.crosstab(receivers.kind, receivers.y).to_dict(),
            "partner_uses": len(partner_df), "ineligible_partner_uses": int((~partner_df.eligible).sum()),
            "receivers_with_ineligible_partner": int(partner_df.loc[~partner_df.eligible, "receiver_id"].nunique()),
            "results": analyze_effects(effect_df),
            "limits": ["Same historical dictionary/feature selection, not corrected selection.",
                       "Eligible subset is a post-hoc sensitivity, not a fresh matched experiment.",
                       "No multiple-comparison correction; all endpoints are exploratory."]}
        receivers.to_csv(args.output / f"{name}_receivers.csv")
        partner_df.to_csv(args.output / f"{name}_partner_audit.csv", index=False)
        effect_df.to_csv(args.output / f"{name}_effects.csv", index=False)
    sources.append(Path(__file__))
    output["inputs"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sources]
    (args.output / "fixed_pilot_recovery.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "results"}
                      for k, v in output["analyses"].items()}, indent=2))


if __name__ == "__main__":
    main()
