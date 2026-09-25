#!/usr/bin/env python3
"""Analysis of the 2026-09-25 exploratory pilots (numpy and the standard library).

Written before any pilot output existed; endpoints as declared in the pilot
manifests and docs/HYPOTHESIS_LEDGER_2026-09-25.md. Every estimate is the mean
of user means with a user-clustered percentile bootstrap (10,000 draws, and
5,000 for the sensitivity check), seed 42. Everything here is exploratory.

  python3 scripts/analyze_pilots.py pilot1 OUT_DIR      # H1, H2 (E1-E7)
  python3 scripts/analyze_pilots.py pilot2 OUT_DIR      # H3, H7 (F1-F4)
  python3 scripts/analyze_pilots.py pilot3 OUT_DIR      # H4, on Pilot 2's tokens
  python3 scripts/analyze_pilots.py --self-test
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

S = [4596, 7693, 2302, 3673, 3455]
C = [6596, 8017, 6608, 2765, 886]
DRAWS = (10000, 5000)


def boot(per_user: Dict[str, float], draws: int, seed: int = 42) -> Tuple[float, float, float, int]:
    users = sorted(u for u, v in per_user.items() if np.isfinite(v))
    v = np.array([per_user[u] for u in users], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan"), float("nan"), len(v)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), len(v)


def user_means(values: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for (user, _), v in values.items():
        if np.isfinite(v):
            acc[user].append(v)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def summary(per_user: Dict[str, float]) -> dict:
    out = {}
    for d in DRAWS:
        est, lo, hi, n = boot(per_user, d)
        out[f"draws_{d}"] = {"estimate": est, "ci95": [lo, hi], "n_users": n,
                             "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
    out["users_positive"] = int(sum(v > 0 for v in per_user.values()))
    out["users_negative"] = int(sum(v < 0 for v in per_user.values()))
    return out


def fmt(res: dict) -> str:
    a, b = res["draws_10000"], res["draws_5000"]
    return (f"{a['estimate']:+.5f} [{a['ci95'][0]:+.5f}, {a['ci95'][1]:+.5f}] (5k: [{b['ci95'][0]:+.5f}, "
            f"{b['ci95'][1]:+.5f}]) users {a['n_users']} +{res['users_positive']}/-{res['users_negative']}")


def read_csv(path: Path) -> List[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ Pilot 1
def pilot1(rows: List[dict]) -> Tuple[dict, List[str]]:
    views = ("behavior_only", "behavior_ses_only", "profile_only", "full")
    delta = {}                                   # (cond, rid) -> {view: delta}
    meta = {}                                    # rid -> (user, kind)
    for r in rows:
        delta[(r["condition"], r["receiver_idx"])] = {v: float(r[f"delta_{v}"]) for v in views}
        meta[r["receiver_idx"]] = (r["user"], r["kind"])
    conds = sorted({c for c, _ in delta})
    rids = sorted(meta)
    singles = [c for c in conds if c.startswith("single_")]

    def per_receiver(fn: Callable[[str, str], float], kind: str, view: str) -> Dict[Tuple[str, str], float]:
        out = {}
        for rid in rids:
            if meta[rid][1] != kind:
                continue
            try:
                out[(meta[rid][0], rid)] = fn(rid, view)
            except KeyError:
                pass
        return out

    d = lambda c: (lambda rid, view: delta[(c, rid)][view])
    minus = lambda a, b: (lambda rid, view: delta[(a, rid)][view] - delta[(b, rid)][view])
    endpoints = {
        "rpU_S (selected edit alone)": d("rpU_S"),
        "E1a rpU_S - randI_S": minus("rpU_S", "randI_S"),
        "E1b rpU_S - randD_S": minus("rpU_S", "randD_S"),
        "E2a rpU_C - randI_C": minus("rpU_C", "randI_C"),
        "E2b rpU_C - randD_C": minus("rpU_C", "randD_C"),
        "E3 rpU_S - rpU_C": minus("rpU_S", "rpU_C"),
        "E4S orig_S - rpU_S - recon_S": lambda rid, v: delta[("orig_S", rid)][v] - delta[("rpU_S", rid)][v] - delta[("recon_S", rid)][v],
        "E4C orig_C - rpU_C - recon_C": lambda rid, v: delta[("orig_C", rid)][v] - delta[("rpU_C", rid)][v] - delta[("recon_C", rid)][v],
        "E5 rpU_S - rpO_S": minus("rpU_S", "rpO_S"),
        "E7 rpO_S - sum(singles)": lambda rid, v: delta[("rpO_S", rid)][v] - sum(delta[(c, rid)][v] for c in singles),
        "orig_S - orig_C (published procedure, mean-prototype policy)": minus("orig_S", "orig_C"),
    }
    res: dict = {"endpoints": {}, "conditions": {}, "E6": {}}
    lines = ["PILOT 1 (exploratory). Mean of user means, 95% user-clustered bootstrap (10k; 5k in brackets).", ""]
    for view in views:
        lines.append(f"== view {view}")
        for kind in ("malicious", "benign"):
            for name, fn in endpoints.items():
                s = summary(user_means(per_receiver(fn, kind, view)))
                res["endpoints"][f"{view}|{kind}|{name}"] = s
                lines.append(f"  {kind:<9} {name:<62} {fmt(s)}")
        for name, (a, b) in {"E6 [rpU_S - rpU_C] malicious minus benign": ("rpU_S", "rpU_C"),
                             "E6b [rpU_S - randI_S] malicious minus benign": ("rpU_S", "randI_S")}.items():
            mal = user_means(per_receiver(minus(a, b), "malicious", view))
            ben = user_means(per_receiver(minus(a, b), "benign", view))
            s = summary({u: mal[u] - ben[u] for u in mal if u in ben})
            res["E6"][f"{view}|{name}"] = s
            lines.append(f"  paired    {name:<62} {fmt(s)}")
        lines.append("")
    for view in ("behavior_only", "profile_only"):
        for kind in ("malicious", "benign"):
            for c in conds:
                s = summary(user_means(per_receiver(d(c), kind, view)))
                res["conditions"][f"{view}|{kind}|{c}"] = s
    lines.append("All conditions, patched minus unedited (behavior_only | profile_only), malicious days:")
    for c in conds:
        b = res["conditions"][f"behavior_only|malicious|{c}"]["draws_10000"]
        p = res["conditions"][f"profile_only|malicious|{c}"]["draws_10000"]
        lines.append(f"  {c:<14} beh {b['estimate']:+.5f} [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]   "
                     f"prof {p['estimate']:+.5f} [{p['ci95'][0]:+.5f}, {p['ci95'][1]:+.5f}]")
    # H1 stopping rule (declared in the ledger)
    sel = res["endpoints"]["behavior_only|malicious|rpU_S (selected edit alone)"]["draws_10000"]["estimate"]
    rule = {}
    for key in ("E1a rpU_S - randI_S", "E1b rpU_S - randD_S"):
        e = res["endpoints"][f"behavior_only|malicious|{key}"]["draws_10000"]
        rule[key] = {"within_20pct_of_selected_effect": bool(abs(e["estimate"]) <= 0.2 * abs(sel)),
                     "interval_covers_zero": not e["excludes_zero"]}
        rule[key]["stop_feature_specific_branch"] = rule[key]["within_20pct_of_selected_effect"] and rule[key]["interval_covers_zero"]
    res["H1_stopping_rule"] = rule
    lines += ["", f"H1 stopping rule (selected effect {sel:+.5f}): {json.dumps(rule)}"]
    return res, lines


# ------------------------------------------------------------------ Pilot 4 (H8)
def pilot4(rows: List[dict]) -> Tuple[dict, List[str]]:
    views = ("behavior_only", "behavior_ses_only", "profile_only", "full")
    delta, meta = {}, {}
    for r in rows:
        delta[(r["condition"], r["receiver_idx"])] = {v: float(r[f"delta_{v}"]) for v in views}
        meta[r["receiver_idx"]] = (r["user"], r["kind"])
    conds = sorted({c for c, _ in delta})
    baselines = [c for c in conds if c not in ("zero", "rpU_S", "randI_S", "randD_S")]

    def per_receiver(fn, kind, view):
        out = {}
        for rid, (u, k) in meta.items():
            if k == kind:
                try:
                    out[(u, rid)] = fn(rid, view)
                except KeyError:
                    pass
        return out

    d = lambda c: (lambda rid, view: delta[(c, rid)][view])
    minus = lambda a, b: (lambda rid, view: delta[(a, rid)][view] - delta[(b, rid)][view])
    res: dict = {"endpoints": {}, "paired": {}}
    lines = ["PILOT 4 / H8 (exploratory). Same tokens and per-token edit sizes as the SAE edit (rpU_S); only the",
             "direction differs. Mean of user means, 95% user-clustered bootstrap (10k; 5k in brackets).", ""]
    for view in views:
        lines.append(f"== view {view}")
        for kind in ("malicious", "benign"):
            for c in ("rpU_S", "randI_S", "randD_S") + tuple(baselines):
                s = summary(user_means(per_receiver(d(c), kind, view)))
                res["endpoints"][f"{view}|{kind}|{c} (edit alone)"] = s
                lines.append(f"  {kind:<9} {c + ' (edit alone)':<40} {fmt(s)}")
            for c in baselines:
                for ref, tag in (("rpU_S", "E8a"), ("randI_S", "E8b"), ("randD_S", "E8b'")):
                    s = summary(user_means(per_receiver(minus(c, ref), kind, view)))
                    res["endpoints"][f"{view}|{kind}|{tag} {c} - {ref}"] = s
                    lines.append(f"  {kind:<9} {tag + ' ' + c + ' - ' + ref:<40} {fmt(s)}")
            s = summary(user_means(per_receiver(minus("rpU_S", "randI_S"), kind, view)))
            res["endpoints"][f"{view}|{kind}|rpU_S - randI_S"] = s
            lines.append(f"  {kind:<9} {'rpU_S - randI_S':<40} {fmt(s)}")
        for c in baselines + ["rpU_S"]:
            mal = user_means(per_receiver(minus(c, "randI_S"), "malicious", view))
            ben = user_means(per_receiver(minus(c, "randI_S"), "benign", view))
            s = summary({u: mal[u] - ben[u] for u in mal if u in ben})
            res["paired"][f"{view}|E8c [{c} - randI_S] malicious minus benign"] = s
            lines.append(f"  paired    {'E8c [' + c + ' - randI_S] mal - ben':<40} {fmt(s)}")
        lines.append("")
    return res, lines


# ------------------------------------------------------------------ Pilot 2
def pilot2(rows: List[dict]) -> Tuple[dict, List[str]]:
    views = ("behavior_only", "behavior_ses_only", "full")
    by = {(r["receiver_idx"], r["variant"]): r for r in rows}
    meta = {r["receiver_idx"]: (r["user"], r["kind"]) for r in rows}
    variants = ("swap_same", "swap_other", "swap_train")

    def dswap(rid, v, model, view):
        return float(by[(rid, v)][f"{model}_{view}"]) - float(by[(rid, "orig")][f"{model}_{view}"])

    def collect(fn, kind) -> Dict[Tuple[str, str], float]:
        out = {}
        for rid, (u, k) in meta.items():
            if k != kind:
                continue
            try:
                out[(u, rid)] = fn(rid)
            except KeyError:
                pass
        return out

    res: dict = {}
    lines = ["PILOT 2 (exploratory). Behaviour-loss change when the profile lines are swapped (swap minus",
             "original); adapted = adapter on, base = adapter off, same inputs. Mean of user means, 95% user bootstrap.", ""]
    for view in views:
        lines.append(f"== view {view}")
        for v in variants:
            for kind in ("malicious", "benign"):
                for label, fn in (("adapted", lambda rid: dswap(rid, v, "adapted", view)),
                                  ("base", lambda rid: dswap(rid, v, "base", view)),
                                  ("F1 adapted - base", lambda rid: dswap(rid, v, "adapted", view) - dswap(rid, v, "base", view))):
                    s = summary(user_means(collect(fn, kind)))
                    res[f"{view}|{v}|{kind}|{label}"] = s
                    lines.append(f"  {v:<10} {kind:<9} {label:<18} {fmt(s)}")
            for label, fn in (("F2 adapted, malicious - benign", lambda rid: dswap(rid, v, "adapted", view)),
                              ("F2b (adapted - base), malicious - benign",
                               lambda rid: dswap(rid, v, "adapted", view) - dswap(rid, v, "base", view))):
                mal, ben = user_means(collect(fn, "malicious")), user_means(collect(fn, "benign"))
                s = summary({u: mal[u] - ben[u] for u in mal if u in ben})
                res[f"{view}|{v}|{label}"] = s
                lines.append(f"  {v:<10} paired    {label:<18} {fmt(s)}")
        for kind in ("malicious", "benign"):
            fam = lambda rid, m: dswap(rid, "swap_train", m, view) - dswap(rid, "swap_same", m, view)
            for label, fn in (("F4 familiarity, adapted", lambda rid: fam(rid, "adapted")),
                              ("F4 familiarity, base", lambda rid: fam(rid, "base")),
                              ("F4 familiarity, adapted - base", lambda rid: fam(rid, "adapted") - fam(rid, "base"))):
                s = summary(user_means(collect(fn, kind)))
                res[f"{view}|{kind}|{label}"] = s
                lines.append(f"  train-same {kind:<9} {label:<30} {fmt(s)}")
        lines.append("")
    # F3: selected-feature activation change on SES tokens under swap, minus controls
    orig_mean = {f: np.mean([float(by[(rid, "orig")][f"mean_act_{f}"]) for rid in meta]) for f in S + C}
    lines.append("== F3 feature activation on session tokens, swap minus original (selected minus control)")
    for v in variants:
        for kind in ("malicious", "benign"):
            def f3(rid, norm):
                ch = lambda f: (float(by[(rid, v)][f"mean_act_{f}"]) - float(by[(rid, "orig")][f"mean_act_{f}"])) / (
                    orig_mean[f] if norm and orig_mean[f] > 0 else 1.0)
                sel = [ch(f) for f in S if not norm or orig_mean[f] > 0]
                ctl = [ch(f) for f in C if not norm or orig_mean[f] > 0]
                return float(np.mean(sel) - np.mean(ctl))
            for label, norm in (("F3 raw", False), ("F3 relative to feature mean (secondary)", True)):
                s = summary(user_means(collect(lambda rid: f3(rid, norm), kind)))
                res[f"F3|{v}|{kind}|{label}"] = s
                lines.append(f"  {v:<10} {kind:<9} {label:<40} {fmt(s)}")
    per_feature = {}
    for f in S + C:
        for v in variants:
            vals = [float(by[(rid, v)][f"mean_act_{f}"]) - float(by[(rid, "orig")][f"mean_act_{f}"])
                    for rid in meta if (rid, v) in by]
            per_feature[f"{f}|{v}"] = {"mean_change": float(np.mean(vals)), "orig_mean": float(orig_mean[f])}
    res["F3_per_feature"] = per_feature
    return res, lines


# ------------------------------------------------------------------ Pilot 3 (H4) on Pilot 2's tokens
def stratified_diff(loss: np.ndarray, active: np.ndarray, strata: np.ndarray) -> float:
    """Mean loss when active minus when inactive, within strata (next-token id), harmonic-count weights."""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        a, i = active & m, ~active & m
        na, ni = int(a.sum()), int(i.sum())
        if na == 0 or ni == 0:
            continue
        w = na * ni / (na + ni)
        num += w * (loss[a].mean() - loss[i].mean())
        den += w
    return num / den if den > 0 else float("nan")


def pilot3(npz: dict, rows: List[dict]) -> Tuple[dict, List[str]]:
    meta = {r["receiver_idx"]: (r["user"], r["kind"]) for r in rows if r["variant"] == "orig"}
    feats = [int(f) for f in npz["feats"]]
    pooled = defaultdict(lambda: defaultdict(list))     # (user, kind) -> key -> arrays
    for rid in [str(x) for x in npz["receivers"]]:
        cls, act, ids = npz[f"{rid}__classes"], npz[f"{rid}__act"], npz[f"{rid}__ids"]
        la, lb = npz[f"{rid}__nll_adapted"], npz[f"{rid}__nll_base"]
        n = len(la)                                        # positions 0..n-1 predict tokens 1..n
        keep = (cls[:n] == "SES") & (cls[1:n + 1] == "SES")
        u, k = meta[rid]
        pooled[(u, k)]["act"].append(act[:n][keep]); pooled[(u, k)]["next"].append(ids[1:n + 1][keep])
        pooled[(u, k)]["la"].append(la[keep]); pooled[(u, k)]["lb"].append(lb[keep])
    res: dict = {}
    lines = ["PILOT 3 / H4 (exploratory, CPU). Session positions predicting a session token. For each feature:",
             "next-token loss when the feature is active minus when inactive, within the same next-token id,",
             "per user and kind; mean of user means, 95% user bootstrap. 'gap' = adapted loss - base loss.", ""]
    for kind in ("malicious", "benign"):
        per = defaultdict(dict)                            # stat -> user -> value
        for (u, k), d in pooled.items():
            if k != kind:
                continue
            act = np.concatenate(d["act"]); nxt = np.concatenate(d["next"])
            la = np.concatenate(d["la"]); lb = np.concatenate(d["lb"])
            for j, f in enumerate(feats):
                on = act[:, j] > 0
                if on.sum() < 5:
                    continue
                for name, loss in (("adapted", la), ("base", lb), ("gap", la - lb)):
                    per[f"{f}|{name}"][u] = stratified_diff(loss, on, nxt)
            for name in ("adapted", "base", "gap"):
                sv = [per[f"{f}|{name}"][u] for f in S if u in per[f"{f}|{name}"]]
                cv = [per[f"{f}|{name}"][u] for f in C if u in per[f"{f}|{name}"]]
                if sv and cv:
                    per[f"sel-ctrl|{name}"][u] = float(np.nanmean(sv) - np.nanmean(cv))
                if sv:
                    per[f"sel|adapted-base"][u] = float(np.nanmean([per[f"{f}|adapted"][u] for f in S if u in per[f"{f}|adapted"]])
                                                        - np.nanmean([per[f"{f}|base"][u] for f in S if u in per[f"{f}|base"]]))
        for key in sorted(per):
            s = summary({u: v for u, v in per[key].items() if np.isfinite(v)})
            res[f"{kind}|{key}"] = s
            lines.append(f"  {kind:<9} {key:<22} {fmt(s)}")
        lines.append("")
    return res, lines


# ------------------------------------------------------------------ self-test
def self_test() -> int:
    rng = np.random.default_rng(1)
    rows = []
    effects = {"zero": 0, "rpU_S": 0.010, "randI_S": 0.002, "randD_S": 0.004, "rpU_C": 0.003, "randI_C": 0.003,
               "randD_C": 0.003, "recon_S": 0.05, "recon_C": 0.05, "orig_S": 0.065, "orig_C": 0.053, "rpO_S": 0.008,
               "rpO_C": 0.002, "single_4596": 0.002, "single_7693": 0.0, "single_2302": 0.003, "single_3673": 0.001,
               "single_3455": 0.001}
    for u in range(12):
        for j in range(6):
            for kind in ("malicious", "benign"):
                rid = f"{u}_{j}_{kind}"
                noise = rng.normal(0, 0.001)
                for c, e in effects.items():
                    val = (e if kind == "malicious" else e / 2) + noise
                    row = {"condition": c, "receiver_idx": rid, "user": f"U{u}", "kind": kind}
                    for v in ("behavior_only", "behavior_ses_only", "profile_only", "full"):
                        row[f"delta_{v}"] = str(val if v != "profile_only" else 0.0)
                    rows.append(row)
    res, _ = pilot1(rows)
    e1 = res["endpoints"]["behavior_only|malicious|E1a rpU_S - randI_S"]["draws_10000"]["estimate"]
    e4 = res["endpoints"]["behavior_only|malicious|E4S orig_S - rpU_S - recon_S"]["draws_10000"]["estimate"]
    e7 = res["endpoints"]["behavior_only|malicious|E7 rpO_S - sum(singles)"]["draws_10000"]["estimate"]
    e6 = res["E6"]["behavior_only|E6 [rpU_S - rpU_C] malicious minus benign"]["draws_10000"]["estimate"]
    checks = {"E1 = 0.008": abs(e1 - 0.008) < 1e-9,
              "E4 = 0.005 (one shared noise term remains)": abs(e4 - 0.005) < 5e-4,
              "E7 = 0.001 (noise enters once per single)": abs(e7 - (0.008 - 0.007)) < 0.002,
              "E6 = 0.0035": abs(e6 - 0.0035) < 1e-9}
    loss = np.array([1.0, 1.0, 3.0, 3.0, 2.0, 2.0]); active = np.array([1, 0, 1, 0, 1, 1], bool)
    checks["stratified diff ignores between-field differences"] = abs(stratified_diff(loss, active, np.array([1, 1, 2, 2, 3, 3]))) < 1e-12
    loss2 = np.array([2.0, 1.0, 4.0, 3.0])
    checks["stratified diff finds within-field difference"] = abs(stratified_diff(loss2, np.array([1, 0, 1, 0], bool), np.array([1, 1, 2, 2])) - 1.0) < 1e-12
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)
    return 0 if all(checks.values()) else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    which, out = sys.argv[1], Path(sys.argv[2])
    if which == "pilot1":
        res, lines = pilot1(read_csv(out / "pilot1_rows.csv"))
    elif which == "pilot2":
        res, lines = pilot2(read_csv(out / "pilot2_rows.csv"))
    elif which == "pilot3":
        res, lines = pilot3(dict(np.load(out / "pilot2_orig_tokens.npz")), read_csv(out / "pilot2_rows.csv"))
    elif which == "pilot4":
        res, lines = pilot4(read_csv(out / "pilot4_rows.csv"))
    else:
        raise SystemExit(__doc__)
    (out / f"analysis_{which}.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / f"analysis_{which}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
