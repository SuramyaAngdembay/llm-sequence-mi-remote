#!/usr/bin/env python3
"""Pre-specified endpoints for token-class SAE interventions (written 2026-09-24,
before the corrected TWOS run and the CERT package-4 run produced results).

Reads `token_delta_sae_causal_candidate_rows.csv` from a run made with
`--token-class-schema` and reports, SEPARATELY for every context mode, donor
type and patch strength (alpha), never pooled:

  * the unpatched base (fresh recomputation, same code path as the patches);
  * each arm's absolute effect: patched minus base, per token-class view;
  * the selected-minus-control difference;
  * when alpha = 0 is present, the reconstruction-only effect of each arm and
    the edit effect net of reconstruction, (alpha - 0) per arm, and their
    difference;
  * the direct paired profile-versus-behaviour contrast of the
    selected-minus-control difference;
  * the historical best-candidate donor difference-in-differences on the full
    score, for continuity with the published endpoint only.

Unit and weighting (declared, not chosen after results):
  receiver example = mean over ALL candidate donors of that donor type (not the
  best one); complete case = receivers present in both arms; primary aggregate
  = mean of receiver-user means (every user weighs the same), with a cluster
  bootstrap over users; the receiver-weighted mean is printed as secondary.

What the output does and does not license:
  * a near-zero selected-minus-control difference means the two patches have
    SIMILAR effects, not that neither changes anything: read both arms;
  * an interval crossing zero is inconclusive, not evidence of equivalence;
    equivalence is reported only against an explicit --margin (TOST at 90%);
  * a monotone dose-response across alphas does not rule out confounding.

Stdlib only.
Usage:
  python3 scripts/analyze_intervention_endpoints.py <run-dir> [--alpha 1.0] [--margin M] [--json OUT]
  python3 scripts/analyze_intervention_endpoints.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

VIEWS = ("full", "profile_only", "behavior_only", "behavior_ses_only", "psy_only", "day_only")


def user_of(example_id: str) -> str:
    return example_id.split(":")[0]


def cluster_boot(by_user: Dict[str, float], draws: int = 10000, seed: int = 42,
                 level: float = 0.95) -> Tuple[float, float]:
    users = sorted(by_user)
    if len(users) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    vals = [by_user[u] for u in users]
    n = len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    lo = (1 - level) / 2
    return ms[int(lo * draws)], ms[min(int((1 - lo) * draws), draws - 1)]


def load_rows(run_dir: Path) -> List[dict]:
    rows = list(csv.DictReader((run_dir / "token_delta_sae_causal_candidate_rows.csv").open()))
    if not rows:
        raise SystemExit("no candidate rows")
    if "delta_full" not in rows[0] or "base_full" not in rows[0]:
        raise SystemExit("run has no per-class columns; it was not made with --token-class-schema")
    stale = [r for r in rows if r.get("base_is_recomputed") not in (None, "", "1")]
    if stale:
        raise SystemExit(f"{len(stale)} rows use the cached base, not the fresh recomputation")
    return rows


def receiver_means(rows: Sequence[dict], key: str) -> Dict[Tuple[str, str], float]:
    """(feature_set, receiver_example_id) -> mean over all candidate donors."""
    acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        acc[(r["feature_set"], r["receiver_example_id"])].append(float(r[key]))
    return {k: st.fmean(v) for k, v in acc.items()}


def user_mean(per_receiver: Dict[str, float]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for rid, v in per_receiver.items():
        acc[user_of(rid)].append(v)
    return {u: st.fmean(v) for u, v in acc.items()}


def summarize(per_receiver: Dict[str, float], margin: Optional[float]) -> dict:
    by_user = user_mean(per_receiver)
    lo, hi = cluster_boot(by_user)
    out = {
        "primary_mean_of_user_means": st.fmean(by_user.values()) if by_user else float("nan"),
        "ci95_cluster_users": [lo, hi],
        "secondary_receiver_weighted_mean": st.fmean(per_receiver.values()) if per_receiver else float("nan"),
        "n_receivers": len(per_receiver),
        "n_users": len(by_user),
        "users_positive": sum(v > 0 for v in by_user.values()),
        "users_negative": sum(v < 0 for v in by_user.values()),
    }
    if margin is not None:
        lo90, hi90 = cluster_boot(by_user, level=0.90)
        out["equivalence_margin"] = margin
        out["ci90_cluster_users"] = [lo90, hi90]
        out["within_margin_tost"] = bool(-margin < lo90 and hi90 < margin)
    return out


def arm_table(rows: Sequence[dict], top: str, ctrl: str, margin: Optional[float]) -> dict:
    """Absolute effects of each arm, their difference and the base, per view."""
    out: Dict[str, dict] = {}
    for view in VIEWS:
        if f"delta_{view}" not in rows[0]:
            continue
        d = receiver_means(rows, f"delta_{view}")
        b = receiver_means(rows, f"base_{view}")
        receivers = sorted({rid for (fs, rid) in d if fs == top} & {rid for (fs, rid) in d if fs == ctrl})
        sel = {rid: d[(top, rid)] for rid in receivers}
        con = {rid: d[(ctrl, rid)] for rid in receivers}
        diff = {rid: sel[rid] - con[rid] for rid in receivers}
        base = {rid: b[(top, rid)] for rid in receivers}
        out[view] = {
            "base_unpatched": summarize(base, None),
            "selected_delta": summarize(sel, None),
            "control_delta": summarize(con, None),
            "selected_minus_control": summarize(diff, margin),
            "_per_receiver_diff": diff,
            "_per_receiver_sel": sel,
            "_per_receiver_con": con,
        }
    return out


def analyze(rows: List[dict], alpha: float, margin: Optional[float]) -> dict:
    sets = sorted({r["feature_set"] for r in rows})
    ctrl = next((s for s in sets if s.startswith("control")), None)
    tops = [s for s in sets if s != ctrl]
    if ctrl is None or not tops:
        raise SystemExit(f"need a top set and a control set, got {sets}")
    top = tops[0]
    alphas = sorted({float(r["alpha"]) for r in rows})
    result: dict = {"top_set": top, "control_set": ctrl, "alphas_present": alphas, "primary_alpha": alpha,
                    "by_context_and_donor": {}}
    for mode in sorted({r["context_mode"] for r in rows}):
        for donor in sorted({r["donor_type"] for r in rows}):
            sub = [r for r in rows if r["context_mode"] == mode and r["donor_type"] == donor]
            at = [r for r in sub if float(r["alpha"]) == alpha]
            if not at:
                continue
            entry = {"at_alpha": arm_table(at, top, ctrl, margin)}
            zero = [r for r in sub if float(r["alpha"]) == 0.0]
            if zero and alpha != 0.0:
                z = arm_table(zero, top, ctrl, None)
                entry["reconstruction_only_alpha0"] = z
                net = {}
                for view in entry["at_alpha"]:
                    if view not in z:
                        continue
                    a1, a0 = entry["at_alpha"][view], z[view]
                    shared = sorted(set(a1["_per_receiver_sel"]) & set(a0["_per_receiver_sel"]))
                    sel_net = {rid: a1["_per_receiver_sel"][rid] - a0["_per_receiver_sel"][rid] for rid in shared}
                    con_net = {rid: a1["_per_receiver_con"][rid] - a0["_per_receiver_con"][rid] for rid in shared}
                    net[view] = {
                        "selected_edit_net_of_reconstruction": summarize(sel_net, None),
                        "control_edit_net_of_reconstruction": summarize(con_net, None),
                        "difference": summarize({rid: sel_net[rid] - con_net[rid] for rid in shared}, margin),
                    }
                entry["edit_net_of_reconstruction"] = net
            t = entry["at_alpha"]
            if "profile_only" in t and "behavior_only" in t:
                p, bh = t["profile_only"]["_per_receiver_diff"], t["behavior_only"]["_per_receiver_diff"]
                shared = sorted(set(p) & set(bh))
                entry["profile_minus_behavior_of_selected_minus_control"] = summarize(
                    {rid: p[rid] - bh[rid] for rid in shared}, None)
            result["by_context_and_donor"][f"{mode} | {donor} donors"] = entry
        # historical endpoint, full score only: best candidate over donors AND alphas
        best: Dict[Tuple[str, str, str], float] = {}
        for r in rows:
            if r["context_mode"] != mode:
                continue
            k = (r["feature_set"], r["donor_type"], r["receiver_example_id"])
            v = float(r["delta_full"])
            best[k] = min(v, best.get(k, v))
        recv = sorted({k[2] for k in best})
        did = {}
        for rid in recv:
            need = [(top, "anomalous", rid), (top, "benign", rid), (ctrl, "anomalous", rid), (ctrl, "benign", rid)]
            if all(k in best for k in need):
                did[rid] = (best[need[0]] - best[need[1]]) - (best[need[2]] - best[need[3]])
        result.setdefault("historical_best_candidate_did_full", {})[mode] = summarize(did, None)
    return result


def strip_private(o):
    if isinstance(o, dict):
        return {k: strip_private(v) for k, v in o.items() if not k.startswith("_")}
    return o


def print_report(res: dict) -> None:
    fmt = lambda s: (f"{s['primary_mean_of_user_means']:+.6f} [{s['ci95_cluster_users'][0]:+.6f}, "
                     f"{s['ci95_cluster_users'][1]:+.6f}] users={s['n_users']} recv={s['n_receivers']}")
    print(f"top={res['top_set']} control={res['control_set']} alphas present={res['alphas_present']} "
          f"primary alpha={res['primary_alpha']}")
    for key, e in res["by_context_and_donor"].items():
        print(f"\n=== {key} ===")
        for view, t in e["at_alpha"].items():
            print(f"  {view:<18} base {t['base_unpatched']['primary_mean_of_user_means']:.5f}"
                  f" | selected {fmt(t['selected_delta'])}\n{'':<21}| control  {fmt(t['control_delta'])}"
                  f"\n{'':<21}| sel-ctrl {fmt(t['selected_minus_control'])}")
            if "within_margin_tost" in t["selected_minus_control"]:
                print(f"{'':<21}| TOST within ±{t['selected_minus_control']['equivalence_margin']}: "
                      f"{t['selected_minus_control']['within_margin_tost']}")
        for view, n in e.get("edit_net_of_reconstruction", {}).items():
            print(f"  [net of alpha=0] {view:<14} sel {fmt(n['selected_edit_net_of_reconstruction'])}"
                  f"\n{'':<32}ctrl {fmt(n['control_edit_net_of_reconstruction'])}"
                  f"\n{'':<32}diff {fmt(n['difference'])}")
        if "profile_minus_behavior_of_selected_minus_control" in e:
            print(f"  profile-minus-behaviour of (sel-ctrl): {fmt(e['profile_minus_behavior_of_selected_minus_control'])}")
    for mode, s in res.get("historical_best_candidate_did_full", {}).items():
        print(f"\n[historical endpoint, continuity only] {mode}: best-candidate donor DiD (full) {fmt(s)}")


def self_test() -> int:
    """Known-answer check: build rows whose endpoints are fixed by construction."""
    rows = []
    for u in range(4):
        for e in range(2):
            rid = f"U{u}:{e}"
            for fs, eff in (("top5", -0.010), ("control5_active", -0.004)):
                for donor, shift in (("benign", 0.0), ("anomalous", 0.003)):
                    for a in (0.0, 1.0):
                        for dn in range(3):
                            recon = 0.001                     # reconstruction-only effect, both arms
                            edit = (eff + shift) * a          # edit effect scales with alpha
                            beh = recon + edit + 0.0001 * dn  # donors differ a little
                            prof = recon + 0.5 * edit
                            full = 0.8 * beh + 0.2 * prof
                            row = {"context_mode": "team", "feature_set": fs, "donor_type": donor, "alpha": a,
                                   "receiver_example_id": rid, "donor_example_id": f"D{dn}", "base_is_recomputed": "1"}
                            for v, dv in (("full", full), ("behavior_only", beh), ("profile_only", prof)):
                                row[f"delta_{v}"] = dv
                                row[f"base_{v}"] = 1.0
                            rows.append(row)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "token_delta_sae_causal_candidate_rows.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        res = analyze(load_rows(Path(td)), alpha=1.0, margin=0.002)
    e = res["by_context_and_donor"]["team | benign donors"]
    beh = e["at_alpha"]["behavior_only"]
    checks = {
        "selected absolute effect = recon + edit + mean donor offset":
            abs(beh["selected_delta"]["primary_mean_of_user_means"] - (0.001 - 0.010 + 0.0001)) < 1e-12,
        "control absolute effect": abs(beh["control_delta"]["primary_mean_of_user_means"] - (0.001 - 0.004 + 0.0001)) < 1e-12,
        "selected minus control = -0.006": abs(beh["selected_minus_control"]["primary_mean_of_user_means"] + 0.006) < 1e-12,
        "reconstruction cancels in the net edit difference":
            abs(e["edit_net_of_reconstruction"]["behavior_only"]["difference"]["primary_mean_of_user_means"] + 0.006) < 1e-12,
        "net selected edit excludes reconstruction":
            abs(e["edit_net_of_reconstruction"]["behavior_only"]["selected_edit_net_of_reconstruction"]["primary_mean_of_user_means"] + 0.010) < 1e-12,
        "profile-minus-behaviour of the difference = +0.003":
            abs(e["profile_minus_behavior_of_selected_minus_control"]["primary_mean_of_user_means"] - 0.003) < 1e-12,
        "a -0.006 difference is not within a ±0.002 margin": beh["selected_minus_control"]["within_margin_tost"] is False,
        "donor types are reported separately": set(res["by_context_and_donor"]) == {"team | benign donors", "team | anomalous donors"},
        "historical DiD = (-0.007 - -0.010) - (-0.001 - -0.004) = 0 on full":
            abs(res["historical_best_candidate_did_full"]["team"]["primary_mean_of_user_means"]) < 1e-12,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--alpha", type=float, default=1.0, help="primary fixed patch strength")
    ap.add_argument("--margin", type=float, default=None, help="equivalence margin; TOST reported only if given")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir is None:
        ap.error("run_dir is required")
    res = analyze(load_rows(args.run_dir), args.alpha, args.margin)
    print_report(res)
    if args.json:
        args.json.write_text(json.dumps(strip_private(res), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
