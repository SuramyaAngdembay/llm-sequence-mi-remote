#!/usr/bin/env python3
"""Analysis of the H6 pilot (EXPLORATORY), written before any H6 output existed.

Follows docs/H6_PROFILE_PATCHING_PROTOCOL_2026-09-27.md. Behaviour-only target
loss; mean of user means, attack and benign receivers separately, with a
user-clustered percentile bootstrap (10,000 and 5,000 draws); the paired
within-user attack-minus-benign difference; absolute rescue first; a rescue
fraction only where the input effect is at least 0.05 nats with an interval
excluding zero.

  python3 scripts/analyze_h6.py OUT_DIR
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pilots import fmt, summary, user_means  # noqa: E402

LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34)
S = [4596, 7693, 2302, 3673, 3455]; C = [6596, 8017, 6608, 2765, 886]


def main():
    out = Path(sys.argv[1])
    rows = list(csv.DictReader((out / "h6_rows.csv").open()))
    L = {(r["condition"], r["receiver_id"]): float(r["behavior_only"]) for r in rows}
    meta = {r["receiver_id"]: (r["user"], r["kind"]) for r in rows}
    conds = sorted({r["condition"] for r in rows})
    res, lines = {}, ["H6 (exploratory). Behaviour-only target loss, nats per token; mean of user means, 95% user bootstrap (10k; 5k).", ""]

    def stat(fn, kind):
        per = {}
        for rid, (u, k) in meta.items():
            if k == kind:
                try:
                    per[(u, rid)] = fn(rid)
                except KeyError:
                    pass
        return user_means(per)

    def report(name, fn):
        um = {k: stat(fn, k) for k in ("malicious", "benign")}
        for k in ("malicious", "benign"):
            s = summary(um[k]); res[f"{name}|{k}"] = dict(s, per_user=um[k])
            lines.append(f"  {k:<9} {name:<44} {fmt(s)}")
        s = summary({u: um["malicious"][u] - um["benign"][u] for u in um["malicious"] if u in um["benign"]})
        res[f"{name}|attack-minus-benign"] = s
        lines.append(f"  {'paired':<9} {name + ' [attack - benign]':<44} {fmt(s)}")
        return um

    # implementation checks
    chk = {c: max(abs(L[(c, rid)] - L[("R_on", rid)]) for rid in meta) for c in ("zero", "self_sess@26", "self_prof@10") if any(k[0] == c for k in L)}
    res["checks_max_abs_vs_R_on"] = chk
    lines.append(f"Implementation checks (max |loss - R_on|): {json.dumps(chk)}\n")
    lines.append("== input effects (variant minus original)")
    eff = {}
    for v in ("R", "D", "P", "T"):
        for ad in ("on", "off"):
            eff[(v, ad)] = report(f"input {v} - O, adapter {ad}", lambda rid, v=v, ad=ad: L[(f"{v}_{ad}", rid)] - L[(f"O_{ad}", rid)])
        report(f"input {v} - O, adapted minus base", lambda rid, v=v: (L[(f"{v}_on", rid)] - L[("O_on", rid)]) - (L[(f"{v}_off", rid)] - L[("O_off", rid)]))
    lines.append("\n== rescue of the R input effect: L(R_on) - L(R_on + patch); positive = moves toward the original")
    rescue_names = [c for c in conds if "@" in c and not c.startswith("self")] + \
                   [c for c in ("sel_sess", "sel_all", "ctrl_sess", "ctrl_all", "rand_sess", "fulldelta_sess", "fulldelta_all") if c in conds]
    resc = {}
    for c in rescue_names:
        resc[c] = report(f"rescue {c}", lambda rid, c=c: L[("R_on", rid)] - L[(c, rid)])
    lines.append("\n== matched-control and mediation contrasts (paired)")
    for l in LAYERS:
        for g in ("prof", "sess"):
            a, b = f"resid_{g}@{l}<-O", f"resid_{g}@{l}<-T"
            if a in conds and b in conds:
                report(f"{a} minus {b}", lambda rid, a=a, b=b: L[(b, rid)] - L[(a, rid)])
    for a, b in (("sel_sess", "ctrl_sess"), ("sel_sess", "rand_sess"), ("sel_all", "ctrl_all"), ("fulldelta_sess", "sel_sess")):
        if a in conds and b in conds:
            report(f"rescue {a} minus rescue {b}", lambda rid, a=a, b=b: L[(b, rid)] - L[(a, rid)])
    # rescue fractions, only with a large, well-determined denominator
    lines.append("\n== rescue fractions (only where the adapter-on R input effect is >= 0.05 with an interval excluding zero)")
    for k in ("malicious", "benign"):
        e = summary(eff[("R", "on")][k])["draws_10000"]
        ok = e["estimate"] >= 0.05 and e["excludes_zero"]
        res[f"fraction_denominator_ok|{k}"] = bool(ok)
        if not ok:
            lines.append(f"  {k}: denominator {e['estimate']:+.4f} {e['ci95']} not eligible"); continue
        rng = np.random.default_rng(42)
        users = sorted(eff[("R", "on")][k])
        for c in rescue_names:
            um = resc[c][k]
            us = [u for u in users if u in um]
            num = np.array([um[u] for u in us]); den = np.array([eff[("R", "on")][k][u] for u in us])
            idx = rng.integers(0, len(us), size=(10000, len(us)))
            fr = num[idx].mean(1) / den[idx].mean(1)
            res[f"fraction {c}|{k}"] = {"estimate": float(num.mean() / den.mean()), "ci95": [float(np.quantile(fr, 0.025)), float(np.quantile(fr, 0.975))]}
        best = sorted(((res[f"fraction {c}|{k}"]["estimate"], c) for c in rescue_names), reverse=True)[:6]
        lines.append(f"  {k}: " + "; ".join(f"{c} {v:.2f} [{res[f'fraction {c}|{k}']['ci95'][0]:.2f}, {res[f'fraction {c}|{k}']['ci95'][1]:.2f}]" for v, c in best))
    # layer profile: where does restoring session positions overtake restoring profile positions?
    lines.append("\n== layer profile (attack | benign), absolute rescue point estimates")
    lines.append("  layer   resid_prof<-O   resid_sess<-O   resid_prof<-T   resid_sess<-T   attn_sess   mlp_sess")
    for l in LAYERS:
        cells = []
        for c in (f"resid_prof@{l}<-O", f"resid_sess@{l}<-O", f"resid_prof@{l}<-T", f"resid_sess@{l}<-T", f"attn_sess@{l}", f"mlp_sess@{l}"):
            m = res.get(f"rescue {c}|malicious", {}).get("draws_10000", {}).get("estimate", float("nan"))
            b = res.get(f"rescue {c}|benign", {}).get("draws_10000", {}).get("estimate", float("nan"))
            cells.append(f"{m:+.4f}|{b:+.4f}")
        lines.append(f"  {l:>5}   " + "   ".join(cells))
    cross = {}
    for k in ("malicious", "benign"):
        for l in LAYERS:
            sp = res[f"rescue resid_sess@{l}<-O|{k}"]["draws_10000"]["estimate"]
            pp = res[f"rescue resid_prof@{l}<-O|{k}"]["draws_10000"]["estimate"]
            if sp > pp:
                cross[k] = l; break
    res["crossover_layer_point_estimate"] = cross
    lines.append(f"\nFirst layer where session-position rescue exceeds profile-position rescue (point estimates): {cross}")
    # token-level selected/control code changes under the swap (layer 26, aligned session positions)
    npz = out / "h6_token_codes.npz"
    if npz.exists():
        z = np.load(npz)
        feats = list(z["feats"])
        ids = sorted({k.split("__")[0] for k in z.files if "__" in k})
        signed, absol, base = defaultdict(list), defaultdict(list), defaultdict(list)
        for rid in ids:
            o, r = z[f"{rid}__O_sess"], z[f"{rid}__R_sess"]
            if o.shape != r.shape:
                continue
            for j, f in enumerate(feats):
                signed[f].append(float((r[:, j] - o[:, j]).mean())); absol[f].append(float(np.abs(r[:, j] - o[:, j]).mean())); base[f].append(float(o[:, j].mean()))
        lines.append("\n== layer-26 SAE codes at aligned session positions, R minus O, token-level (mean over receivers)")
        tok = {}
        for f in feats:
            tok[int(f)] = {"signed": float(np.mean(signed[f])), "absolute": float(np.mean(absol[f])), "original_mean": float(np.mean(base[f]))}
            lines.append(f"  {'sel' if f in S else 'ctl'} {int(f):>5}: signed {tok[int(f)]['signed']:+.5f}  absolute {tok[int(f)]['absolute']:.5f}  original mean {tok[int(f)]['original_mean']:.5f}")
        res["token_level_code_changes"] = tok
    (out / "analysis_h6.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "analysis_h6.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
