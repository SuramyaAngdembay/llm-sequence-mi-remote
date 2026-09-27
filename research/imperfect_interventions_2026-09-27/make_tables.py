#!/usr/bin/env python3
"""Markdown tables for README.md from the frozen evaluation outputs."""
import json
import math
import sys

S = json.load(open("results/eval/summary.json"))["table"]; W = json.load(open("results/eval/worlds.json"))
CASES = {
    "C1a_irrelevant_exact": ("1", "irrelevant, exact realization"),
    "C1b_irrelevant_crosstalk": ("1", "irrelevant, crosstalk 0.15"),
    "C1c_irrelevant_crosstalk_hi": ("1", "irrelevant, crosstalk 0.4"),
    "C1d_irrelevant_exec_noise": ("1", "irrelevant, execution noise 5%"),
    "C1e_irrelevant_under": ("1", "irrelevant, edit applied at 10%"),
    "C1f_irrelevant_mlp": ("1", "irrelevant, MLP readout (loose K)"),
    "C1g_irrelevant_coherent": ("1", "irrelevant, protected row at cosine 0.97"),
    "C1h_irrelevant_uniform_sigma": ("1", "irrelevant, uniform scales"),
    "C2a_relevant_exact": ("2", "relevant, exact realization"),
    "C2b_relevant_crosstalk": ("2", "relevant, crosstalk 0.15"),
    "C2c_relevant_mlp": ("2", "relevant, MLP readout (loose K)"),
    "C2d_relevant_coherent": ("2", "relevant, protected row at cosine 0.97"),
    "C3a_relevant_under": ("3", "relevant, edit applied at 2%"),
    "C3b_small_relevant_under": ("3", "weakly relevant, edit applied at 5%"),
    "C3c_relevant_exec_noise": ("3", "relevant, execution noise 30%"),
    "C4_many_to_one": ("4", "decoder writes an encoder-invisible direction"),
    "C5_interaction": ("5", "AND with an invisible partner"),
    "C6a_over_k": ("6", "insertion into a full TopK code (over-k)"),
    "C6b_deletion": ("6", "deletion requiring a support change"),
    "C7_steep_exec_noise": ("2/3", "relevant, steep MLP, execution noise 30%"),
}
rng = lambda v: "undefined" if any(isinstance(x, float) and math.isnan(x) for x in v) else f"{min(v):+.3f} to {max(v):+.3f}"
print("| case | condition | τ (mean) | ideal rescue I2 | ideal rescue I1 | necessity | interaction | realization error |")
print("|---|---|---|---|---|---|---|---|")
for c, (case, label) in CASES.items():
    ws = W[c]
    f = lambda k: [w.get(k, float("nan")) for w in ws]
    nec = f("necessity"); inter = f("interaction")
    print(f"| {case} | {label} | {sum(f('tau'))/len(ws):.3f} | {rng(f('ideal2'))} | {rng(f('ideal1'))} | "
          f"{rng(nec) if not all(math.isnan(x) for x in nec) else 'n/a'} | {rng(inter) if not all(math.isnan(x) for x in inter) else 'n/a'} | "
          f"{sum(f('mean_realization_error'))/len(ws):.2f} |")
print()
procs = [("P1_applied", "P1 applied"), ("P2_realization_gated", "P2 gated"), ("P3_direct_ideal", "P3 direct"),
         ("P4a_bound_certifiedK", "P4a bound"), ("P4b_bound_sampledK", "P4b sampled K"), ("P5_two_lifts", "P5 two lifts")]
print("| condition | truth (I2) | " + " | ".join(p[1] for p in procs) + " |")
print("|---|---|" + "---|" * len(procs))
for c, (case, label) in CASES.items():
    ws = W[c]
    t2 = [w["ideal2"] for w in ws]; tau = [w["tau"] for w in ws]
    if any(math.isnan(x) for x in t2):
        truth = "ill-posed"
    else:
        cls = {"non-neg" if abs(a) >= b else "neg" for a, b in zip(t2, tau)}
        truth = "/".join(sorted(cls))
    cells = []
    for p, _ in procs:
        r = S[c][p]
        if r["ill_posed"] == 1:
            cells.append("ill-posed"); continue
        s = f"{r['correct']:.2f}"
        if r["false_negligible"]:
            s += f", **FN {r['false_negligible']:.2f}**"
        if r["false_nonnegligible"]:
            s += f", **FP {r['false_nonnegligible']:.2f}**"
        cells.append(s)
    print(f"| {c} | {truth} | " + " | ".join(cells) + " |")
