import json, sys
T = json.load(open(sys.argv[1]))
procs = ["P0_zero_in_ci", "P1_applied", "P2_realization_gated", "P3_direct_ideal", "P4a_bound_certifiedK", "P4b_bound_sampledK", "P4c_hoffman_eq_certifiedK", "P5_two_lifts"]
short = {"negligible": "neg", "non-negligible": "NON", "inconclusive": "inc", "ill-posed": "ill", "insufficient (bound unavailable)": "n/a", "not identified (lift-dependent)": "nid"}
for c, P in T.items():
    print(c)
    for p in procs:
        r = P.get(p)
        if not r: continue
        d = " ".join(f"{short.get(k,k)}={v:.2f}" for k, v in r["decisions"].items())
        fn = "" if r["false_negligible_rate"] is None else f" FN={r['false_negligible_rate']:.2f}"
        cov = "" if r["coverage_of_ideal2"] is None else f" cov={r['coverage_of_ideal2']:.2f}"
        wid = "" if r["median_width"] is None else f" w={r['median_width']:.3f}"
        print(f"   {p:<28} {d}{fn}{cov}{wid}")
