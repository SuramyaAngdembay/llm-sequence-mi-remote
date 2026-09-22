import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1] + "/token_delta_sae_causal_candidate_rows.csv")))
f = lambda r, k: float(r[k])
users = {r["receiver_example_id"].split(":")[0] for r in rows}
print(f"rows: {len(rows)}   receiver users: {len(users)}")
print("\nPRE-REG RULE 5a: per-class sums reconstruct the scalar score")
e = [abs(f(r, "base_full") - f(r, "base_score_recomputed")) for r in rows]
print(f"  max |base_full - base_score_recomputed| = {max(e):.3e}   (must be ~0)")
print("\nPRE-REG RULE 5b: patched counts == base counts is asserted in-run; a violation kills the job")
print("\nPRE-REG RULE 1: cached vs matched-batch base")
d = [abs(f(r, "base_cache_minus_recomputed")) for r in rows]
print(f"  mean |diff| {st.fmean(d):.3e}   max {max(d):.3e}")
print("\nbase_is_recomputed set on every row:", all(r.get("base_is_recomputed") == "1" for r in rows))
print("delta equals the recomputed delta:", all(abs(f(r, "delta") - f(r, "delta_recomputed")) < 1e-9 for r in rows))
print("\nper-class target share:")
cls = ["DAY", "PSY", "SESCOUNT", "SES", "OTHER", "SPECIAL"]
tot = st.fmean([sum(f(r, "n_" + c) for c in cls) for r in rows])
for c in cls:
    m = st.fmean([f(r, "n_" + c) for r in rows])
    print(f"  {c:<9}{m:7.1f}  ({100 * m / tot:5.1f}%)")
