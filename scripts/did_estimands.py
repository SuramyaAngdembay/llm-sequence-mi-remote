"""Secondary analysis (motivated by ledger C6, written before results): does the
published donor difference-in-differences depend on keeping the best candidate?

For each context mode, per receiver (complete cases), compute
  DiD = (d_sel,anom - d_sel,ben) - (d_ctrl,anom - d_ctrl,ben)
under four ways of collapsing candidates:
  best      : min over donors and alphas   (the published endpoint)
  best@1    : min over donors at alpha 1
  mean@1    : mean over donors at alpha 1  (average edit)
  equalized : best over donors and alphas, benign candidates subsampled to the
              receiver's anomalous candidate count (200 draws, averaged)
View: full score. Aggregates: receiver-weighted mean (the published statistic)
and mean of user means with a 10,000-draw cluster bootstrap over users.
"""
import csv, random, statistics as st, sys
from collections import defaultdict
import numpy as np

path = sys.argv[1]
data = defaultdict(lambda: defaultdict(list))   # (mode, recv) -> (set, dtype, alpha) -> [(donor, delta)]
for r in csv.DictReader(open(path)):
    data[(r["context_mode"], r["receiver_example_id"])][(r["feature_set"], r["donor_type"], float(r["alpha"]))].append(
        (r["donor_example_id"], float(r["delta_full"])))
ARMS = ("top5", "control5_active")
ALPHAS = (0.25, 0.5, 0.75, 1.0)

def cand(cells, arm, dtype):
    """donor -> {alpha: delta}"""
    out = defaultdict(dict)
    for a in ALPHAS:
        for donor, d in cells.get((arm, dtype, a), []):
            out[donor][a] = d
    return out

def boot(user_vals, draws=10000, seed=42):
    users = sorted(user_vals); v = np.array([user_vals[u] for u in users])
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
    return np.quantile(m, [0.025, 0.975])

rng = random.Random(7)
modes = sorted({m for m, _ in data})
for mode in modes:
    est = defaultdict(dict)   # estimand -> recv -> DiD
    counts = []
    for (m, recv), cells in data.items():
        if m != mode:
            continue
        c = {(arm, dt): cand(cells, arm, dt) for arm in ARMS for dt in ("benign", "anomalous")}
        if any(len(v) == 0 for v in c.values()):
            continue
        nb, na = len(c[("top5", "benign")]), len(c[("top5", "anomalous")])
        counts.append((nb, na))
        def val(arm, dt, how, donors=None):
            cd = c[(arm, dt)]
            ds = donors if donors is not None else list(cd)
            if how == "best":
                return min(min(cd[d].values()) for d in ds)
            if how == "best@1":
                return min(cd[d][1.0] for d in ds if 1.0 in cd[d])
            if how == "mean@1":
                return st.fmean(cd[d][1.0] for d in ds if 1.0 in cd[d])
        for how in ("best", "best@1", "mean@1"):
            est[how][recv] = ((val("top5", "anomalous", how) - val("top5", "benign", how))
                              - (val("control5_active", "anomalous", how) - val("control5_active", "benign", how)))
        # equalized candidate counts: subsample the larger donor pool to the smaller one
        k = min(nb, na)
        vals = []
        for _ in range(200):
            pick = {}
            for dt in ("benign", "anomalous"):
                pool = sorted(c[("top5", dt)])
                pick[dt] = rng.sample(pool, k) if len(pool) > k else pool
            vals.append((val("top5", "anomalous", "best", pick["anomalous"]) - val("top5", "benign", "best", pick["benign"]))
                        - (val("control5_active", "anomalous", "best", pick["anomalous"]) - val("control5_active", "benign", "best", pick["benign"])))
        est["equalized"][recv] = st.fmean(vals)
    nb = np.array([x for x, _ in counts]); na = np.array([y for _, y in counts])
    print(f"\n=== {mode}: {len(counts)} complete receivers | benign candidates median {np.median(nb):.0f} (range {nb.min()}-{nb.max()}), "
          f"anomalous median {np.median(na):.0f} (range {na.min()}-{na.max()}); receivers with fewer anomalous than benign: {(na < nb).sum()}")
    for how in ("best", "best@1", "mean@1", "equalized"):
        per = est[how]
        byu = defaultdict(list)
        for rid, v in per.items():
            byu[rid.split(":")[0]].append(v)
        um = {u: st.fmean(v) for u, v in byu.items()}
        lo, hi = boot(um)
        print(f"  {how:<10} receiver-weighted {st.fmean(per.values()):+.6f} | mean of user means {st.fmean(um.values()):+.6f} "
              f"[{lo:+.6f}, {hi:+.6f}] users +{sum(v > 0 for v in um.values())}/-{sum(v < 0 for v in um.values())}")
