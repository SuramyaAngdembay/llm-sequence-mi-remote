# (1) Is the host really per-user-constant on LANL?  (2) Did any window hit token truncation?
import json, re, collections, numpy as np, pandas as pd
D="/anvil/scratch/x-bbhusal1/lanl/data/full"
src=collections.defaultdict(collections.Counter); dst=collections.defaultdict(collections.Counter)
n=0
with open(f"{D}/train.jsonl") as f:
    for line in f:
        r=json.loads(line); u=r["user_id"]
        for ev in r["text"].split(" | "):
            m=re.search(r" sc=(\S+) dc=(\S+)",ev)
            if m: src[u][m.group(1)]+=1; dst[u][m.group(2)]+=1; n+=1
def modal_frac(cnt):
    return np.array([c.most_common(1)[0][1]/sum(c.values()) for c in cnt.values() if sum(c.values())>=20])
fs, fd = modal_frac(src), modal_frac(dst)
print(f"events={n} users(>=20 ev)={len(fs)}")
print("per-user share of events from the user's MODAL source host: median=%.2f  q25=%.2f  q75=%.2f  mean=%.2f" % (np.median(fs),np.percentile(fs,25),np.percentile(fs,75),fs.mean()))
print("per-user share of events to the user's MODAL dest host:     median=%.2f  q25=%.2f  q75=%.2f  mean=%.2f" % (np.median(fd),np.percentile(fd,25),np.percentile(fd,75),fd.mean()))
print("fraction of users with >=80%% of events from one source host: %.2f" % (fs>=0.8).mean())
for m in ["full","host_anon"]:
    s=pd.read_parquet(f"/anvil/scratch/x-bbhusal1/lanl/deltas_{m}/example_scores.parquet",columns=["n_tokens","y"])
    print(f"[{m}] windows={len(s)} n_tokens max={int(s.n_tokens.max())} p99={int(np.percentile(s.n_tokens,99))} mean={s.n_tokens.mean():.0f}  positives max_tokens={int(s[s.y==1].n_tokens.max())}")
# LANL AP + prevalence for the table
for m in ["full","user_anon","host_anon","shuffle"]:
    a=json.load(open(f"/anvil/scratch/x-bbhusal1/lanl/results/ap_{m}.json"))
    print(f"[{m}] seen AP={a['seen_ap']:.4f} (prev {a['seen_pos']}/{a['seen_n']}={a['seen_pos']/a['seen_n']:.4f})  unseen AP={a['unseen_ap']:.4f} (prev {a['unseen_pos']}/{a['unseen_n']}={a['unseen_pos']/a['unseen_n']:.4f})")
print("DONE_CHECKS")
