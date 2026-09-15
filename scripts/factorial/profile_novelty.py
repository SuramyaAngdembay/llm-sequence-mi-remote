# Test the mechanism: is the malicious user that INVERTS at 8B (CDE1846) the one
# whose profile is LEAST novel relative to the benign training population?
import json, numpy as np
D="/anvil/scratch/x-bbhusal1/p1/jsonl_r62_full"
ORG=("role","b_unit","f_unit","dept","team","ITAdmin"); PSY=("O","C","E","A","N")
train={}                      # benign train users -> context (profile constant per user)
with open(f"{D}/train.jsonl") as f:
    for line in f:
        r=json.loads(line); u=r["user_id"]
        if u not in train: train[u]=r["context"]
mal={}
with open(f"{D}/eval.jsonl") as f:
    for line in f:
        r=json.loads(line)
        if r["y"]==1 and r["user_id"] not in mal: mal[r["user_id"]]=r["context"]
print(f"benign train users: {len(train)}   malicious eval users: {sorted(mal)}")
torg=[tuple(c[k] for k in ORG) for c in train.values()]
tpsy=np.array([[c[k] for k in PSY] for c in train.values()],dtype=float)
roc8b={"ACM2278":0.919,"CDE1846":0.155,"CMP2946":0.527,"MBG3183":0.527}
roc3b={"ACM2278":0.995,"CDE1846":0.803,"CMP2946":0.966,"MBG3183":0.941}
print(f"{'user':9s} {'org_exact':>9s} {'psy_minL1':>9s} {'psy_within10':>12s} {'8B_full':>8s} {'3B_full':>8s}  org_tuple / OCEAN")
for u,c in sorted(mal.items()):
    org=tuple(c[k] for k in ORG); psy=np.array([c[k] for k in PSY],dtype=float)
    n_org=sum(1 for t in torg if t==org)
    d=np.abs(tpsy-psy).sum(1); mn=float(d.min()); within=int((d<=10).sum())
    print(f"{u:9s} {n_org:9d} {mn:9.1f} {within:12d} {roc8b[u]:8.3f} {roc3b[u]:8.3f}  {org} / {tuple(int(x) for x in psy)}")
print("DONE_NOVELTY")
