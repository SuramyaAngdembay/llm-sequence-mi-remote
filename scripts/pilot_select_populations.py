#!/usr/bin/env python3
"""Freeze the populations for the 2026-09-25 exploratory pilots (stdlib only).

Written and run before any pilot scoring. Output: one JSON manifest listing,
by example index into the r4.2 `eval.jsonl`,

  * malicious receivers: up to --per-user attack days of each confirmation user,
    ordered by sha256(salt | example_id);
  * matched benign receivers: for each attack day, the same user's benign day
    nearest in day_index (earlier on a tie), each used at most once;
  * a donor policy per department: up to --donors benign days from distinct
    never-malicious users of that department, in hash order (an intervention
    policy fixed in advance, not searched per receiver);
  * profile-swap partners per receiver user: a never-malicious user of the same
    department and one of a different department, in hash order, with the
    partner day whose DAY and PSY lines are used;
  * with --train-jsonl (added 2026-09-25, before any pilot scoring): a
    same-department user from the adapter's TRAINING split, in hash order, and
    the DAY and PSY lines of one of that user's training days. Every eval user
    is absent from training, so this is the only familiar-identity contrast.

Usage:
  python3 scripts/pilot_select_populations.py --eval-jsonl E --confirmation-users C \
      --discovery-users D --out manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

SALT = "pilot-2026-09-25"


def h(*parts: str) -> str:
    return hashlib.sha256("|".join((SALT,) + parts).encode()).hexdigest()


def day_fields(text: str) -> dict:
    first = text.split("\n", 1)[0]
    if not first.startswith("DAY "):
        raise ValueError(f"record does not start with a DAY line: {first[:60]!r}")
    return dict(kv.split("=", 1) for kv in first[4:].split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", type=Path, required=True)
    ap.add_argument("--confirmation-users", type=Path, required=True)
    ap.add_argument("--discovery-users", type=Path, required=True)
    ap.add_argument("--per-user", type=int, default=10)
    ap.add_argument("--donors", type=int, default=8)
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    conf = {l.strip() for l in args.confirmation_users.read_text().splitlines() if l.strip()}
    disc = {l.strip() for l in args.discovery_users.read_text().splitlines() if l.strip()}
    if conf & disc:
        raise SystemExit(f"confirmation and discovery users overlap: {sorted(conf & disc)[:5]}")

    rows = []
    with args.eval_jsonl.open() as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            f = day_fields(r["text"])
            rows.append({"idx": i, "id": r["example_id"], "user": r["user_id"], "y": int(r["y"]),
                         "day": int(r["day_index"]), "dept": f.get("dept"), "week": f.get("week")})
    by_user = defaultdict(list)
    for r in rows:
        by_user[r["user"]].append(r)
    malicious_users = {u for u, rs in by_user.items() if any(r["y"] for r in rs)}
    benign_users = sorted(set(by_user) - malicious_users)
    missing = sorted(conf - set(by_user))
    if missing:
        raise SystemExit(f"confirmation users absent from eval.jsonl: {missing[:5]}")

    receivers, matched = [], []
    for u in sorted(conf):
        attacks = sorted((r for r in by_user[u] if r["y"] == 1), key=lambda r: h("recv", r["id"]))[: args.per_user]
        benign = [r for r in by_user[u] if r["y"] == 0]
        used = set()
        for a in attacks:
            cands = sorted((r for r in benign if r["idx"] not in used), key=lambda r: (abs(r["day"] - a["day"]), r["day"] > a["day"], r["day"]))
            if not cands:
                raise SystemExit(f"no benign day left for {a['id']}")
            b = cands[0]
            used.add(b["idx"])
            receivers.append(a["idx"])
            matched.append(b["idx"])

    depts = sorted({rows[i]["dept"] for i in receivers + matched})
    donor_policy = {}
    for d in depts:
        users = sorted((u for u in benign_users if by_user[u][0]["dept"] == d), key=lambda u: h("donor-user", d, u))
        picks = []
        for u in users[: args.donors]:
            day = sorted(by_user[u], key=lambda r: h("donor-day", r["id"]))[0]
            picks.append(day["idx"])
        donor_policy[d] = picks

    partners = {}
    for u in sorted(conf):
        d = by_user[u][0]["dept"]
        same = sorted((v for v in benign_users if by_user[v][0]["dept"] == d), key=lambda v: h("same", u, v))
        other = sorted((v for v in benign_users if by_user[v][0]["dept"] != d), key=lambda v: h("other", u, v))
        pick = lambda v: sorted(by_user[v], key=lambda r: h("partner-day", r["id"]))[0]["idx"]
        partners[u] = {"same_dept_user": same[0] if same else None, "same_dept_day": pick(same[0]) if same else None,
                       "other_dept_user": other[0], "other_dept_day": pick(other[0])}

    train_partners = {}
    if args.train_jsonl is not None:
        tdays = defaultdict(list)
        with args.train_jsonl.open() as fh:
            for line in fh:
                r = json.loads(line)
                tdays[r["user_id"]].append((r["example_id"], r["text"]))
        overlap = set(tdays) & set(by_user)
        if overlap:
            raise SystemExit(f"train and eval users overlap: {sorted(overlap)[:5]}")
        tdept = {v: day_fields(days[0][1])["dept"] for v, days in tdays.items()}
        for u in sorted(conf):
            d = by_user[u][0]["dept"]
            cands = sorted((v for v in tdays if tdept[v] == d), key=lambda v: h("train-same", u, v))
            if not cands:
                train_partners[u] = None
                continue
            day_id, text = sorted(tdays[cands[0]], key=lambda t: h("train-day", t[0]))[0]
            train_partners[u] = {"user": cands[0], "day_id": day_id, "profile_lines": text.split("\n")[:2]}

    out = {
        "salt": SALT,
        "eval_jsonl": str(args.eval_jsonl),
        "eval_jsonl_sha256": hashlib.sha256(args.eval_jsonl.read_bytes()).hexdigest(),
        "confirmation_users": sorted(conf),
        "discovery_confirmation_disjoint": True,
        "malicious_receivers": receivers,
        "matched_benign_receivers": matched,
        "receiver_pairs": [[a, b] for a, b in zip(receivers, matched)],
        "donor_policy_by_dept": donor_policy,
        "swap_partners_by_user": partners,
        **({"train_jsonl_sha256": hashlib.sha256(args.train_jsonl.read_bytes()).hexdigest(),
            "train_partners_by_user": train_partners} if args.train_jsonl is not None else {}),
        "counts": {
            "malicious": len(receivers), "benign_matched": len(matched), "users": len(conf),
            "departments": len(depts),
            "donors_per_dept": {d: len(v) for d, v in donor_policy.items()},
            "users_without_same_dept_partner": sorted(u for u, p in partners.items() if p["same_dept_user"] is None),
        },
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["counts"], indent=1))


if __name__ == "__main__":
    main()
