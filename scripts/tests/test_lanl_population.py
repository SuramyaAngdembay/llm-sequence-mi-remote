#!/usr/bin/env python3
"""Regression checks for the LANL user-sampling / fold coupling.

The defect (review of 2026-09-24, verified on the saved eval.jsonl the same
day): ordinary users were kept iff md5(user) % 10 == 0 and assigned
fold = md5(user) % 5, fold 4 unseen. md5 % 10 == 0 implies md5 % 5 == 0, so
every sampled ordinary user fell in seen fold 0. Measured on the real split:
the seen pool held 988 negative-only users, the unseen pool 23 users, all with
attack windows, and zero negative-only users.

These checks pin:
  A. the legacy rules reproduce the coupling on synthetic user ids;
  B. the salted rules spread sampled users across all five folds;
  C. the real lanl_split.py refuses the legacy configuration, and when forced
     to reproduce it, reports the population problems it creates;
  D. the real lanl_split.py on a salted population passes every check, and
     its unseen pool contains negative-only users and no training users;
  E/F. unit checks for training-user leakage and identical salts.

Usage (standard library only): python3 scripts/tests/test_lanl_population.py
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LANL = Path(__file__).resolve().parents[1] / "lanl"
sys.path.insert(0, str(LANL))
from lanl_hashing import (  # noqa: E402
    check_population,
    check_salts_independent,
    fold_of,
    keep_sampled_user,
    population_report,
)

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


USERS = [f"U{i}" for i in range(1, 6001)]
RED = {f"U{i}" for i in range(100, 6001, 100)}          # 60 attack users
ORDINARY = [u for u in USERS if u not in RED]


def build_windows(path: Path, *, legacy: bool) -> None:
    kept = [u for u in USERS if u in RED or keep_sampled_user(u, 10, legacy=legacy)]
    with path.open("w") as fh:
        n = 0
        for u in kept:
            labels = [1, 1] + [0] * 6 if u in RED else [0] * 6
            for day, y in enumerate(labels):
                n += 1
                rec = {"example_id": f"{u}:{day}", "user_id": u, "day_index": day, "y": y, "n_events": 32}
                for key in ("text", "text_user_anon", "text_host_anon", "text_shuffle"):
                    rec[key] = f"t00 su={u} du={u} sc=C1 dc=C2 at=NTLM lt=3 or=LogOn res=Success"
                fh.write(json.dumps(rec) + "\n")


def run_split(src: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(LANL / "lanl_split.py"), str(src), str(out),
           "--train-n", "1200", "--val-n", "100", "--seen-n", "600", "--unseen-n", "600", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


print("\nA. the legacy rules reproduce the coupling")
legacy_kept = [u for u in ORDINARY if keep_sampled_user(u, 10, legacy=True)]
legacy_folds = collections.Counter(fold_of(u, legacy=True) for u in legacy_kept)
check("every legacy-sampled ordinary user is in fold 0", set(legacy_folds) == {0}, str(dict(legacy_folds)))

print("\nB. the salted rules keep sampling and folds independent")
kept = [u for u in ORDINARY if keep_sampled_user(u, 10)]
frac = len(kept) / len(ORDINARY)
check("about one in ten ordinary users is sampled", 0.08 <= frac <= 0.12, f"{frac:.3f}")
folds = collections.Counter(fold_of(u) for u in kept)
shares = {f: folds[f] / len(kept) for f in range(5)}
check("sampled users fill all five folds at 15-25% each",
      all(0.15 <= v <= 0.25 for v in shares.values()), str({f: round(v, 3) for f, v in shares.items()}))

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    print("\nC. the real split on the legacy configuration")
    build_windows(td / "legacy.jsonl", legacy=True)
    refused = run_split(td / "legacy.jsonl", td / "legacy_out", "--legacy-md5-folds", "--legacy-md5-sample")
    check("it refuses legacy md5 sampling with legacy md5 folds",
          refused.returncode != 0 and "couples" in refused.stderr, refused.stderr.strip().splitlines()[-1][:120])
    forced = run_split(td / "legacy.jsonl", td / "legacy_out", "--legacy-md5-folds", "--legacy-md5-sample",
                       "--allow-population-problems")
    check("forced reproduction still completes", forced.returncode == 0, forced.stderr[-200:])
    rep = json.loads((td / "legacy_out" / "population_report.json").read_text())
    check("the unseen pool has zero negative-only users, as measured on LANL",
          rep["unseen"]["negative_only_users"] == 0, json.dumps(rep["unseen"]))
    check("the report flags the coupling and the missing unseen negatives",
          any("single fold" in p for p in rep["problems"]) and any("unseen pool" in p for p in rep["problems"]),
          "; ".join(rep["problems"]))

    print("\nD. the real split on a salted population")
    build_windows(td / "salted.jsonl", legacy=False)
    ok = run_split(td / "salted.jsonl", td / "salted_out")
    check("the split passes its population checks", ok.returncode == 0 and "POPULATION CHECK passed" in ok.stdout,
          (ok.stdout + ok.stderr)[-300:])
    rep = json.loads((td / "salted_out" / "population_report.json").read_text())
    check("both pools contain negative-only users and attack users",
          all(rep[p]["negative_only_users"] > 0 and rep[p]["users_with_positive"] > 0 for p in ("seen", "unseen")),
          json.dumps({p: rep[p] for p in ("seen", "unseen")}))
    train_users = {json.loads(l)["user_id"] for l in (td / "salted_out" / "full" / "train.jsonl").open()}
    unseen = {json.loads(l)["user_id"] for l in (td / "salted_out" / "full" / "eval.jsonl").open()
              if json.loads(l)["seen"] == 0}
    check("no unseen user appears in training", not (train_users & unseen), f"{len(unseen)} unseen users")

print("\nE/F. unit checks")
toy = population_report([{"user_id": "U1", "y": 1, "seen": 0}, {"user_id": "U2", "y": 0, "seen": 0},
                         {"user_id": "U3", "y": 1, "seen": 1}, {"user_id": "U4", "y": 0, "seen": 1}])
problems = check_population(toy, unseen_users={"U1", "U2"}, train_users={"U2", "U4"})
check("an unseen user found in training is reported", any("appear in training" in p for p in problems), str(problems))
try:
    check_salts_independent("same", "same", legacy_sample=False, legacy_fold=False)
    check("identical salts are refused", False)
except ValueError:
    check("identical salts are refused", True)

print()
if FAIL:
    print(f"{len(FAIL)} check(s) FAILED: {FAIL}")
    sys.exit(1)
print("all checks passed")
