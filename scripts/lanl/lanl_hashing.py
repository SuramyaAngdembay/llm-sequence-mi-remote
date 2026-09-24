#!/usr/bin/env python3
"""User-level hashing and population checks shared by the LANL ETL and split.

Why this exists (review of 2026-09-24, verified the same day): the ETL kept an
ordinary user iff md5(user) % 10 == 0, and the split assigned
fold = md5(user) % 5 with fold 4 unseen. Any integer divisible by 10 is
divisible by 5, so every sampled ordinary user landed in fold 0 (seen). The
unseen pool therefore held only red-team users (23 of them, all with attack
windows) and no negative-only users, while the seen pool held 988
negative-only users. Seen-versus-unseen then changed training exposure AND
user composition at once.

The fix keeps the two decisions independent: each hashes the user id with its
own salt through SHA-256. The legacy md5 rules remain available only to
reproduce the historical artifacts.
"""
from __future__ import annotations

import collections
import hashlib
from typing import Dict, Iterable, List, Mapping, Optional, Set

SAMPLE_SALT = "lanl-user-sample-v2"
FOLD_SALT = "lanl-user-fold-v2"
UNSEEN_FOLD = 4


def legacy_md5(user: str) -> int:
    return int(hashlib.md5(user.encode()).hexdigest(), 16)


def salted(user: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{user}".encode()).hexdigest(), 16)


def check_salts_independent(sample_salt: str, fold_salt: str, *, legacy_sample: bool, legacy_fold: bool) -> None:
    """Refuse configurations in which sampling and fold assignment share a hash."""
    if legacy_sample and legacy_fold:
        raise ValueError(
            "legacy md5 sampling with legacy md5 folds couples the two decisions "
            "(md5 % 10 == 0 implies md5 % 5 == 0); use only to reproduce old artifacts"
        )
    if not legacy_sample and not legacy_fold and sample_salt == fold_salt:
        raise ValueError("sample and fold salts must differ")


def keep_sampled_user(user: str, sample_mod: int, *, salt: str = SAMPLE_SALT, legacy: bool = False) -> bool:
    if sample_mod <= 1:
        return True
    h = legacy_md5(user) if legacy else salted(user, salt)
    return h % sample_mod == 0


def fold_of(user: str, n_folds: int = 5, *, salt: str = FOLD_SALT, legacy: bool = False) -> int:
    h = legacy_md5(user) if legacy else salted(user, salt)
    return h % n_folds


def population_report(records: Iterable[Mapping], *, fold_by_user: Optional[Mapping[str, int]] = None) -> Dict:
    """User-level composition of each evaluation pool.

    `records` carry user_id, y and seen (1 = seen pool, 0 = unseen pool).
    """
    pools: Dict[int, Dict[str, int]] = {}
    windows: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    user_max: Dict[int, Dict[str, int]] = collections.defaultdict(dict)
    for r in records:
        s, u, y = int(r["seen"]), str(r["user_id"]), int(r["y"])
        windows[s]["windows"] += 1
        windows[s]["positives"] += y
        user_max[s][u] = max(user_max[s].get(u, 0), y)
    out: Dict[str, Dict] = {}
    for s, name in ((1, "seen"), (0, "unseen")):
        um = user_max.get(s, {})
        neg_only = sorted(u for u, m in um.items() if m == 0)
        attack_users = {u for u, m in um.items() if m == 1}
        entry = {
            "windows": int(windows[s]["windows"]),
            "positives": int(windows[s]["positives"]),
            "users": len(um),
            "users_with_positive": len(attack_users),
            "negative_only_users": len(neg_only),
        }
        if fold_by_user is not None:
            entry["negative_only_user_folds"] = dict(
                sorted(collections.Counter(int(fold_by_user[u]) for u in neg_only).items())
            )
        out[name] = entry
    return out


def check_population(
    report: Mapping,
    *,
    unseen_users: Optional[Set[str]] = None,
    train_users: Optional[Set[str]] = None,
    min_negative_only_users: int = 1,
) -> List[str]:
    """Return every problem found; an empty list means the checks passed."""
    problems: List[str] = []
    for pool in ("seen", "unseen"):
        entry = report.get(pool, {})
        if entry.get("users_with_positive", 0) == 0:
            problems.append(f"{pool} pool has no user with an attack window")
        if entry.get("negative_only_users", 0) < min_negative_only_users:
            problems.append(
                f"{pool} pool has {entry.get('negative_only_users', 0)} negative-only users "
                f"(need >= {min_negative_only_users}); its negatives would come only from attack users"
            )
    folds = set()
    for pool in ("seen", "unseen"):
        folds |= set(report.get(pool, {}).get("negative_only_user_folds", {}))
    if report.get("seen", {}).get("negative_only_user_folds") is not None and len(folds) <= 1:
        problems.append(f"negative-only users occupy a single fold {sorted(folds)}: sampling and folds are coupled")
    if unseen_users is not None and train_users is not None:
        leaked = sorted(unseen_users & train_users)
        if leaked:
            problems.append(f"{len(leaked)} unseen users also appear in training (first {leaked[:5]})")
    return problems
