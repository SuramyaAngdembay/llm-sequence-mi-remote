#!/usr/bin/env python3
"""Construction and validation for the history-prefix probe (work package 3).

Question: does access to the *same user's* earlier static profile reduce the
current profile prediction loss more than an equally long *other user's*
profile, and does that change behavioural detection?

Three predeclared conditions, all on one frozen adapter, all scoring the same
targets:

  A  the original current-day text, no prefix
  B  the same user's static profile from a strictly earlier record, prepended
  C  a different user's static profile from a strictly earlier record,
     prepended in the identical format and matched to B in token length

Design points that the scoring depends on, each of which is easy to get wrong:

* **Token ids of the current day must be identical across conditions.** The
  prefix and the current text are tokenized separately and their id sequences
  concatenated. Re-tokenizing a joined string could merge tokens across the
  boundary and silently change the current-day tokenization, making the three
  conditions incomparable.
* **Scored targets must be identical across conditions.** In A the current
  text's first token is not a target (nothing precedes it). Adding a prefix
  would make it predictable and so add a scored target that A never had. This
  module therefore excludes the current day's first token as a target in
  *every* condition, and scores exactly the current-day tokens at within-day
  positions 1..n-1.
* **No historical-prefix token is ever a target.** The prefix supplies context
  only.
* **Length matching is a declared eligibility rule, not a search for an
  outcome.** For each recipient, donors are examined in a fixed seeded order
  and the first whose prefix tokenizes to exactly the recipient's prefix length
  is used. If none is found within `max_candidates`, the example is excluded
  and counted. The rule is fixed before any loss is computed.

The prefix carries only the static organizational and psychometric fields.
`week` is deliberately omitted, since it is temporal rather than an identity
attribute; that makes the prefix's DAY line differ in format from a training
DAY line, which is recorded rather than hidden.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

# Static fields of the DAY line, in serialization order, minus `week`.
DAY_STATIC_FIELDS = ("project", "role", "b_unit", "f_unit", "dept", "team", "itadmin")
PSY_FIELDS = ("O", "C", "E", "A", "N")
PREFIX_SEPARATOR = "\n"


def parse_profile(text: str) -> Dict[str, str]:
    """Pull the DAY/PSY field values out of a serialized user-day."""
    out: Dict[str, str] = {}
    for line in text.split("\n"):
        if line.startswith("DAY ") or line.startswith("PSY "):
            for tok in line.split(" ")[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k] = v
    return out


def build_prefix(profile: Dict[str, str]) -> str:
    """The historical-profile prefix: static DAY fields (no `week`) then PSY.

    Ends with a newline so the current-day text starts on its own line, the
    same boundary the model saw between lines during adaptation.
    """
    missing = [f for f in DAY_STATIC_FIELDS + PSY_FIELDS if f not in profile]
    if missing:
        raise ValueError(f"profile is missing fields: {missing}")
    day = "DAY " + " ".join(f"{f}={profile[f]}" for f in DAY_STATIC_FIELDS)
    psy = "PSY " + " ".join(f"{f}={profile[f]}" for f in PSY_FIELDS)
    return day + "\n" + psy + PREFIX_SEPARATOR


def profile_overlap(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Fraction of static profile fields on which two users agree."""
    fields = DAY_STATIC_FIELDS + PSY_FIELDS
    return sum(1 for f in fields if a.get(f) == b.get(f)) / len(fields)


def donor_order(recipient: str, donor_pool: Sequence[str], seed: int = 42) -> List[str]:
    """Deterministic donor order for a recipient: a hash-keyed shuffle.

    Depends only on the recipient id, the pool and the seed — never on labels,
    scores, or any observed effect. The recipient is removed from its own pool.
    """
    def key(u: str) -> str:
        return hashlib.sha256(f"{seed}:{recipient}:{u}".encode()).hexdigest()

    return sorted((u for u in donor_pool if u != recipient), key=key)


def choose_length_matched_donor(
    recipient: str,
    recipient_prefix_len: int,
    donor_pool: Sequence[str],
    donor_profile: Dict[str, Dict[str, str]],
    token_len: "callable",
    seed: int = 42,
    max_candidates: int = 200,
) -> Tuple[Optional[str], int]:
    """First donor, in the deterministic order, whose prefix has exactly the
    recipient's prefix token length. Returns (donor_id or None, n_examined)."""
    for i, u in enumerate(donor_order(recipient, donor_pool, seed), start=1):
        if i > max_candidates:
            break
        prof = donor_profile.get(u)
        if prof is None:
            continue
        try:
            if token_len(build_prefix(prof)) == recipient_prefix_len:
                return u, i
        except ValueError:
            continue
    return None, min(len(donor_pool), max_candidates)


def assemble(
    prefix_ids: Sequence[int], current_ids: Sequence[int]
) -> Tuple[List[int], List[bool]]:
    """Concatenate prefix and current-day ids and mark the scored targets.

    Returns (input_ids, is_scored_target) where `is_scored_target[i]` refers to
    the token at position i being *predicted*. Exactly the current-day tokens
    at within-day positions 1..n-1 are marked, in every condition, so target
    membership does not depend on whether a prefix is present.
    """
    ids = list(prefix_ids) + list(current_ids)
    tgt = [False] * len(ids)
    start = len(prefix_ids)
    for j in range(1, len(current_ids)):          # skip the current day's first token
        tgt[start + j] = True
    return ids, tgt


def validate_conditions(
    built: Dict[str, Tuple[Sequence[int], Sequence[bool]]],
    n_current: int,
    max_seq_len: int,
) -> Dict[str, object]:
    """Checks that must pass before any loss from these sequences is used.

    Raises on any violation; returns a small record of what was checked.
    """
    names = sorted(built)
    if "A" not in names:
        raise ValueError("condition A (no prefix) is required as the reference")
    ids_A, tgt_A = built["A"]
    if len(ids_A) != n_current:
        raise ValueError("condition A must contain exactly the current-day tokens")

    scored_A = [i for i, t in enumerate(tgt_A) if t]
    if scored_A != list(range(1, n_current)):
        raise ValueError("condition A must score current-day positions 1..n-1")

    n_scored = len(scored_A)
    for name in names:
        ids, tgt = built[name]
        if len(ids) > max_seq_len:
            raise ValueError(f"condition {name} is {len(ids)} tokens, over max_seq_len {max_seq_len}")
        scored = [i for i, t in enumerate(tgt) if t]
        if len(scored) != n_scored:
            raise ValueError(f"condition {name} scores {len(scored)} targets, A scores {n_scored}")
        # the scored block must be the final n_current-1 tokens, contiguous
        if scored != list(range(len(ids) - n_current + 1, len(ids))):
            raise ValueError(f"condition {name} does not score exactly the current-day tail")
        # and the current-day ids must be byte-identical to A's
        if list(ids[len(ids) - n_current:]) != list(ids_A):
            raise ValueError(f"condition {name} changed the current-day token ids")
        if any(tgt[i] for i in range(len(ids) - n_current)):
            raise ValueError(f"condition {name} marks a prefix token as a target")

    return {
        "conditions": names,
        "n_current_tokens": int(n_current),
        "n_scored_targets": int(n_scored),
        "prefix_tokens": {n: int(len(built[n][0]) - n_current) for n in names},
        "checks": [
            "current-day token ids identical across conditions",
            "scored target membership identical across conditions",
            "current day's first token never a target (so a prefix adds none)",
            "no prefix token is ever a target",
            "no condition exceeds max_seq_len",
        ],
    }
