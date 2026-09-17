#!/usr/bin/env python3
"""Token-class decomposition of the adapted-NLL anomaly score.

The detector score of an example is the mean next-token negative log-likelihood
over scored targets (`extract_adapter_deltas.per_example_nll`):

    s_full = (1 / N) * sum_{t in targets} loss_t

This module splits that sum by the *class of the token being predicted* (after
the causal shift), so that for any partition of the targets into classes C,

    s_full = sum_{c in C} (N_c / N) * s_c ,        s_c = loss_sum_c / N_c

and any sub-group P of classes yields an exact conditional score for the
remaining targets B:

    s_B = (loss_sum_total - loss_sum_P) / (N - N_P)
        = ((N_P + N_B) * s_full - N_P * s_P) / N_B .

NOTE: `s_full - s_P` is NOT the behaviour score. It equals
(N_B / (N_P + N_B)) * (s_B - s_P) and retains a negatively weighted profile
term, which can reverse a behaviour-based ranking. `behavior_score_from_means`
below implements the correct identity and `scripts/tests/` contains a synthetic
case where the naive difference flips the ranking.

Scope of the intervention: removing profile targets removes their *direct*
contribution to the score. The profile text stays in the context and still
conditions every behavioural prediction.

Class schemas
-------------
CERT (`cert_class_spans`): classes are serialized lines, identified by their
line *prefix*, never by line index. `build_session_jsonl_fast.py` deletes the
`PSY ` line under `--profile-mode no_psy` and both `PSY `/`DAY ` lines under
`no_profile`, so line index 1 is PSY in one condition and SESCOUNT in another.
  DAY      -> "DAY "      organizational header. Contains `week=` as well as
                          static attributes, so dropping all DAY loss is not
                          purely dropping static personal attributes; the
                          `week=` span is additionally accumulated as the
                          sub-class DAY_WEEK (a subset of DAY, not a separate
                          partition class).
  PSY      -> "PSY "      psychometric line.
  SESCOUNT -> "SESSIONS " session-count line.
  SES      -> "SES "      behavioural session lines.
  OTHER    -> any line matching no known prefix (expected empty on CERT).
  SPECIAL  -> tokens with an empty character span (added special tokens).

LANL (`lanl_class_spans`): identity lives in *fields*, not lines, so classes
are `key=value` spans within each event line (see `scripts/lanl/lanl_etl.py:
serialize_event`). Provided here for a later phase; not yet validated against
LANL data.

A token is assigned the class of the span containing its *first* character,
which is the convention already used by `feature_token_attribution.py`. Tokens
whose character span crosses a class boundary are counted separately so the
ambiguous mass is measurable rather than assumed negligible.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Dict, List, Sequence, Tuple

Span = Tuple[int, int, str]

# ---------------------------------------------------------------- CERT schema

CERT_LINE_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("DAY ", "DAY"),
    ("PSY ", "PSY"),
    ("SESSIONS ", "SESCOUNT"),
    ("SES ", "SES"),
)
CERT_CLASSES: Tuple[str, ...] = ("DAY", "PSY", "SESCOUNT", "SES", "OTHER", "SPECIAL")
CERT_SUBCLASSES: Tuple[str, ...] = ("DAY_WEEK",)

# Frozen score views. Declared here so downstream code cannot silently change
# what "behaviour-only" means between datasets or between runs.
CERT_PROFILE_CLASSES: Tuple[str, ...] = ("DAY", "PSY")
CERT_VIEWS: Dict[str, Tuple[str, ...]] = {
    # every scored target: reproduces the original detector score
    "full": CERT_CLASSES,
    # direct profile contribution only
    "profile_only": ("DAY", "PSY"),
    # PRIMARY behaviour-only view: every non-profile target. INCLUDES SESCOUNT
    # (and OTHER/SPECIAL, expected to be empty/negligible on CERT).
    "behavior_only": ("SESCOUNT", "SES", "OTHER", "SPECIAL"),
    # SECONDARY behaviour-only view: session lines only, EXCLUDES SESCOUNT.
    "behavior_ses_only": ("SES",),
    # diagnostic: PSY alone, DAY alone
    "psy_only": ("PSY",),
    "day_only": ("DAY",),
}

# ---------------------------------------------------------------- LANL schema
# Field keys of `serialize_event`: "t{hh} su=.. du=.. sc=.. dc=.. at=.. lt=..
# or=.. res=..". `t{hh}` is a bare hour token with no "=".
LANL_IDENTITY_KEYS: Tuple[str, ...] = ("su", "du", "sc", "dc")
LANL_BEHAVIOR_KEYS: Tuple[str, ...] = ("at", "lt", "or", "res")
LANL_CLASSES: Tuple[str, ...] = ("ID_USER", "ID_HOST", "BEHAV", "HOUR", "OTHER", "SPECIAL")
LANL_VIEWS: Dict[str, Tuple[str, ...]] = {
    "full": LANL_CLASSES,
    "identity_only": ("ID_USER", "ID_HOST"),
    "behavior_only": ("BEHAV", "HOUR", "OTHER", "SPECIAL"),
    "behavior_no_hour": ("BEHAV", "OTHER", "SPECIAL"),
}


def cert_class_spans(text: str) -> List[Span]:
    """Contiguous [start, end) character spans covering `text`, one per line.

    A line's span includes its terminating newline, so a token that begins with
    "\\n" is attributed to the line it closes (the existing attribution
    convention). Classification is by line prefix, never by line index.
    """
    spans: List[Span] = []
    pos = 0
    n = len(text)
    while pos <= n:
        nl = text.find("\n", pos)
        end = n if nl == -1 else nl + 1  # include the newline in this line
        line = text[pos : (n if nl == -1 else nl)]
        cls = "OTHER"
        for prefix, name in CERT_LINE_PREFIXES:
            if line.startswith(prefix):
                cls = name
                break
        if end > pos:
            spans.append((pos, end, cls))
        if nl == -1:
            break
        pos = end
    return spans


def cert_subclass_spans(text: str) -> List[Span]:
    """`week=<value>` spans inside DAY lines, as the DAY_WEEK sub-class.

    DAY_WEEK is a subset of DAY, reported alongside it; it is not part of the
    partition and must not be added to the class sums.
    """
    out: List[Span] = []
    for start, end, cls in cert_class_spans(text):
        if cls != "DAY":
            continue
        line = text[start:end]
        i = line.find("week=")
        if i == -1:
            continue
        j = line.find(" ", i)
        if j == -1:
            j = len(line.rstrip("\n"))
        out.append((start + i, start + j, "DAY_WEEK"))
    return out


def lanl_class_spans(text: str) -> List[Span]:
    """Field-level spans for the LANL event serialization.

    Each whitespace-delimited field becomes a span; `key=value` fields are
    classified by key, and a bare `t<hh>` field is the HOUR class. Separators
    (spaces, newlines) attach to the preceding field's span so the text stays
    covered contiguously.

    Not yet validated against LANL data; CERT line classes do not transfer.
    """
    spans: List[Span] = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i] in " \n":
            i += 1
        if i >= n:
            break
        j = i
        while j < n and text[j] not in " \n":
            j += 1
        field = text[i:j]
        k = field.find("=")
        if k == -1:
            cls = "HOUR" if field[:1] == "t" and field[1:].isdigit() else "OTHER"
        else:
            key = field[:k]
            if key in ("su", "du"):
                cls = "ID_USER"
            elif key in ("sc", "dc"):
                cls = "ID_HOST"
            elif key in LANL_BEHAVIOR_KEYS:
                cls = "BEHAV"
            else:
                cls = "OTHER"
        end = j
        while end < n and text[end] in " \n":
            end += 1
        spans.append((i, end, cls))
        i = end
    if spans and spans[0][0] > 0:
        spans.insert(0, (0, spans[0][0], "OTHER"))
    return spans


# ------------------------------------------------------- token classification


def classify_tokens(
    offsets: Sequence[Tuple[int, int]], spans: Sequence[Span]
) -> Tuple[List[str], List[bool]]:
    """Map each token to a class and flag class-boundary-crossing tokens.

    `offsets` is a fast tokenizer's `offset_mapping` for the *truncated*
    tokenization actually scored. Tokens with an empty span (a == b), which is
    how HF marks added special tokens, are classified SPECIAL.

    Returns (classes, crosses_boundary) of the same length as `offsets`.
    """
    starts = [s for s, _, _ in spans]
    classes: List[str] = []
    crosses: List[bool] = []
    for a, b in offsets:
        if b <= a:
            classes.append("SPECIAL")
            crosses.append(False)
            continue
        idx = bisect_right(starts, a) - 1
        if idx < 0:
            classes.append("OTHER")
            crosses.append(False)
            continue
        cls = spans[idx][2]
        classes.append(cls)
        end_idx = bisect_right(starts, b - 1) - 1
        crosses.append(end_idx != idx and 0 <= end_idx < len(spans))
    return classes, crosses


def classify_subclass(
    offsets: Sequence[Tuple[int, int]], subspans: Sequence[Span]
) -> List[bool]:
    """True where a token starts inside one of the sub-class spans."""
    if not subspans:
        return [False] * len(offsets)
    starts = [s for s, _, _ in subspans]
    flags: List[bool] = []
    for a, b in offsets:
        if b <= a:
            flags.append(False)
            continue
        idx = bisect_right(starts, a) - 1
        flags.append(idx >= 0 and subspans[idx][0] <= a < subspans[idx][1])
    return flags


# ------------------------------------------------------------- accumulation


def accumulate_class_losses(
    classes: Sequence[str],
    token_losses: Sequence[float],
    target_mask: Sequence[float],
    class_names: Sequence[str],
    *,
    crosses: Sequence[bool] | None = None,
    subclass_flags: Sequence[bool] | None = None,
    subclass_name: str = "DAY_WEEK",
) -> Dict[str, float]:
    """Accumulate summed losses and target counts per class, after the shift.

    Convention (identical to `extract_adapter_deltas.per_example_nll`):
      * `classes[i]` is the class of *input token* i;
      * `token_losses[i]` is the loss of the target at input position i+1, i.e.
        cross entropy of logits at position i against token i+1;
      * `target_mask[i]` is 1 where that target is scored (attention_mask[1:]),
        which excludes padding;
      * input token 0 is never a target and is therefore never counted.

    The loss is attributed to the class of the token being *predicted*
    (`classes[i + 1]`), not to the class of the context position.
    """
    n_shift = len(token_losses)
    if len(target_mask) != n_shift:
        raise ValueError(f"target_mask length {len(target_mask)} != losses {n_shift}")
    if len(classes) < n_shift + 1:
        raise ValueError(f"classes length {len(classes)} < losses + 1 = {n_shift + 1}")

    out: Dict[str, float] = {}
    for name in class_names:
        out[f"loss_sum_{name}"] = 0.0
        out[f"n_{name}"] = 0
        out[f"n_boundary_{name}"] = 0
    out["loss_sum_total"] = 0.0
    out["n_targets"] = 0
    out[f"loss_sum_{subclass_name}"] = 0.0
    out[f"n_{subclass_name}"] = 0

    for i in range(n_shift):
        if not target_mask[i]:
            continue
        tgt = i + 1
        cls = classes[tgt]
        if f"loss_sum_{cls}" not in out:
            cls = "OTHER"
        loss = float(token_losses[i])
        out[f"loss_sum_{cls}"] += loss
        out[f"n_{cls}"] += 1
        out["loss_sum_total"] += loss
        out["n_targets"] += 1
        if crosses is not None and crosses[tgt]:
            out[f"n_boundary_{cls}"] += 1
        if subclass_flags is not None and subclass_flags[tgt]:
            out[f"loss_sum_{subclass_name}"] += loss
            out[f"n_{subclass_name}"] += 1
    return out


# ------------------------------------------------------------- score views


def view_score(row: Dict[str, float], classes: Sequence[str]) -> float:
    """Mean NLL over the targets of `classes`; NaN when that view has no target."""
    loss = sum(float(row.get(f"loss_sum_{c}", 0.0)) for c in classes)
    n = sum(int(row.get(f"n_{c}", 0)) for c in classes)
    if n == 0:
        return float("nan")
    return loss / n


def behavior_score_from_means(
    s_full: float, s_profile: float, n_profile: int, n_behavior: int
) -> float:
    """The correct conditional behaviour score from the two means and counts.

        s_B = ((N_P + N_B) * s_full - N_P * s_P) / N_B

    Provided to cross-check the directly accumulated behaviour score. The naive
    `s_full - s_profile` is a different quantity; see `naive_difference`.
    """
    if n_behavior == 0:
        return float("nan")
    return ((n_profile + n_behavior) * s_full - n_profile * s_profile) / n_behavior


def naive_difference(s_full: float, s_profile: float) -> float:
    """The INCORRECT subtraction, kept only so tests can demonstrate its failure."""
    return s_full - s_profile
