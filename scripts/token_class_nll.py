#!/usr/bin/env python3
"""Per-token-class NLL accumulation, shared by the causal-intervention scorer.

The causal intervention answers "does patching feature f lower the anomaly
score?". That score is a mean over ALL scored tokens, so a drop can come from
the profile tokens, the behaviour tokens, or both. These helpers accumulate the
same token losses into per-class sums and counts, so one intervention can be
read against the full, profile-only and behaviour-only views.

Conventions are identical to `token_class_decomposition.accumulate_class_losses`:
the loss at input position i is the loss of the target at i+1, and it is
attributed to the class of the token being PREDICTED, i.e. `classes[i+1]`.
Padding is excluded by the attention mask exactly as in the scalar path, so
summing the per-class sums and dividing by the summed counts reproduces the
scalar per-example NLL. `scripts/tests/test_token_class_nll.py` checks that
against the reference Python implementation rather than assuming it.

Kept separate from the intervention script so it can be tested without loading
a language model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import token_class_decomposition as tcd


def build_class_index(schema: str) -> Tuple[List[str], Dict[str, int]]:
    if schema == "cert":
        names = list(tcd.CERT_CLASSES)
    elif schema == "lanl":
        names = list(tcd.LANL_CLASSES)
    else:
        raise ValueError(f"unknown token-class schema {schema!r}")
    return names, {n: i for i, n in enumerate(names)}


def class_ids_for_texts(
    texts: Sequence[str],
    offsets: torch.Tensor,
    schema: str,
    class_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Per-token class indices for the EXACT tokenization being scored.

    `offsets` is the fast tokenizer's offset_mapping from the same call that
    produced the input_ids, so truncation and padding are already applied and
    the alignment cannot drift.
    """
    span_fn = tcd.cert_class_spans if schema == "cert" else tcd.lanl_class_spans
    other = class_to_idx["OTHER"]
    out = torch.zeros(offsets.shape[:2], dtype=torch.long)
    for bi, text in enumerate(texts):
        names, _ = tcd.classify_tokens(
            [(int(a), int(b)) for a, b in offsets[bi].tolist()], span_fn(text)
        )
        out[bi] = torch.tensor([class_to_idx.get(n, other) for n in names], dtype=torch.long)
    return out


def _scatter_class(
    token_loss: torch.Tensor, mask: torch.Tensor, target_cls: torch.Tensor, n_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    b = token_loss.shape[0]
    sums = torch.zeros((b, n_classes), device=token_loss.device, dtype=torch.float32)
    counts = torch.zeros((b, n_classes), device=token_loss.device, dtype=torch.float32)
    sums.scatter_add_(1, target_cls, token_loss * mask)
    counts.scatter_add_(1, target_cls, mask)
    return sums, counts


def per_example_class_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    class_ids: torch.Tensor,
    n_classes: int,
    *,
    loss_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-class summed loss and target count, same chunking as the scalar path."""
    batch_size = int(logits.shape[0])
    chunk_size = batch_size if int(loss_batch_size) <= 0 else min(batch_size, int(loss_batch_size))
    sums: List[torch.Tensor] = []
    counts: List[torch.Tensor] = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        shift_logits = logits[start:end, :-1, :].float().contiguous()
        shift_labels = input_ids[start:end, 1:].contiguous()
        shift_mask = attention_mask[start:end, 1:].contiguous().float()
        target_cls = class_ids[start:end, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        cs, cc = _scatter_class(token_loss, shift_mask, target_cls, n_classes)
        sums.append(cs)
        counts.append(cc)
        del shift_logits, shift_labels, shift_mask, token_loss, target_cls
        if logits.device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(sums, dim=0), torch.cat(counts, dim=0)


def per_example_class_nll_from_hidden(
    lm_head: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    class_ids: torch.Tensor,
    n_classes: int,
    *,
    max_logit_elements: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-class sums/counts while chunking the LM head over token positions."""
    batch_size, seq_len, _ = hidden_states.shape
    sums = torch.zeros((batch_size, n_classes), device=hidden_states.device, dtype=torch.float32)
    counts = torch.zeros((batch_size, n_classes), device=hidden_states.device, dtype=torch.float32)
    if seq_len <= 1:
        return sums, counts

    vocab_size = int(getattr(lm_head, "out_features", 0) or getattr(getattr(lm_head, "weight", None), "shape", [0])[0])
    if vocab_size <= 0:
        raise ValueError("Could not infer lm_head vocab size for chunked class NLL")
    token_chunk = max(1, int(max_logit_elements) // max(1, batch_size * vocab_size))
    token_chunk = min(token_chunk, seq_len - 1)

    for pos_start in range(0, seq_len - 1, token_chunk):
        pos_end = min(pos_start + token_chunk, seq_len - 1)
        labels = input_ids[:, pos_start + 1 : pos_end + 1].contiguous()
        mask = attention_mask[:, pos_start + 1 : pos_end + 1].contiguous().float()
        target_cls = class_ids[:, pos_start + 1 : pos_end + 1].contiguous()
        if float(mask.sum().item()) == 0.0:
            continue
        logits = lm_head(hidden_states[:, pos_start:pos_end, :])
        token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.size())
        cs, cc = _scatter_class(token_loss, mask, target_cls, n_classes)
        sums += cs
        counts += cc
        del logits, labels, mask, token_loss, target_cls
        if hidden_states.device.type == "cuda":
            torch.cuda.empty_cache()

    return sums, counts


def views_from_class(
    sums: np.ndarray, counts: np.ndarray, class_names: Sequence[str],
    views: Dict[str, Tuple[str, ...]],
) -> Dict[str, np.ndarray]:
    """Exact conditional mean per view: summed class loss / summed class count.

    Never `s_full - s_profile`. With weights w_c = N_c / N that difference is
    `w_B * (s_B - s_P)`, which reorders examples relative to the conditional
    behaviour score; the ranking-preserving form is `s_full - w_P * s_P`. Taking
    the conditional mean straight from sums and counts avoids the question.
    """
    idx = {n: i for i, n in enumerate(class_names)}
    out: Dict[str, np.ndarray] = {}
    for vname, members in views.items():
        cols = [idx[m] for m in members if m in idx]
        if not cols:
            out[vname] = np.full(len(sums), np.nan, dtype=np.float64)
            continue
        num = sums[:, cols].sum(axis=1)
        den = counts[:, cols].sum(axis=1)
        out[vname] = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
    return out
