"""Retrieval metrics for SnapBench.

The paper reports macro-averaged Recall@k over the 1,145 queries.
Because clean and corrupted variants share the same labels, a paired
delta is just recall(corrupted) - recall(clean) on the same query set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def recall_at_k(
    ranked_ids: Sequence[str],
    positive_ids: Iterable[str],
    k: int = 1,
) -> float:
    """Return 1.0 if any labeled positive appears in the top-k ranks."""
    positives = set(positive_ids)
    if not positives:
        raise ValueError("positive_ids must not be empty")
    return float(any(item_id in positives for item_id in ranked_ids[:k]))


def recall_from_scores(
    scores: np.ndarray,
    gallery_ids: Sequence[str],
    positive_ids: Sequence[Iterable[str]],
    k: int = 1,
) -> float:
    """Macro-averaged R@k from a [n_query, n_gallery] score matrix.

    Higher scores rank first. Ties are broken by gallery order (stable argsort).
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    if scores.shape[0] != len(positive_ids):
        raise ValueError("scores rows must match the number of queries")
    if scores.shape[1] != len(gallery_ids):
        raise ValueError("scores columns must match gallery_ids")

    order = np.argsort(-scores, axis=1, kind="stable")
    hits = []
    for row, pos in zip(order, positive_ids):
        ranked = [gallery_ids[i] for i in row]
        hits.append(recall_at_k(ranked, pos, k=k))
    return float(np.mean(hits))


def paired_delta(clean: float, corrupted: float) -> float:
    """Paired robustness delta in the same units as the metric (usually R@1)."""
    return corrupted - clean
