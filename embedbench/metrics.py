"""Retrieval accuracy metrics.

Relevance is binary and document-level: a retrieved chunk counts through its
``doc_id``.  Duplicate chunks from the same document are de-duplicated by first
occurrence before metrics are computed.  Precision@k uses ``k`` as the
denominator, the standard IR definition, even when fewer than ``k`` unique
documents were retrieved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k must be an integer")
    if k < 1:
        raise ValueError("k must be >= 1")


def dedupe_ranked_docs(doc_ids: Iterable[str], limit: int | None = None) -> list[str]:
    """Return document IDs in first-occurrence order, optionally truncated."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    seen: set[str] = set()
    ranked: list[str] = []
    for doc_id in doc_ids:
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
        if limit is not None and len(ranked) >= limit:
            break
    return ranked


def recall_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k unique documents."""

    _validate_k(k)
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        raise ValueError("relevant must contain at least one document ID")
    top_k = dedupe_ranked_docs(retrieved, k)
    hits = len(relevant_set.intersection(top_k))
    return hits / len(relevant_set)


def precision_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of the top-k slots that are relevant (denominator is k)."""

    _validate_k(k)
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        raise ValueError("relevant must contain at least one document ID")
    top_k = dedupe_ranked_docs(retrieved, k)
    hits = len(relevant_set.intersection(top_k))
    return hits / k


def mrr_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Mean reciprocal rank contribution: 1 / rank of the first relevant doc."""

    _validate_k(k)
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        raise ValueError("relevant must contain at least one document ID")
    for rank, doc_id in enumerate(dedupe_ranked_docs(retrieved, k), start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Binary nDCG@k with the standard log2(rank + 1) discount."""

    _validate_k(k)
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        raise ValueError("relevant must contain at least one document ID")

    top_k = dedupe_ranked_docs(retrieved, k)
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """1.0 when at least one relevant document appears in the top k, else 0.0."""

    _validate_k(k)
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        raise ValueError("relevant must contain at least one document ID")
    return 1.0 if relevant_set.intersection(dedupe_ranked_docs(retrieved, k)) else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Per-query metric tuple."""

    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
        }


def evaluate_retrieval(
    relevant: Sequence[str],
    retrieved: Sequence[str],
    k: int,
) -> RetrievalMetrics:
    """Compute all five metrics for one query."""

    return RetrievalMetrics(
        recall=recall_at_k(relevant, retrieved, k),
        precision=precision_at_k(relevant, retrieved, k),
        mrr=mrr_at_k(relevant, retrieved, k),
        ndcg=ndcg_at_k(relevant, retrieved, k),
        hit_rate=hit_rate_at_k(relevant, retrieved, k),
    )


def aggregate_retrieval_metrics(
    rows: Sequence[RetrievalMetrics],
) -> dict[str, float | int]:
    """Average per-query metrics; raises when there are no rows."""

    if not rows:
        raise ValueError("Cannot aggregate metrics for zero queries")
    count = len(rows)
    return {
        "n_queries": count,
        "recall": sum(row.recall for row in rows) / count,
        "precision": sum(row.precision for row in rows) / count,
        "mrr": sum(row.mrr for row in rows) / count,
        "ndcg": sum(row.ndcg for row in rows) / count,
        "hit_rate": sum(row.hit_rate for row in rows) / count,
    }
