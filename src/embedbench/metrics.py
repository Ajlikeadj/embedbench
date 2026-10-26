"""Retrieval-quality and efficiency metrics.

The module has two layers:

Scalar layer
    One query at a time (:func:`hit_rate_at_k`, :func:`mrr_at_k`,
    :func:`ndcg_at_k`, :func:`precision_at_k`, :func:`recall_at_k`).  Used by
    the file-based pipeline and kept stable for existing callers.

Vectorised layer
    :func:`evaluate_rankings` computes the same five metrics for a whole batch
    of queries with NumPy discount matrices, and is what
    :class:`embedbench.Evaluator` uses.  The test suite checks both layers agree,
    so callers can mix them freely.

Relevance is binary and document-level: a retrieved chunk counts through its
``doc_id``.  Duplicate chunks from the same document are de-duplicated by first
occurrence before metrics are computed.  Precision@k uses ``k`` as the
denominator, the standard IR definition, even when fewer than ``k`` unique
documents were retrieved.

Efficiency metrics (:func:`latency_stats`, :func:`throughput`,
:func:`memory_footprint`, :func:`estimate_cost`) are pure functions of their
inputs so they can be unit tested without timing real hardware.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

METRIC_NAMES: tuple[str, ...] = ("hit_rate", "mrr", "ndcg", "precision", "recall")
"""Metric names in the order used by reports and CSV columns."""

DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10)
"""Cut-offs evaluated when the caller does not specify any."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k must be an integer")
    if k < 1:
        raise ValueError("k must be >= 1")


def _validate_k_values(k_values: Iterable[int]) -> tuple[int, ...]:
    values = tuple(k_values)
    if not values:
        raise ValueError("k_values must contain at least one cut-off")
    for value in values:
        _validate_k(value)
    return tuple(dict.fromkeys(values))


def _relevance_set(relevant: Iterable[str]) -> set[str]:
    relevance = {item for item in relevant if item}
    if not relevance:
        raise ValueError("relevant must contain at least one document ID")
    return relevance


def dedupe_ranked_docs(doc_ids: Iterable[str], limit: int | None = None) -> list[str]:
    """Return document IDs in first-occurrence order, optionally truncated.

    Args:
        doc_ids: Retrieved document IDs, possibly with repeats from adjacent
            chunks of the same document.
        limit: Keep at most this many unique IDs. ``None`` keeps all.

    Returns:
        The de-duplicated document IDs.

    Raises:
        ValueError: If ``limit`` is negative.
    """

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


# ---------------------------------------------------------------------------
# Scalar retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k unique documents.

    Args:
        relevant: Document IDs that are relevant for the query.
        retrieved: Ranked document IDs, best first; repeats are ignored.
        k: Ranking cut-off.

    Returns:
        ``hits / len(relevant)`` in ``[0, 1]``.

    Raises:
        ValueError: If ``k < 1`` or ``relevant`` is empty.
    """

    _validate_k(k)
    relevant_set = _relevance_set(relevant)
    top_k = dedupe_ranked_docs(retrieved, k)
    hits = len(relevant_set.intersection(top_k))
    return hits / len(relevant_set)


def precision_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of the top-k slots that are relevant (denominator is ``k``).

    Args:
        relevant: Document IDs that are relevant for the query.
        retrieved: Ranked document IDs, best first.
        k: Ranking cut-off.

    Returns:
        ``hits / k`` in ``[0, 1]``.

    Raises:
        ValueError: If ``k < 1`` or ``relevant`` is empty.
    """

    _validate_k(k)
    relevant_set = _relevance_set(relevant)
    top_k = dedupe_ranked_docs(retrieved, k)
    hits = len(relevant_set.intersection(top_k))
    return hits / k


def mrr_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Reciprocal rank of the first relevant document in the top ``k``.

    Args:
        relevant: Document IDs that are relevant for the query.
        retrieved: Ranked document IDs, best first.
        k: Ranking cut-off.

    Returns:
        ``1 / rank`` of the first hit, or ``0.0`` when nothing relevant was
        retrieved.

    Raises:
        ValueError: If ``k < 1`` or ``relevant`` is empty.
    """

    _validate_k(k)
    relevant_set = _relevance_set(relevant)
    for rank, doc_id in enumerate(dedupe_ranked_docs(retrieved, k), start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Binary nDCG@k with the standard ``log2(rank + 1)`` discount.

    Args:
        relevant: Document IDs that are relevant for the query.
        retrieved: Ranked document IDs, best first.
        k: Ranking cut-off.

    Returns:
        DCG@k divided by the ideal DCG@k in ``[0, 1]``.

    Raises:
        ValueError: If ``k < 1`` or ``relevant`` is empty.
    """

    _validate_k(k)
    relevant_set = _relevance_set(relevant)

    top_k = dedupe_ranked_docs(retrieved, k)
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """``1.0`` when at least one relevant document appears in the top ``k``.

    Args:
        relevant: Document IDs that are relevant for the query.
        retrieved: Ranked document IDs, best first.
        k: Ranking cut-off.

    Returns:
        ``1.0`` or ``0.0``.

    Raises:
        ValueError: If ``k < 1`` or ``relevant`` is empty.
    """

    _validate_k(k)
    relevant_set = _relevance_set(relevant)
    return 1.0 if relevant_set.intersection(dedupe_ranked_docs(retrieved, k)) else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Per-query metric tuple for a single cut-off."""

    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit_rate: float

    def to_dict(self) -> dict[str, float]:
        """Return the metrics as a plain mapping."""

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
    """Compute all five metrics for one query at one cut-off.

    Args:
        relevant: Document IDs that are relevant for the query.
        retrieved: Ranked document IDs, best first.
        k: Ranking cut-off.

    Returns:
        The per-query metrics.

    Raises:
        ValueError: If ``k < 1`` or ``relevant`` is empty.
    """

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
    """Average per-query metrics across a run.

    Args:
        rows: One entry per evaluated query.

    Returns:
        ``n_queries`` plus the mean of every metric.

    Raises:
        ValueError: If ``rows`` is empty.
    """

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


# ---------------------------------------------------------------------------
# Vectorised batch retrieval metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchRetrievalMetrics:
    """Per-query metric arrays for one cut-off.

    Every attribute is a float64 array with one entry per query, which makes
    slicing (for example by query bucket) cheap in stress workloads.

    Attributes:
        hit_rate: Array of 0.0/1.0 whether a relevant document was retrieved.
        mrr: Array of reciprocal ranks of the first relevant document.
        ndcg: Array of nDCG values.
        precision: Array of precision values.
        recall: Array of recall values.
    """

    hit_rate: np.ndarray
    mrr: np.ndarray
    ndcg: np.ndarray
    precision: np.ndarray
    recall: np.ndarray

    def __len__(self) -> int:
        return int(self.hit_rate.shape[0])

    def mean(self) -> dict[str, float]:
        """Return the mean of every metric over the batch."""

        if len(self) == 0:
            return dict.fromkeys(METRIC_NAMES, 0.0)
        return {
            "hit_rate": float(self.hit_rate.mean()),
            "mrr": float(self.mrr.mean()),
            "ndcg": float(self.ndcg.mean()),
            "precision": float(self.precision.mean()),
            "recall": float(self.recall.mean()),
        }

    def rows(self) -> list[dict[str, float]]:
        """Return one metric mapping per query."""

        return [
            {
                "hit_rate": float(self.hit_rate[index]),
                "mrr": float(self.mrr[index]),
                "ndcg": float(self.ndcg[index]),
                "precision": float(self.precision[index]),
                "recall": float(self.recall[index]),
            }
            for index in range(len(self))
        ]


def evaluate_rankings(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    k: int,
) -> BatchRetrievalMetrics:
    """Compute retrieval metrics for a batch of ranked document lists.

    Ranking metrics are computed with per-position relevance masks and a
    discount vector, so the cost is dominated by the ``queries x k`` matrix
    rather than by Python-level per-query work.

    Args:
        rankings: One ranked document-ID sequence per query, best first.
        relevant: One sequence of relevant document IDs per query.  Must be
            aligned with ``rankings``.
        k: Ranking cut-off.

    Returns:
        Per-query metric arrays.

    Raises:
        ValueError: If ``k < 1``, the inputs have different lengths, or any
            query has no relevant document.
    """

    _validate_k(k)
    if len(rankings) != len(relevant):
        raise ValueError(
            f"rankings and relevant must have the same length "
            f"(got {len(rankings)} and {len(relevant)})"
        )
    if not rankings:
        return BatchRetrievalMetrics(
            hit_rate=np.zeros(0, dtype=np.float64),
            mrr=np.zeros(0, dtype=np.float64),
            ndcg=np.zeros(0, dtype=np.float64),
            precision=np.zeros(0, dtype=np.float64),
            recall=np.zeros(0, dtype=np.float64),
        )

    relevance_sets: list[set[str]] = []
    top_k: list[list[str]] = []
    for ranking, relevant_docs in zip(rankings, relevant, strict=True):
        relevance_sets.append(_relevance_set(relevant_docs))
        top_k.append(dedupe_ranked_docs(ranking, k))

    count = len(top_k)
    mask = np.zeros((count, k), dtype=np.float64)
    for row, docs in enumerate(top_k):
        relevance_set = relevance_sets[row]
        for position, doc_id in enumerate(docs):
            if doc_id in relevance_set:
                mask[row, position] = 1.0

    ranks = np.arange(1, k + 1, dtype=np.float64)
    discounts = 1.0 / np.log2(ranks + 1.0)
    gains = mask * discounts

    dcg = gains.sum(axis=1)
    relevant_counts = np.array([len(item) for item in relevance_sets], dtype=np.float64)
    ideal_hits = np.minimum(relevant_counts, k).astype(np.int64)
    cumulative = np.cumsum(discounts)
    idcg = cumulative[ideal_hits - 1]
    ndcg = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)

    hits = mask.sum(axis=1)
    any_hit = hits > 0
    first_hit = np.argmax(mask > 0, axis=1) + 1
    mrr = np.where(any_hit, 1.0 / first_hit, 0.0)
    precision = hits / float(k)
    recall = hits / relevant_counts

    return BatchRetrievalMetrics(
        hit_rate=any_hit.astype(np.float64),
        mrr=mrr.astype(np.float64),
        ndcg=ndcg.astype(np.float64),
        precision=precision.astype(np.float64),
        recall=recall.astype(np.float64),
    )


def evaluate_rankings_at_k(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[int, BatchRetrievalMetrics]:
    """Run :func:`evaluate_rankings` for several cut-offs.

    Args:
        rankings: One ranked document-ID sequence per query, best first.
        relevant: One sequence of relevant document IDs per query.
        k_values: Cut-offs to evaluate.

    Returns:
        A mapping from ``k`` to per-query metrics, in the requested order.

    Raises:
        ValueError: If ``k_values`` is empty or contains an invalid cut-off.
    """

    return {
        cut_off: evaluate_rankings(rankings, relevant, cut_off)
        for cut_off in _validate_k_values(k_values)
    }


def mean_metrics_by_k(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, dict[str, float]]:
    """Return mean metrics keyed by cut-off.

    Args:
        rankings: One ranked document-ID sequence per query.
        relevant: One sequence of relevant document IDs per query.
        k_values: Cut-offs to evaluate.

    Returns:
        ``{"1": {...}, "5": {...}, "10": {...}}`` with JSON-friendly keys.
    """

    by_k = evaluate_rankings_at_k(rankings, relevant, k_values)
    return {str(cut_off): metrics.mean() for cut_off, metrics in by_k.items()}


# ---------------------------------------------------------------------------
# Latency, throughput, memory and cost
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, matching ``numpy.percentile``.

    Args:
        values: Non-empty sample of measurements.
        q: Percentile in ``(0, 100]``.

    Returns:
        The requested percentile.

    Raises:
        ValueError: If ``values`` is empty or ``q`` is outside ``(0, 100]``.
    """

    if not 0 < q <= 100:
        raise ValueError("q must be in (0, 100]")
    if len(values) == 0:
        raise ValueError("values must not be empty")
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, q, method="linear"))


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency distribution and throughput for a set of measurements.

    Attributes:
        count: Number of measurements.
        mean_ms: Arithmetic mean in milliseconds.
        min_ms: Fastest measurement.
        max_ms: Slowest measurement.
        p50_ms: Median.
        p90_ms: 90th percentile.
        p95_ms: 95th percentile.
        p99_ms: 99th percentile.
        total_ms: Sum of all measurements.
        qps: Queries per second implied by ``total_ms``, or ``None`` when the
            total is zero.
    """

    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float
    qps: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""

        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "total_ms": self.total_ms,
            "qps": self.qps,
        }


def latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Summarise per-query latency measurements in milliseconds.

    Args:
        latencies_ms: Raw measurements in milliseconds. Negative values are
            rejected because they indicate a timing bug upstream.

    Returns:
        A :class:`LatencyStats` with all fields rounded to three decimals.

    Raises:
        ValueError: If ``latencies_ms`` is empty or contains a negative value.
    """

    if len(latencies_ms) == 0:
        raise ValueError("latencies_ms must not be empty")
    array = np.asarray(latencies_ms, dtype=np.float64)
    if np.any(array < 0):
        raise ValueError("latencies_ms must be non-negative")
    total_ms = float(array.sum())
    qps = round(1000.0 * array.size / total_ms, 3) if total_ms > 0 else None
    return LatencyStats(
        count=int(array.size),
        mean_ms=round(float(array.mean()), 3),
        min_ms=round(float(array.min()), 3),
        max_ms=round(float(array.max()), 3),
        p50_ms=round(percentile(array, 50), 3),
        p90_ms=round(percentile(array, 90), 3),
        p95_ms=round(percentile(array, 95), 3),
        p99_ms=round(percentile(array, 99), 3),
        total_ms=round(total_ms, 3),
        qps=qps,
    )


def warm_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats | None:
    """Latency statistics with the first (cold-start) measurement dropped.

    Args:
        latencies_ms: Raw measurements in milliseconds.

    Returns:
        Warm statistics, or ``None`` when fewer than two measurements exist.
    """

    if len(latencies_ms) < 2:
        return None
    return latency_stats(latencies_ms[1:])


def throughput(units: int, duration_s: float) -> float | None:
    """Return units per second, or ``None`` for a non-positive duration.

    Args:
        units: Number of completed items (chunks, queries, batches).
        duration_s: Wall-clock time in seconds.

    Returns:
        Throughput rounded to three decimals, or ``None``.
    """

    if duration_s <= 0 or units < 0:
        return None
    return round(units / duration_s, 3)


@dataclass(frozen=True, slots=True)
class MemoryFootprint:
    """Static memory cost of holding a vector collection.

    Attributes:
        n_vectors: Number of stored vectors.
        dimension: Vector width.
        bytes_per_element: Bytes per component (4 for float32).
        vector_bytes: Raw vector payload.
        index_bytes: Index overhead estimate.
        total_bytes: ``vector_bytes`` plus ``index_bytes``.
    """

    n_vectors: int
    dimension: int
    bytes_per_element: int
    vector_bytes: int
    index_bytes: int
    total_bytes: int

    @property
    def vector_mb(self) -> float:
        """Raw payload in mebibytes."""

        return round(self.vector_bytes / (1024 * 1024), 3)

    @property
    def total_mb(self) -> float:
        """Total memory in mebibytes."""

        return round(self.total_bytes / (1024 * 1024), 3)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""

        return {
            "n_vectors": self.n_vectors,
            "dimension": self.dimension,
            "bytes_per_element": self.bytes_per_element,
            "vector_bytes": self.vector_bytes,
            "index_bytes": self.index_bytes,
            "total_bytes": self.total_bytes,
            "vector_mb": self.vector_mb,
            "total_mb": self.total_mb,
        }


def memory_footprint(
    n_vectors: int,
    dimension: int,
    *,
    bytes_per_element: int = 4,
    index_overhead_ratio: float = 0.0,
) -> MemoryFootprint:
    """Estimate the memory needed to store ``n_vectors`` vectors.

    Args:
        n_vectors: Number of vectors.
        dimension: Vector width.
        bytes_per_element: Component size, 4 for float32.
        index_overhead_ratio: Extra fraction for index structures (for example
            ``0.1`` when an index keeps an auxiliary graph).

    Returns:
        The estimated footprint.

    Raises:
        ValueError: If any input is negative or ``dimension`` is zero.
    """

    if n_vectors < 0:
        raise ValueError("n_vectors must be >= 0")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if bytes_per_element < 1:
        raise ValueError("bytes_per_element must be >= 1")
    if index_overhead_ratio < 0:
        raise ValueError("index_overhead_ratio must be >= 0")

    vector_bytes = n_vectors * dimension * bytes_per_element
    index_bytes = round(vector_bytes * index_overhead_ratio)
    return MemoryFootprint(
        n_vectors=int(n_vectors),
        dimension=int(dimension),
        bytes_per_element=int(bytes_per_element),
        vector_bytes=vector_bytes,
        index_bytes=index_bytes,
        total_bytes=vector_bytes + index_bytes,
    )


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Token usage expressed in US dollars for a hosted embedding model.

    Attributes:
        tokens: Total tokens billed.
        usd_per_1m_tokens: Applied price, or ``None`` when the model has no
            known price.
        usd: Estimated spend, or ``None`` when the price is unknown.
    """

    tokens: int
    usd_per_1m_tokens: float | None
    usd: float | None

    @property
    def price_known(self) -> bool:
        """Whether a price was available."""

        return self.usd is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""

        return {
            "tokens": self.tokens,
            "usd_per_1m_tokens": self.usd_per_1m_tokens,
            "usd": self.usd,
            "price_known": self.price_known,
        }


def estimate_cost(tokens: int, usd_per_1m_tokens: float | None) -> CostEstimate:
    """Convert a token count into a dollar estimate.

    Args:
        tokens: Total billed tokens.
        usd_per_1m_tokens: Price per million tokens, or ``None`` when unknown.
            Local models use ``0.0`` (free).

    Returns:
        The cost estimate.  An unknown price yields ``usd=None``.

    Raises:
        ValueError: If ``tokens`` is negative or the price is negative.
    """

    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    if usd_per_1m_tokens is not None and usd_per_1m_tokens < 0:
        raise ValueError("usd_per_1m_tokens must be >= 0")
    if usd_per_1m_tokens is None:
        return CostEstimate(tokens=int(tokens), usd_per_1m_tokens=None, usd=None)
    usd = round(int(tokens) / 1_000_000 * float(usd_per_1m_tokens), 6)
    return CostEstimate(
        tokens=int(tokens),
        usd_per_1m_tokens=float(usd_per_1m_tokens),
        usd=usd,
    )


def cost_quality_ratio(ndcg: float, usd: float | None) -> float | None:
    """Return nDCG per US dollar, a simple value-for-money score.

    Args:
        ndcg: Retrieval quality at the reference cut-off.
        usd: Estimated spend, or ``None``/``0.0`` for free local models.

    Returns:
        ``ndcg / usd`` rounded to three decimals, or ``None`` when the cost is
        unknown or free (the ratio is undefined rather than infinite).
    """

    if usd is None or usd <= 0:
        return None
    return round(float(ndcg) / float(usd), 3)


def summarize_metrics(
    by_k: Mapping[str, Mapping[str, float]],
    *,
    reference_k: int,
) -> dict[str, float]:
    """Pick a single reference cut-off out of a ``by_k`` mapping.

    Args:
        by_k: Mapping produced by :func:`mean_metrics_by_k`.
        reference_k: Cut-off to expose as the headline numbers.

    Returns:
        The metrics at ``reference_k``, or an empty mapping when absent.
    """

    return dict(by_k.get(str(reference_k), {}))


__all__ = [
    "DEFAULT_K_VALUES",
    "METRIC_NAMES",
    "BatchRetrievalMetrics",
    "CostEstimate",
    "LatencyStats",
    "MemoryFootprint",
    "RetrievalMetrics",
    "aggregate_retrieval_metrics",
    "cost_quality_ratio",
    "dedupe_ranked_docs",
    "estimate_cost",
    "evaluate_rankings",
    "evaluate_rankings_at_k",
    "evaluate_retrieval",
    "hit_rate_at_k",
    "latency_stats",
    "mean_metrics_by_k",
    "memory_footprint",
    "mrr_at_k",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "recall_at_k",
    "summarize_metrics",
    "throughput",
    "warm_latency_stats",
]
