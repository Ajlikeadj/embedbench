from __future__ import annotations

import math

import pytest
from embedbench.metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    dedupe_ranked_docs,
    evaluate_retrieval,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_precision_mrr_hit_hand_computed() -> None:
    relevant = {"d1", "d3"}
    retrieved = ["d2", "d1", "d4", "d3"]

    assert recall_at_k(relevant, retrieved, k=4) == pytest.approx(1.0)
    assert precision_at_k(relevant, retrieved, k=4) == pytest.approx(0.5)
    assert mrr_at_k(relevant, retrieved, k=4) == pytest.approx(0.5)
    assert hit_rate_at_k(relevant, retrieved, k=4) == pytest.approx(1.0)

    # At k=2 only d1 is retrieved.  Precision uses k as the denominator.
    assert recall_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)
    assert precision_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)
    assert mrr_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)
    assert hit_rate_at_k(relevant, retrieved, k=2) == pytest.approx(1.0)


def test_ndcg_hand_computed_binary_relevance() -> None:
    relevant = {"d1", "d3"}
    retrieved = ["d2", "d1", "d4", "d3"]

    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 + 1.0 / math.log2(3)
    expected = dcg / idcg

    assert ndcg_at_k(relevant, retrieved, k=4) == pytest.approx(expected)

    # At k=2, only the hit at rank 2 contributes.
    dcg_k2 = 1.0 / math.log2(3)
    assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(dcg_k2 / idcg)

    # A perfect ranking has nDCG 1.
    assert ndcg_at_k(relevant, ["d1", "d3", "d2"], k=3) == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank_one() -> None:
    assert ndcg_at_k({"x"}, ["x", "y"], k=2) == pytest.approx(1.0)


def test_hit_rate_misses() -> None:
    assert hit_rate_at_k({"x"}, ["a", "b", "c"], k=3) == pytest.approx(0.0)
    assert recall_at_k({"x"}, ["a", "b", "c"], k=3) == pytest.approx(0.0)
    assert precision_at_k({"x"}, ["a", "b", "c"], k=3) == pytest.approx(0.0)
    assert mrr_at_k({"x"}, ["a", "b", "c"], k=3) == pytest.approx(0.0)
    assert ndcg_at_k({"x"}, ["a", "b", "c"], k=3) == pytest.approx(0.0)


def test_duplicate_chunks_for_same_document_count_once() -> None:
    retrieved = ["d2", "d2", "d1"]
    relevant = {"d1"}

    assert dedupe_ranked_docs(retrieved) == ["d2", "d1"]
    assert recall_at_k(relevant, retrieved, k=3) == pytest.approx(1.0)
    assert precision_at_k(relevant, retrieved, k=3) == pytest.approx(1.0 / 3.0)
    assert mrr_at_k(relevant, retrieved, k=3) == pytest.approx(0.5)
    assert ndcg_at_k(relevant, retrieved, k=3) == pytest.approx(1.0 / math.log2(3))
    assert hit_rate_at_k(relevant, retrieved, k=3) == pytest.approx(1.0)


def test_precision_uses_k_even_when_fewer_results_exist() -> None:
    assert precision_at_k({"a"}, ["a"], k=5) == pytest.approx(0.2)
    assert recall_at_k({"a"}, ["a"], k=5) == pytest.approx(1.0)
    assert ndcg_at_k({"a"}, ["a"], k=5) == pytest.approx(1.0)


def test_evaluate_retrieval_returns_all_metrics() -> None:
    result = evaluate_retrieval({"d1", "d3"}, ["d2", "d1", "d4", "d3"], k=4)
    assert isinstance(result, RetrievalMetrics)
    assert result.to_dict() == {
        "recall": pytest.approx(1.0),
        "precision": pytest.approx(0.5),
        "mrr": pytest.approx(0.5),
        "ndcg": pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))),
        "hit_rate": pytest.approx(1.0),
    }


def test_aggregate_metrics_is_arithmetic_mean() -> None:
    rows = [
        RetrievalMetrics(1.0, 0.8, 1.0, 0.9, 1.0),
        RetrievalMetrics(0.0, 0.0, 0.0, 0.1, 0.0),
    ]
    aggregate = aggregate_retrieval_metrics(rows)
    assert aggregate["n_queries"] == 2
    assert aggregate["recall"] == pytest.approx(0.5)
    assert aggregate["precision"] == pytest.approx(0.4)
    assert aggregate["mrr"] == pytest.approx(0.5)
    assert aggregate["ndcg"] == pytest.approx(0.5)
    assert aggregate["hit_rate"] == pytest.approx(0.5)


def test_empty_relevant_raises() -> None:
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(set(), ["a"], k=1)
    with pytest.raises(ValueError, match="relevant"):
        evaluate_retrieval([], ["a"], k=1)
    with pytest.raises(ValueError, match="zero queries"):
        aggregate_retrieval_metrics([])


@pytest.mark.parametrize("bad_k", [0, -1, True])
def test_invalid_k_raises(bad_k: int) -> None:
    with pytest.raises(ValueError):
        recall_at_k({"a"}, ["a"], k=bad_k)


def test_dedupe_preserves_order_and_limit() -> None:
    assert dedupe_ranked_docs(["b", "a", "b", "c", "a"], limit=2) == ["b", "a"]
    assert dedupe_ranked_docs([]) == []
