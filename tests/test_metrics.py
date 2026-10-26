from __future__ import annotations

import math

import pytest

from embedbench.metrics import (
    METRIC_NAMES,
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    cost_quality_ratio,
    dedupe_ranked_docs,
    estimate_cost,
    evaluate_rankings,
    evaluate_rankings_at_k,
    evaluate_retrieval,
    hit_rate_at_k,
    latency_stats,
    mean_metrics_by_k,
    memory_footprint,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    summarize_metrics,
    throughput,
    warm_latency_stats,
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


# ---------------------------------------------------------------------------
# Vectorised batch metrics
# ---------------------------------------------------------------------------


def test_batch_metrics_match_scalar_functions() -> None:
    rankings = [
        ["d2", "d1", "d4", "d3"],
        ["d1", "d1", "d1"],  # duplicates collapse before scoring
        ["a", "b", "c"],  # complete miss
        [],  # empty retrieval
        ["d3", "d2"],  # relevant at rank 1
    ]
    relevant = [
        ["d1", "d3"],
        ["d1"],
        ["x"],
        ["x"],
        ["d3"],
    ]

    for k in (1, 2, 4):
        batch = evaluate_rankings(rankings, relevant, k)
        assert len(batch) == len(rankings)
        for index in range(len(rankings)):
            scalar = evaluate_retrieval(relevant[index], rankings[index], k)
            assert batch.hit_rate[index] == pytest.approx(scalar.hit_rate)
            assert batch.mrr[index] == pytest.approx(scalar.mrr)
            assert batch.ndcg[index] == pytest.approx(scalar.ndcg)
            assert batch.precision[index] == pytest.approx(scalar.precision)
            assert batch.recall[index] == pytest.approx(scalar.recall)


def test_batch_means_and_rows() -> None:
    batch = evaluate_rankings([["a"], ["b"]], [["a"], ["a"]], k=1)
    means = batch.mean()
    assert means["hit_rate"] == pytest.approx(0.5)
    assert means["mrr"] == pytest.approx(0.5)
    assert means["ndcg"] == pytest.approx(0.5)
    rows = batch.rows()
    assert rows[0]["ndcg"] == pytest.approx(1.0)
    assert rows[1]["ndcg"] == pytest.approx(0.0)


def test_batch_ideal_dcg_caps_at_k() -> None:
    # Three relevant documents but k=2: a perfect top-2 ranking is nDCG 1.0,
    # because IDCG is also capped at k.
    batch = evaluate_rankings([["a", "b"]], [["a", "b", "c"]], k=2)
    assert batch.ndcg[0] == pytest.approx(1.0)
    assert batch.recall[0] == pytest.approx(2 / 3)


def test_batch_empty_input_is_zero_length() -> None:
    batch = evaluate_rankings([], [], k=5)
    assert len(batch) == 0
    assert batch.mean() == dict.fromkeys(METRIC_NAMES, 0.0)


def test_batch_rejects_mismatched_lengths_and_empty_relevance() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_rankings([["a"]], [], k=1)
    with pytest.raises(ValueError, match="relevant"):
        evaluate_rankings([["a"]], [[]], k=1)
    with pytest.raises(ValueError):
        evaluate_rankings([["a"]], [["a"]], k=0)


def test_evaluate_rankings_at_k_and_means() -> None:
    rankings = [["d2", "d1", "d4", "d3"]]
    relevant = [["d1", "d3"]]
    by_k = evaluate_rankings_at_k(rankings, relevant, [1, 2, 4])
    assert list(by_k) == [1, 2, 4]
    assert by_k[1].ndcg[0] == pytest.approx(0.0)
    means = mean_metrics_by_k(rankings, relevant, [4])
    assert means["4"]["recall"] == pytest.approx(1.0)
    assert set(means["4"]) == set(METRIC_NAMES)

    with pytest.raises(ValueError):
        evaluate_rankings_at_k(rankings, relevant, [])


# ---------------------------------------------------------------------------
# Latency, throughput, memory and cost
# ---------------------------------------------------------------------------


def test_latency_stats_percentiles_and_qps() -> None:
    stats = latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats.count == 4
    assert stats.mean_ms == pytest.approx(25.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(40.0)
    assert stats.p50_ms == pytest.approx(25.0)
    assert stats.p95_ms == pytest.approx(38.5)
    assert stats.p99_ms == pytest.approx(39.7)
    assert stats.total_ms == pytest.approx(100.0)
    assert stats.qps == pytest.approx(40.0)
    assert stats.to_dict()["qps"] == pytest.approx(40.0)


def test_latency_stats_single_sample() -> None:
    stats = latency_stats([7.5])
    assert stats.p50_ms == pytest.approx(7.5)
    assert stats.p95_ms == pytest.approx(7.5)
    assert stats.qps == pytest.approx(round(1000.0 / 7.5, 3))


def test_latency_stats_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        latency_stats([])
    with pytest.raises(ValueError, match="non-negative"):
        latency_stats([1.0, -1.0])


def test_warm_latency_drops_first_sample() -> None:
    assert warm_latency_stats([100.0]) is None
    warm = warm_latency_stats([100.0, 10.0, 20.0])
    assert warm is not None
    assert warm.count == 2
    assert warm.mean_ms == pytest.approx(15.0)


def test_percentile_validation_and_values() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([5.0], 100) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="q must be"):
        percentile([1.0], 0)
    with pytest.raises(ValueError, match="q must be"):
        percentile([1.0], 101)
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 50)


def test_throughput() -> None:
    assert throughput(100, 2.0) == pytest.approx(50.0)
    assert throughput(100, 0.0) is None
    assert throughput(0, -1.0) is None


def test_memory_footprint_math() -> None:
    footprint = memory_footprint(1_000, 768)
    assert footprint.vector_bytes == 1_000 * 768 * 4
    assert footprint.index_bytes == 0
    assert footprint.total_bytes == footprint.vector_bytes
    assert footprint.vector_mb == pytest.approx(2.93)

    with_overhead = memory_footprint(1_000, 768, index_overhead_ratio=0.1)
    assert with_overhead.index_bytes == 307_200
    assert with_overhead.total_bytes == 3_379_200

    with pytest.raises(ValueError):
        memory_footprint(-1, 768)
    with pytest.raises(ValueError):
        memory_footprint(10, 0)
    with pytest.raises(ValueError):
        memory_footprint(10, 8, index_overhead_ratio=-0.5)


def test_estimate_cost_math_and_unknown_price() -> None:
    estimate = estimate_cost(1_000_000, 0.02)
    assert estimate.usd == pytest.approx(0.02)
    assert estimate.price_known is True
    assert estimate.to_dict()["usd_per_1m_tokens"] == pytest.approx(0.02)

    unknown = estimate_cost(1_000_000, None)
    assert unknown.usd is None
    assert unknown.price_known is False

    assert estimate_cost(0, 0.02).usd == 0.0
    assert estimate_cost(10, 0.0).usd == 0.0

    with pytest.raises(ValueError):
        estimate_cost(-1, 0.02)
    with pytest.raises(ValueError):
        estimate_cost(1, -0.02)


def test_cost_quality_ratio() -> None:
    assert cost_quality_ratio(0.8, 0.02) == pytest.approx(40.0)
    assert cost_quality_ratio(0.8, 0.0) is None
    assert cost_quality_ratio(0.8, None) is None


def test_summarize_metrics_picks_reference_cut_off() -> None:
    by_k = {"1": {"ndcg": 0.2}, "5": {"ndcg": 0.6}}
    assert summarize_metrics(by_k, reference_k=5) == {"ndcg": 0.6}
    assert summarize_metrics(by_k, reference_k=10) == {}
