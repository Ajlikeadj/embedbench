"""Tests for the Evaluator and BenchmarkResult."""

from __future__ import annotations

import json

import pytest

from embedbench.datasets import demo_dataset
from embedbench.errors import DatasetError
from embedbench.evaluator import BenchmarkResult, Evaluator, ModelResult
from embedbench.models import BaseEmbeddingModel
from tests.fake_backend import FakeEmbeddingModel


def _model(name: str = "demo", **options: object) -> FakeEmbeddingModel:
    return FakeEmbeddingModel(name, {"dimension": 64, **options})


def _evaluate(models: list[BaseEmbeddingModel], **kwargs: object) -> BenchmarkResult:
    evaluator = Evaluator(
        models,
        k_values=(1, 3, 5),
        index_backend="numpy",
        show_progress=False,
        **kwargs,  # type: ignore[arg-type]
    )
    return evaluator.evaluate(demo_dataset())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_evaluate_records_quality_latency_memory_and_cost() -> None:
    result = _evaluate([_model(price_per_1m_tokens=0.02)])

    assert len(result.models) == 1
    model = result.models[0]
    assert model.status == "ok"
    assert model.metrics_at(5)["ndcg"] > 0.3
    assert set(model.metrics_at(5)) == {"hit_rate", "mrr", "ndcg", "precision", "recall"}

    assert model.latency["count"] == demo_dataset().num_queries
    assert model.latency["p50_ms"] >= 0
    assert model.latency["p95_ms"] >= model.latency["p50_ms"]
    assert model.latency_warm is not None

    assert model.tokens["total_tokens"] > 0
    assert model.tokens["price_known"] is True
    assert model.cost["usd"] is not None and model.cost["usd"] > 0

    assert model.memory["vector_bytes"] == demo_dataset().num_documents * 64 * 4
    assert model.memory["index_disk_bytes"] is not None
    assert model.index["vectors"] == demo_dataset().num_documents
    assert model.index["dimension"] == 64
    assert model.index["backend"] == "numpy"

    assert model.throughput["documents_per_second"] is not None
    assert len(model.per_query) == demo_dataset().num_queries


def test_per_query_metrics_average_to_reported_metrics() -> None:
    result = _evaluate([_model()])
    model = result.models[0]
    rows = [row["metrics_by_k"]["5"]["ndcg"] for row in model.per_query]
    assert sum(rows) / len(rows) == pytest.approx(model.metrics_at(5)["ndcg"])


def test_result_serializes_to_json_and_picks_best() -> None:
    result = _evaluate([_model()])
    payload = result.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == "2.0"
    assert decoded["k_values"] == [1, 3, 5]
    assert decoded["reference_k"] == 5
    assert decoded["summary"]["models_requested"] == 1
    assert decoded["summary"]["best_spec"] == "fake:demo"
    assert result.best() is result.models[0]
    assert result.best().spec == "fake:demo"


def test_markdown_renders_headline_table() -> None:
    markdown = _evaluate([_model()]).markdown()
    assert "# Embedding benchmark: demo" in markdown
    assert "| Model | Dim |" in markdown
    assert "fake:demo" in markdown


def test_multiple_models_are_ranked_by_quality() -> None:
    result = _evaluate([_model("demo"), _model("other")])
    assert len(result.successful) == 2
    recommendation = result.recommend("accuracy")
    assert len(recommendation["ranking"]) == 2
    assert recommendation["winner"] in {"fake:demo", "fake:other"}


# ---------------------------------------------------------------------------
# Failure isolation and validation
# ---------------------------------------------------------------------------


def test_failing_model_does_not_stop_the_run() -> None:
    result = _evaluate([_model("missing"), _model("demo")])

    assert [model.spec for model in result.failed] == ["fake:missing"]
    assert [model.spec for model in result.successful] == ["fake:demo"]
    assert "not installed" in (result.failed[0].error or "")
    assert result.to_dict()["summary"]["failed"] == 1


def test_invalid_k_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        Evaluator([_model()], k_values=())
    with pytest.raises(ValueError):
        Evaluator([_model()], k_values=(0,))


def test_empty_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        Evaluator([])


def test_invalid_dataset_raises() -> None:
    evaluator = Evaluator([_model()], k_values=(1,), index_backend="numpy", show_progress=False)
    dataset = demo_dataset()
    dataset.queries = []
    with pytest.raises(DatasetError):
        evaluator.evaluate(dataset)


def test_evaluate_many_returns_one_result_per_dataset() -> None:
    evaluator = Evaluator([_model()], k_values=(3,), index_backend="numpy", show_progress=False)
    results = evaluator.evaluate_many([demo_dataset(), demo_dataset()])
    assert len(results) == 2
    assert all(result.dataset_name == "demo" for result in results)


def test_identical_documents_rank_deterministically() -> None:
    from embedbench.datasets import DatasetDocument, RetrievalDataset, RetrievalQuery

    dataset = RetrievalDataset(
        name="ties",
        documents=[
            DatasetDocument(doc_id="first", text="alpha beta gamma"),
            DatasetDocument(doc_id="second", text="alpha beta gamma"),
        ],
        queries=[RetrievalQuery(query_id="q", query="alpha beta", relevant_doc_ids=("second",))],
    )
    evaluator = Evaluator([_model()], k_values=(1, 2), index_backend="numpy", show_progress=False)
    first = evaluator.evaluate(dataset)
    second = evaluator.evaluate(dataset)

    assert first.models[0].status == "ok"
    ranking = first.models[0].per_query[0]["ranked_doc_ids"]
    assert ranking == second.models[0].per_query[0]["ranked_doc_ids"]
    assert sorted(ranking) == ["first", "second"]
    assert first.models[0].metrics_at(2)["hit_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Recommendation weighting
# ---------------------------------------------------------------------------


def _manual_result() -> BenchmarkResult:
    def model(spec: str, ndcg: float, qps: float, usd: float | None) -> ModelResult:
        return ModelResult(
            spec=spec,
            provider=spec.split(":")[0],
            model=spec.split(":")[1],
            metrics_by_k={"10": {"ndcg": ndcg, "hit_rate": ndcg, "mrr": ndcg}},
            latency={"qps": qps, "p95_ms": 1000.0 / qps},
            cost={"usd": usd, "price_known": usd is not None},
        )

    return BenchmarkResult(
        dataset_name="manual",
        dataset_stats={"documents": 10, "queries": 5},
        k_values=[10],
        models=[
            model("accurate:slow", ndcg=0.9, qps=10, usd=0.0),
            model("fast:paid", ndcg=0.5, qps=100, usd=1.0),
        ],
    )


def test_recommendation_weights_change_the_winner() -> None:
    result = _manual_result()
    assert result.recommend("accuracy")["winner"] == "accurate:slow"
    assert result.recommend("speed")["winner"] == "fast:paid"
    assert result.recommend("cost")["winner"] == "accurate:slow"
    with pytest.raises(ValueError, match="Unknown priority"):
        result.recommend("cheapest")


def test_recommendation_handles_unknown_cost_neutrally() -> None:
    result = _manual_result()
    result.models[1].cost = {"usd": None}
    ranking = result.recommend("balanced")["ranking"]
    assert {row["spec"]: row["cost"] for row in ranking}["fast:paid"] == 0.5


def test_recommendation_with_no_successful_models() -> None:
    result = BenchmarkResult(
        dataset_name="empty",
        dataset_stats={},
        k_values=[5],
        models=[ModelResult(spec="x:y", provider="x", model="y", status="error", error="boom")],
    )
    recommendation = result.recommend()
    assert recommendation["winner"] is None
    assert recommendation["ranking"] == []
    assert result.best() is None
