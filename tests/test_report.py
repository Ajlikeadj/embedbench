from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embedbench.report import (
    recommend_models,
    render_html,
    write_html_report,
    write_reports,
)


def make_model(
    spec: str,
    *,
    ndcg: float,
    p95: float,
    cost: float,
    recall: float = 0.8,
) -> dict:
    return {
        "spec": spec,
        "backend": spec.split(":", 1)[0],
        "model": spec.split(":", 1)[1],
        "status": "ok",
        "error": None,
        "metrics_k": 3,
        "accuracy": {
            "n_queries": 3,
            "recall": recall,
            "precision": 0.5,
            "mrr": ndcg - 0.05,
            "ndcg": ndcg,
            "hit_rate": 1.0,
        },
        "latency_ms": {
            "count": 3,
            "p50_ms": p95 / 2,
            "p95_ms": p95,
            "p99_ms": p95 * 1.2,
            "mean_ms": p95 / 1.7,
            "min_ms": p95 / 3,
            "max_ms": p95 * 1.2,
            "cold_query_ms": p95 * 1.4,
            "warm_p95_ms": p95 * 0.9,
            "queries_per_second": 1000 / p95,
        },
        "timing": {
            "model_load_ms": 10.0,
            "indexing_seconds": 0.1,
            "embedding_seconds": 0.05,
            "index_build_ms": 2.0,
            "total_seconds": 0.4,
        },
        "throughput": {
            "chunks_per_second": 40.0,
            "embed_chunks_per_second": 80.0,
            "queries_per_second": 1000 / p95,
        },
        "tokens": {
            "index_tokens": 500,
            "query_tokens": 50,
            "total_tokens": 550,
            "estimated": False,
            "method": "tokenizer:Fake",
            "cost_usd": cost,
            "price_per_1m_tokens": 0.02,
        },
        "index": {
            "backend": "faiss",
            "dimension": 384,
            "vectors": 4,
            "disk_size_bytes": 1000,
            "disk_size_mb": 0.001,
        },
        "cache": {"enabled": True, "hits": 4, "misses": 0, "hit_rate": 1.0},
        "resources": {
            "ram_peak_mb": 300.0,
            "peak_vram_mb": None,
            "gpu_peak_allocated_mb": None,
        },
        "per_query": [],
    }


@pytest.fixture
def sample_results() -> dict:
    models = [
        make_model("hf:accurate", ndcg=0.9, p95=100.0, cost=0.01, recall=0.95),
        make_model("hf:fast", ndcg=0.8, p95=10.0, cost=0.002, recall=0.7),
        make_model("hf:middle", ndcg=0.85, p95=50.0, cost=0.005, recall=0.85),
    ]
    models[0]["per_query"] = [
        {
            "query_id": "q1",
            "query": "safe query",
            "latency_ms": 100.0,
            "metrics": {"ndcg": 0.9, "mrr": 0.8, "hit_rate": 1.0, "recall": 1.0, "precision": 0.5},
            "ranked_doc_ids": ["a.md"],
            "relevant_doc_ids": ["a.md"],
        },
        {
            "query_id": "q2",
            "query": "</script><script>alert(1)</script>",
            "latency_ms": 101.0,
            "metrics": {"ndcg": 0.9, "mrr": 0.8, "hit_rate": 1.0, "recall": 1.0, "precision": 0.5},
            "ranked_doc_ids": ["a.md"],
            "relevant_doc_ids": ["a.md"],
        },
    ]
    recommendation = recommend_models(models, "balanced", top_k=3)
    return {
        "schema_version": "1.0",
        "generated_at": "2026-10-26T00:00:00+00:00",
        "tool": {"name": "embedbench", "version": "0.1.0"},
        "config": {
            "docs": "./docs",
            "models": [model["spec"] for model in models],
            "top_k": 3,
            "seed": 42,
            "report": {
                "title": "Test report",
                "priority": "balanced",
                "cost_per_1m_tokens": 0.02,
            },
        },
        "run": {
            "seed": 42,
            "top_k": 3,
            "price_per_1m_tokens": 0.02,
            "environment": {"packages": {"numpy": "2.0"}},
            "corpus": {
                "documents": 2,
                "chunks": 4,
                "skipped_files": [],
                "queries": 3,
            },
            "evalset": {"queries": 3, "source": "jsonl"},
        },
        "models": models,
        "recommendation": recommendation,
    }


def test_recommendations_change_with_priority(sample_results: dict) -> None:
    models = sample_results["models"]
    accuracy = recommend_models(models, "accuracy", top_k=3)
    speed = recommend_models(models, "speed", top_k=3)
    balanced = recommend_models(models, "balanced", top_k=3)

    assert accuracy["selected"] == "hf:accurate"
    assert speed["selected"] == "hf:fast"
    assert balanced["selected"] == "hf:middle"
    assert set(balanced["by_priority"]) == {"accuracy", "speed", "balanced"}
    assert balanced["weights"] == {"accuracy": 0.5, "speed": 0.3, "cost": 0.2}
    assert balanced["rationale"]
    scores = {row["spec"]: row for row in balanced["scores"]}
    assert scores["hf:middle"]["balanced_score"] == pytest.approx(0.5417, abs=0.001)
    assert scores["hf:accurate"]["accuracy_score"] == pytest.approx(1.0)
    assert scores["hf:fast"]["speed_score"] == pytest.approx(1.0)


def test_recommendation_with_no_successful_models() -> None:
    recommendation = recommend_models([{"spec": "x:y", "status": "error"}], "balanced")
    assert recommendation["selected"] is None
    assert recommendation["scores"] == []
    assert "No model" in recommendation["rationale"]


def test_html_report_is_standalone_and_escapes_script_breakouts(sample_results: dict) -> None:
    html = render_html(sample_results)

    assert "<title>Test report</title>" in html
    assert "Best model for your priority" in html
    assert 'id="chart-scatter"' in html
    assert 'id="chart-metrics"' in html
    assert 'id="chart-latency"' in html
    assert 'id="chart-cost"' in html
    assert "hf:accurate" in html
    assert "Accuracy-first" in html
    assert "Speed-first" in html
    assert "<script src=" not in html
    assert "</script><script>alert(1)</script>" not in html
    assert "<\\/script>" in html
    assert "__TITLE__" not in html
    assert "__DATA__" not in html
    assert "__BODY__" not in html


def test_write_reports_creates_all_artifacts(sample_results: dict, tmp_path: Path) -> None:
    artifacts = write_reports(sample_results, tmp_path)

    assert artifacts["results_json"].is_file()
    assert artifacts["results_csv"].is_file()
    assert artifacts["queries_csv"].is_file()
    assert artifacts["report_html"].is_file()

    payload = json.loads(artifacts["results_json"].read_text(encoding="utf-8"))
    assert payload["artifacts"]["results_json"] == str(artifacts["results_json"])
    assert payload["recommendation"]["selected"] == "hf:middle"

    with artifacts["results_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["spec"] == "hf:accurate"
    assert "ndcg" in rows[0]

    with artifacts["queries_csv"].open(newline="", encoding="utf-8") as handle:
        query_rows = list(csv.DictReader(handle))
    assert len(query_rows) == 2
    assert query_rows[0]["model"] == "hf:accurate"


def test_write_html_report_priority_override(sample_results: dict, tmp_path: Path) -> None:
    target = tmp_path / "speed.html"
    write_html_report(sample_results, target, priority="speed", title="Speed review")
    text = target.read_text(encoding="utf-8")
    assert "<title>Speed review</title>" in text
    assert "Best for Speed-first" in text
    assert "hf:fast" in text
