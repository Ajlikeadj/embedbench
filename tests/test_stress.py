"""Tests for the stress workloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embedbench.datasets import demo_dataset
from embedbench.stress import StressReport, StressRunner
from tests.fake_backend import FakeEmbeddingModel


def _runner(*models: FakeEmbeddingModel) -> StressRunner:
    return StressRunner(
        list(models),
        k_values=(1, 3),
        batch_size=16,
        index_backend="numpy",
        show_progress=False,
    )


def test_needle_workload_reports_per_context_cells() -> None:
    runner = _runner(FakeEmbeddingModel("demo", {"dimension": 64}))
    report = runner.needle_in_haystack(
        context_sizes=(256, 512),
        depths=(0.0, 0.5, 1.0),
        chunk_size=200,
        seed=3,
    )

    assert isinstance(report, StressReport)
    assert report.kind == "needle"
    assert len(report.cells) == 2
    assert all(cell["status"] == "ok" for cell in report.cells)
    assert {cell["config"]["context_tokens"] for cell in report.cells} == {256, 512}
    assert report.cells[0]["documents"] >= 1
    assert report.cells[0]["queries"]
    assert "queries_total" in report.cells[0]

    payload = report.to_dict()
    assert payload["workload"] == "needle"
    json.dumps(payload)

    markdown = report.markdown()
    assert "# Stress workload: needle" in markdown
    assert "## Depth breakdown" in markdown


def test_chunk_size_workload_uses_document_groups() -> None:
    dataset = demo_dataset()
    runner = _runner(FakeEmbeddingModel("demo", {"dimension": 64}))
    report = runner.chunk_size_sensitivity(
        dataset.documents,
        dataset.queries,
        chunk_tokens=(128, 256),
    )

    assert report.kind == "chunk-size"
    assert len(report.cells) == 2
    assert {cell["config"]["chunk_tokens"] for cell in report.cells} == {128, 256}
    for cell in report.successful_cells():
        assert cell["metrics"]["ndcg"] >= 0.0
        assert cell["documents"] == dataset.num_documents


def test_query_noise_workload_includes_clean_baseline() -> None:
    dataset = demo_dataset()
    runner = _runner(FakeEmbeddingModel("demo", {"dimension": 64}))
    report = runner.query_robustness(dataset, kinds=("typo", "truncate"), seed=5)

    assert report.kind == "query-noise"
    variants = {cell["config"]["variant"] for cell in report.cells}
    assert variants == {"clean", "typo", "truncate"}
    markdown = report.markdown()
    assert "## Robustness vs clean baseline" in markdown
    assert "typo" in markdown


def test_stress_report_writes_json_and_markdown(tmp_path: Path) -> None:
    dataset = demo_dataset()
    report = _runner(FakeEmbeddingModel("demo", {"dimension": 32})).query_robustness(
        dataset, kinds=("typo",), seed=2
    )
    artifacts = report.write(tmp_path / "stress", stem="query-noise")
    assert artifacts["json"].is_file()
    assert artifacts["markdown"].is_file()
    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert payload["kind"] == "stress"
    assert payload["models"] == ["fake:demo"]


def test_stress_records_failing_models() -> None:
    runner = _runner(FakeEmbeddingModel("missing", {"dimension": 16}))
    report = runner.query_robustness(demo_dataset(), kinds=("typo",), seed=1)

    assert report.successful_cells() == []
    assert all(cell["status"] == "error" for cell in report.cells)
    assert "## Failures" in report.markdown()


def test_stress_requires_models_and_documents() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        StressRunner([])
    runner = _runner(FakeEmbeddingModel("demo", {"dimension": 8}))
    dataset = demo_dataset()
    with pytest.raises(Exception, match="at least one document"):
        runner.chunk_size_sensitivity([], dataset.queries)
