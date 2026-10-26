"""Tests for the reporter: Markdown, CSV, JSON and plot fallbacks."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest
from rich.console import Console

from embedbench.datasets import demo_dataset
from embedbench.errors import ReportError
from embedbench.evaluator import Evaluator
from embedbench.reporter import (
    plot_cost_vs_quality,
    plot_latency_vs_quality,
    plot_quality_bars,
    render_console,
    render_markdown,
    write_json,
    write_queries_csv,
    write_reports,
    write_summary_csv,
)
from tests.fake_backend import FakeEmbeddingModel


@pytest.fixture
def result():  # type: ignore[no-untyped-def]
    evaluator = Evaluator(
        [FakeEmbeddingModel("demo", {"dimension": 64, "price_per_1m_tokens": 0.02})],
        k_values=(1, 3),
        index_backend="numpy",
        show_progress=False,
    )
    return evaluator.evaluate(demo_dataset())


def test_render_markdown_contains_sections(result) -> None:  # type: ignore[no-untyped-def]
    markdown = render_markdown(result)
    assert "# Embedding benchmark: demo" in markdown
    assert "## Dataset" in markdown
    assert "## Retrieval quality @ 3" in markdown
    assert "## Efficiency and cost" in markdown
    assert "## Recommendation (balanced)" in markdown
    assert "## Method" in markdown
    assert "fake:demo" in markdown
    assert "| ---" in markdown


def test_render_markdown_escapes_pipes_and_handles_failures(result) -> None:  # type: ignore[no-untyped-def]
    failing = Evaluator(
        [FakeEmbeddingModel("missing", {"dimension": 8})],
        k_values=(3,),
        index_backend="numpy",
        show_progress=False,
    ).evaluate(demo_dataset())
    markdown = render_markdown(failing)
    assert "## Failures" in markdown
    assert "not installed" in markdown


def test_render_console_writes_table(result) -> None:  # type: ignore[no-untyped-def]
    buffer = io.StringIO()
    console = Console(file=buffer, width=160, force_terminal=False)
    render_console(result, console)
    output = buffer.getvalue()
    assert "Retrieval comparison" in output
    assert "fake:demo" in output
    assert "Recommended (balanced)" in output


def test_write_reports_bundle(result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    artifacts = write_reports(result, tmp_path / "out", plots=False)
    assert set(artifacts) == {"results_json", "summary_csv", "queries_csv", "results_md"}
    for path in artifacts.values():
        assert path.is_file()

    payload = json.loads(Path(artifacts["results_json"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["recommendation"]["winner"] == "fake:demo"
    assert payload["models"][0]["spec"] == "fake:demo"

    markdown = Path(artifacts["results_md"]).read_text(encoding="utf-8")
    assert "fake:demo" in markdown


def test_summary_csv_has_one_row_per_model(result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = write_summary_csv(result, tmp_path / "summary.csv")
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    assert len(rows) == 1
    row = rows[0]
    assert row["spec"] == "fake:demo"
    assert row["status"] == "ok"
    assert float(row["ndcg"]) > 0.0
    assert float(row["usd"]) > 0.0
    assert int(row["vectors"]) == demo_dataset().num_documents


def test_queries_csv_has_one_row_per_query(result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = write_queries_csv(result, tmp_path / "queries.csv")
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    assert len(rows) == demo_dataset().num_queries
    assert rows[0]["query_id"] == "q1"
    assert "|" not in rows[0]["relevant_doc_ids"]


def test_write_json_is_utf8_and_complete(result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = write_json(result, tmp_path / "results.json", priority="speed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["recommendation"]["priority"] == "speed"


def test_plots_raise_clear_error_without_matplotlib(
    result, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ReportError, match="embedbench\\[plots\\]"):
        plot_latency_vs_quality(result, Path("unused.png"))
    with pytest.raises(ReportError, match="embedbench\\[plots\\]"):
        plot_quality_bars(result, Path("unused.png"))
    with pytest.raises(ReportError, match="embedbench\\[plots\\]"):
        plot_cost_vs_quality(result, Path("unused.png"))


def test_write_reports_skips_plots_when_unavailable(
    result, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    artifacts = write_reports(result, tmp_path / "out", plots=True)
    assert "latency_vs_quality_png" not in artifacts
    assert "quality_bars_png" not in artifacts
    assert all(path.suffix != ".png" for path in artifacts.values())


def test_plots_render_png_files(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from embedbench.evaluator import Evaluator

    evaluator = Evaluator(
        [
            FakeEmbeddingModel("cheap", {"dimension": 64, "price_per_1m_tokens": 0.02}),
            FakeEmbeddingModel("dear", {"dimension": 32, "price_per_1m_tokens": 0.10}),
        ],
        k_values=(1, 3),
        index_backend="numpy",
        show_progress=False,
    )
    bench = evaluator.evaluate(demo_dataset())

    latency = plot_latency_vs_quality(bench, tmp_path / "latency.png")
    bars = plot_quality_bars(bench, tmp_path / "bars.png")
    cost = plot_cost_vs_quality(bench, tmp_path / "cost.png")
    for path in (latency, bars, cost):
        assert path.is_file()
        assert path.stat().st_size > 1000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    artifacts = write_reports(bench, tmp_path / "bundle", plots=True)
    assert artifacts["latency_vs_quality_png"].is_file()
    assert artifacts["quality_bars_png"].is_file()
    assert artifacts["cost_vs_quality_png"].is_file()
