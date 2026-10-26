from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from embedbench.cli import app
from embedbench.ollama_client import OllamaClient
from tests.fake_backend import FakeEmbeddingBackend  # noqa: F401 - registers backend


def write_corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text(
        "# Alpha Gateway\n\n"
        "The Alpha gateway listens on port 8443 for telemetry ingestion. "
        "It stores raw telemetry for thirty days and accepts batches of up to 500 events.\n\n"
        "## Operations\n\n"
        "Use the health endpoint on port 8787. Restart the gateway with the alphactl restart command.\n",
        encoding="utf-8",
    )
    (docs / "beta.md").write_text(
        "# Beta Billing\n\n"
        "Beta uses a monthly subscription with a Growth plan at 499 dollars per month. "
        "The plan monitors up to 50 robots and includes two million API calls.\n\n"
        "Annual subscriptions receive a 15 percent discount. Monthly fees are non-refundable.\n",
        encoding="utf-8",
    )
    (docs / "gamma.md").write_text(
        "# Gamma Troubleshooting\n\n"
        "Error GAMMA-500 means the local spool is full. Check outbound HTTPS connectivity and free disk space.\n\n"
        "If the error persists, run gamma doctor and inspect logs under /var/log/gamma.\n",
        encoding="utf-8",
    )
    return docs


def write_evalset(path: Path) -> None:
    examples = [
        {
            "query_id": "q1",
            "query": "Which port does the Alpha gateway use for telemetry?",
            "relevant_doc_ids": ["alpha.md"],
        },
        {
            "query_id": "q2",
            "query": "How many robots does the Beta Growth plan support?",
            "relevant_doc_ids": ["beta.md"],
        },
        {
            "query_id": "q3",
            "query": "What should I check when I see error GAMMA-500?",
            "relevant_doc_ids": ["gamma.md"],
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8")


def write_config(tmp_path: Path, docs: Path, evalset: Path) -> Path:
    config = {
        "docs": str(docs),
        "models": ["fake:hash"],
        "top_k": 2,
        "seed": 11,
        "output_dir": str(tmp_path / "results"),
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 200,
            "chunk_overlap": 40,
            "min_chunk_chars": 40,
        },
        "evalset": {"path": str(evalset)},
        "backends": {"extra": {"fake": {"dimension": 256}}},
        "cache": {"enabled": True, "dir": str(tmp_path / "cache")},
        "report": {
            "title": "Smoke test report",
            "priority": "balanced",
            "cost_per_1m_tokens": 0.02,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def load_results(directory: Path) -> dict:
    return json.loads((directory / "results.json").read_text(encoding="utf-8"))


def test_end_to_end_cli_run_cache_report_and_failures(tmp_path: Path) -> None:
    docs = write_corpus(tmp_path)
    evalset = tmp_path / "evalset.jsonl"
    write_evalset(evalset)
    config = write_config(tmp_path, docs, evalset)
    results_dir = tmp_path / "results"
    runner = CliRunner()

    first = runner.invoke(
        app,
        ["run", "--config", str(config), "--top-k", "3", "--no-progress"],
    )
    assert first.exit_code == 0, first.output + repr(first.exception)

    for name in (
        "results.json",
        "results.csv",
        "queries.csv",
        "report.html",
        "config.snapshot.yaml",
        "run.log",
    ):
        assert (results_dir / name).is_file(), name

    results = load_results(results_dir)
    assert results["config"]["top_k"] == 3
    assert results["run"]["successful_models"] == 1
    assert results["run"]["failed_models"] == 0
    model = results["models"][0]
    assert model["spec"] == "fake:hash"
    assert model["status"] == "ok"
    assert model["accuracy"]["n_queries"] == 3
    assert model["index"]["dimension"] == 256
    assert model["index"]["backend"] in {"faiss", "numpy"}
    assert 0.0 <= model["accuracy"]["ndcg"] <= 1.0
    assert 0.0 <= model["accuracy"]["recall"] <= 1.0
    assert model["latency_ms"]["p95_ms"] is not None
    assert model["timing"]["indexing_seconds"] >= 0
    assert model["cache"]["misses"] == results["run"]["corpus"]["chunks"]
    assert results["recommendation"]["selected"] == "fake:hash"
    assert results["run"]["evalset"]["source"] == "jsonl"

    html = (results_dir / "report.html").read_text(encoding="utf-8")
    assert "Smoke test report" in html
    assert "Best model for your priority" in html
    assert 'id="chart-scatter"' in html
    assert "<script src=" not in html

    snapshot = yaml.safe_load((results_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["top_k"] == 3

    # Warm cache: the same corpus should be a full cache hit and avoid embedding.
    second = runner.invoke(
        app,
        ["run", "--config", str(config), "--top-k", "3", "--no-progress"],
    )
    assert second.exit_code == 0, second.output + repr(second.exception)
    warm = load_results(results_dir)["models"][0]
    assert warm["cache"]["hits"] == warm["timing"]["cached_chunks"]
    assert warm["cache"]["misses"] == 0
    assert warm["timing"]["embedded_chunks"] == 0
    assert warm["cache"]["hit_rate"] == 1.0

    # Report command can re-render with another priority.
    report_path = tmp_path / "speed_report.html"
    report = runner.invoke(
        app,
        [
            "report",
            "--results",
            str(results_dir),
            "--output",
            str(report_path),
            "--priority",
            "speed",
        ],
    )
    assert report.exit_code == 0, report.output + repr(report.exception)
    assert report_path.is_file()
    assert "Best for Speed-first" in report_path.read_text(encoding="utf-8")

    # One failing model does not prevent the other model or the artifacts.
    partial_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    partial_config["models"] = ["fake:hash", "fake:missing"]
    partial_config["output_dir"] = str(tmp_path / "partial")
    partial_path = tmp_path / "partial.yaml"
    partial_path.write_text(yaml.safe_dump(partial_config, sort_keys=False), encoding="utf-8")
    partial = runner.invoke(
        app,
        ["run", "--config", str(partial_path), "--no-progress"],
    )
    assert partial.exit_code == 0, partial.output + repr(partial.exception)
    partial_results = load_results(tmp_path / "partial")
    assert partial_results["run"]["successful_models"] == 1
    assert partial_results["run"]["failed_models"] == 1
    failures = [model for model in partial_results["models"] if model["status"] == "error"]
    assert failures and "not installed" in failures[0]["error"]
    assert (tmp_path / "partial" / "report.html").is_file()

    # list-models is offline-safe and emits valid JSON.
    listing = runner.invoke(
        app,
        ["list-models", "--json", "--ollama-host", "http://127.0.0.1:1"],
    )
    assert listing.exit_code == 0, listing.output + repr(listing.exception)
    payload = json.loads(listing.output)
    assert payload["ollama_available"] is False
    assert any(model["spec"] == "hf:BAAI/bge-small-en-v1.5" for model in payload["models"])
    assert "fake" in payload["backends"]


def test_generate_evalset_cli_with_mocked_ollama(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "limits.md").write_text(
        "# Limits\n\nThe maximum telemetry payload is 5 MB and the gateway listens on port 8443. "
        "Batches may contain up to 500 events, and raw records are kept for 30 days.\n",
        encoding="utf-8",
    )
    output = tmp_path / "generated.jsonl"

    monkeypatch.setattr(OllamaClient, "require_model", lambda self, model: None)
    monkeypatch.setattr(
        OllamaClient,
        "generate",
        lambda self, model, prompt, **kwargs: json.dumps(
            {
                "questions": [
                    {
                        "question": "What is the maximum telemetry payload size?",
                        "answer": "5 MB",
                    }
                ]
            }
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "generate-evalset",
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--model",
            "fake-llm",
            "--max-chunks",
            "1",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output + repr(result.exception)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["query"] == "What is the maximum telemetry payload size?"
    assert records[0]["relevant_doc_ids"] == ["limits.md"]
    assert records[0]["source_chunk_id"].startswith("limits.md#")
    assert records[0]["metadata"]["generator_model"] == "fake-llm"


def test_auto_evalset_run_and_all_models_failing(tmp_path: Path, monkeypatch) -> None:
    docs = write_corpus(tmp_path)
    evalset = tmp_path / "evalset.jsonl"
    write_evalset(evalset)

    monkeypatch.setattr(OllamaClient, "require_model", lambda self, model: None)
    monkeypatch.setattr(
        OllamaClient,
        "generate",
        lambda self, model, prompt, **kwargs: json.dumps(
            {
                "questions": [
                    {
                        "question": "What is the maximum telemetry payload size?",
                        "answer": "5 MB",
                    }
                ]
            }
        ),
    )

    auto_config = {
        "docs": str(docs),
        "models": ["fake:hash"],
        "top_k": 2,
        "output_dir": str(tmp_path / "auto"),
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 200,
            "chunk_overlap": 40,
            "min_chunk_chars": 40,
        },
        "evalset": {
            "auto_generate": True,
            "max_chunks": 3,
            "min_chunk_chars": 20,
            "dedupe_threshold": 1.0,
        },
        "backends": {"extra": {"fake": {"dimension": 256}}},
        "cache": {"enabled": False},
    }
    auto_path = tmp_path / "auto.yaml"
    auto_path.write_text(yaml.safe_dump(auto_config, sort_keys=False), encoding="utf-8")

    runner = CliRunner()
    auto = runner.invoke(app, ["run", "--config", str(auto_path), "--no-progress"])
    assert auto.exit_code == 0, auto.output + repr(auto.exception)
    auto_results = load_results(tmp_path / "auto")
    assert auto_results["run"]["evalset"]["source"] == "synthetic"
    assert auto_results["run"]["evalset"]["generation"]["chunks_used"] == 3
    assert auto_results["run"]["evalset"]["queries"] >= 1
    assert (tmp_path / "auto" / "evalset.generated.jsonl").is_file()

    # Every model failing is a clear nonzero exit, but artifacts are still written.
    fail_config = yaml.safe_load(auto_path.read_text(encoding="utf-8"))
    fail_config["models"] = ["fake:missing"]
    fail_config["evalset"] = {"path": str(evalset)}
    fail_config["output_dir"] = str(tmp_path / "failed")
    fail_path = tmp_path / "failed.yaml"
    fail_path.write_text(yaml.safe_dump(fail_config, sort_keys=False), encoding="utf-8")

    failed = runner.invoke(app, ["run", "--config", str(fail_path), "--no-progress"])
    assert failed.exit_code == 1
    failed_results = load_results(tmp_path / "failed")
    assert failed_results["run"]["successful_models"] == 0
    assert failed_results["models"][0]["status"] == "error"
    assert (tmp_path / "failed" / "report.html").is_file()
