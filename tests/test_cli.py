"""CLI tests for the dataset evaluator, stress commands and provider catalogue."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from embedbench.cli import app
from tests.fake_backend import FakeEmbeddingModel  # noqa: F401 - registers "fake"


def _invoke(*args: str):  # type: ignore[no-untyped-def]
    return CliRunner().invoke(app, list(args))


def test_help_lists_every_command() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0, result.output
    for command in ("run", "bench", "stress", "providers", "report", "list-models"):
        assert command in result.output


def test_stress_help_lists_workloads() -> None:
    result = _invoke("stress", "--help")
    assert result.exit_code == 0, result.output
    for workload in ("needle", "chunk-size", "noisy"):
        assert workload in result.output


def test_providers_json_contains_extras_and_prices() -> None:
    result = _invoke("providers", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    providers = {row["provider"]: row for row in payload["providers"]}
    assert providers["openai"]["extra"] == "openai"
    assert providers["ollama"]["api_key_env"] is None
    prices = {(row["provider"], row["model"]) for row in payload["pricing"]}
    assert ("openai", "text-embedding-3-small") in prices


def test_bench_demo_writes_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "bench"
    result = _invoke(
        "bench",
        "--demo",
        "--models",
        "fake:demo",
        "--k",
        "1,3",
        "--index",
        "numpy",
        "--no-plots",
        "--output-dir",
        str(output),
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Recommended" in result.output
    for name in ("results.json", "results.md", "summary.csv", "queries.csv"):
        assert (output / name).is_file(), name
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["models"][0]["spec"] == "fake:demo"
    assert payload["k_values"] == [1, 3]


def test_bench_requires_a_dataset_source() -> None:
    result = _invoke("bench", "--models", "fake:demo")
    assert result.exit_code == 1
    assert "Provide --docs" in result.output


def test_bench_all_models_failing_exits_nonzero(tmp_path: Path) -> None:
    result = _invoke(
        "bench",
        "--demo",
        "--models",
        "fake:missing",
        "--index",
        "numpy",
        "--no-plots",
        "--output-dir",
        str(tmp_path / "failed"),
    )
    assert result.exit_code == 1
    assert (tmp_path / "failed" / "results.json").is_file()


def test_stress_noisy_rejects_unknown_noise_kind(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("# One\n\nalpha beta gamma delta epsilon", encoding="utf-8")
    evalset = tmp_path / "evalset.jsonl"
    evalset.write_text(
        json.dumps({"query_id": "q1", "query": "alpha beta", "relevant_doc_ids": ["one.md"]})
        + "\n",
        encoding="utf-8",
    )
    result = _invoke(
        "stress",
        "noisy",
        "--models",
        "fake:demo",
        "--docs",
        str(docs),
        "--evalset",
        str(evalset),
        "--noise",
        "gibberish",
        "--index",
        "numpy",
        "--output-dir",
        str(tmp_path / "stress"),
    )
    assert result.exit_code == 1
    assert "Unknown query noise kind" in result.output


def test_stress_chunk_size_runs(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    body = "Paragraph about retrieval quality and chunk boundaries. " * 30
    (docs / "one.md").write_text(f"# One\n\n{body}", encoding="utf-8")
    evalset = tmp_path / "evalset.jsonl"
    evalset.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "chunk boundaries",
                "relevant_doc_ids": ["one.md"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "stress"
    result = _invoke(
        "stress",
        "chunk-size",
        "--models",
        "fake:demo",
        "--docs",
        str(docs),
        "--evalset",
        str(evalset),
        "--chunk-sizes",
        "128,256",
        "--k",
        "1,3",
        "--index",
        "numpy",
        "--output-dir",
        str(output),
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert (output / "chunk-size.json").is_file()
    payload = json.loads((output / "chunk-size.json").read_text(encoding="utf-8"))
    assert payload["workload"] == "chunk-size"
    assert len(payload["cells"]) == 2
