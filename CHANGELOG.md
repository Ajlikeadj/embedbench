# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Trusted-publishing workflow (`.github/workflows/publish.yml`) and a step-by-step
  [PUBLISHING.md](PUBLISHING.md) guide for TestPyPI and PyPI releases.

### Fixed

- Package metadata now points `Homepage`, `Documentation`, `Issues` and `Repository` at
  the real repository instead of placeholder URLs.

## [0.2.0] — 2026-10-26

### Added

- **`src/` layout and hatchling packaging.** `pip install embedbench` works, with
  optional extras: `local`, `faiss`, `openai`, `cohere`, `voyage`, `docs`, `plots`,
  `all`, `dev`. Cloud SDKs, PyTorch, FAISS and matplotlib are no longer core
  dependencies.
- **`models/` provider package** with one `BaseEmbeddingModel` contract:
  - `models/local.py`: Ollama and sentence-transformers providers.
  - `models/providers.py`: OpenAI, Cohere and Voyage AI with request chunking,
    bounded exponential backoff, concurrent async batches, `tiktoken` token counts,
    input-type mapping and a list-price table.
  - `register_model` / `create_model`, entry points (`embedbench.models`) and `.py`
    plugin loading.
- **`Evaluator` and `BenchmarkResult`** (`embedbench.evaluator`) as a first-class Python
  API, plus a `bench` CLI command with JSON/Markdown/CSV/PNG artifacts and
  accuracy/speed/balanced/cost recommendations.
- **`datasets.py`** with `RetrievalDataset`, chunk-group-aware relevance, JSONL
  persistence, corpus loading and deterministic synthetic generators:
  `needle_in_haystack`, `chunk_size_sensitivity`, `query_variants`, `demo_dataset`.
- **`stress.py`** (`StressRunner`, `StressReport`) and the `stress needle`,
  `stress chunk-size` and `stress noisy` CLI commands.
- **`reporter.py`** for result-based Rich tables, Markdown, CSV, JSON and matplotlib
  plots (`latency-vs-quality`, `quality-bars`, `cost-vs-quality`).
- **Vectorised metrics** (`evaluate_rankings`, `evaluate_rankings_at_k`,
  `mean_metrics_by_k`) alongside the scalar functions, plus latency percentiles,
  QPS, memory footprint and cost estimators.
- **`embedbench providers`** command listing providers, extras, API-key variables and
  list prices.
- **CI**: Ruff lint/format, pytest on Python 3.10–3.13 with coverage, wheel/sdist build,
  `twine check`, and a clean-wheel smoke test.
- Tests for datasets, evaluator, models (including fake cloud SDKs), reporter, stress,
  CLI and the Ollama client.

### Changed

- `embedbench.models` (data classes) is now `embedbench.schemas`.
- `embedbench.backends` is a compatibility shim over `embedbench.models`; the old
  `EmbeddingBackend` name and `register_backend` alias still work.
- `BaseEmbeddingModel.embed_documents` / `embed_queries` now return
  `list[list[float]]`; arithmetic call sites use `embed_documents_matrix` /
  `embed_queries_matrix`.
- `embedbench run` reports the same metrics as before, now computed through the shared
  vectorised engine.
- Ruff formatting applied across the codebase; lint set expanded (E, F, W, I, B, UP,
  C4, SIM, RUF) with a Google-style docstring policy in mind.

### Removed

- Scratch `probes/` scripts from the repository root (they targeted the 0.1.x backend
  internals and were never part of the package).
- `sentence-transformers`, `faiss-cpu`, `pypdf` and `python-docx` from the core
  dependency list; install `embedbench[local,faiss,docs]` for the previous behaviour.

## [0.1.0]

Initial release: local document ingestion, chunking, synthetic eval-set generation,
Ollama and sentence-transformers backends, retrieval metrics, HTML/JSON/CSV reports and
a Typer CLI.
