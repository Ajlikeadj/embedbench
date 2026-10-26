"""End-to-end benchmark orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from embedbench import __version__
from embedbench.backends import create_backend, load_plugins
from embedbench.backends.base import EmbeddingBackend, heuristic_token_count
from embedbench.cache import EmbeddingCache
from embedbench.chunking import chunk_documents
from embedbench.config import RunConfig, save_config_snapshot
from embedbench.errors import BackendError, ConfigError, EvalsetError, IngestionError
from embedbench.evalset import (
    SyntheticEvalsetGenerator,
    load_evalset,
    save_evalset,
    validate_evalset_documents,
)
from embedbench.index import build_index
from embedbench.ingest import load_documents
from embedbench.logging_utils import get_logger
from embedbench.metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    dedupe_ranked_docs,
    evaluate_retrieval,
)
from embedbench.models import Chunk, EvalExample, ModelSpec, TokenCount, TokenUsage
from embedbench.ollama_client import OllamaClient, OllamaError
from embedbench.report import console_summary, recommend_models, write_reports
from embedbench.resources import ResourceSampler, environment_info
from embedbench.utils import (
    percentile_summary,
    set_global_seed,
    utc_now_iso,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class BenchmarkRun:
    """Results plus the paths of every generated artifact."""

    results: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(slots=True)
class _PreparedCorpus:
    examples: list[EvalExample]
    source: str
    evalset_path: Path | None
    generation: dict[str, Any] | None
    prep_seconds: float
    ingest_seconds: float
    chunk_seconds: float


def _unique_models(specs: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for spec in specs:
        cleaned = spec.strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _safe_count(backend: EmbeddingBackend, texts: Sequence[str], label: str) -> TokenCount:
    try:
        return backend.count_tokens(texts)
    except Exception as exc:  # token counting must not fail the retrieval benchmark
        logger.warning("Token counting failed for %s (%s); using heuristic", label, exc)
        return TokenCount(
            tokens=heuristic_token_count(texts),
            estimated=True,
            method="heuristic (tokenizer counting failed)",
        )


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _latency_payload(latencies_ms: Sequence[float]) -> dict[str, float | int | None]:
    summary = percentile_summary(latencies_ms)
    warm = latencies_ms[1:] if len(latencies_ms) > 1 else []
    warm_summary = percentile_summary(warm)
    total_seconds = sum(latencies_ms) / 1000.0
    return {
        "count": int(summary["count"]),
        "mean_ms": _round_or_none(summary["mean"]),
        "min_ms": _round_or_none(summary["min"]),
        "max_ms": _round_or_none(summary["max"]),
        "p50_ms": _round_or_none(summary["p50"]),
        "p95_ms": _round_or_none(summary["p95"]),
        "p99_ms": _round_or_none(summary["p99"]),
        "cold_query_ms": _round_or_none(latencies_ms[0]) if latencies_ms else None,
        "warm_p50_ms": _round_or_none(warm_summary["p50"]),
        "warm_p95_ms": _round_or_none(warm_summary["p95"]),
        "warm_p99_ms": _round_or_none(warm_summary["p99"]),
        "queries_per_second": (
            round(len(latencies_ms) / total_seconds, 3) if total_seconds > 0 else None
        ),
    }


def _prepare_evalset(
    config: RunConfig,
    chunks: Sequence[Chunk],
    *,
    console: Console,
    show_progress: bool,
) -> tuple[list[EvalExample], str, Path | None, dict[str, Any] | None]:
    evalset_config = config.evalset
    generation: dict[str, Any] | None = None
    source_path: Path | None = evalset_config.path
    output_dir = Path(config.output_dir)

    if evalset_config.path is not None:
        examples = load_evalset(evalset_config.path)
        source = "jsonl"
        console.print(
            f"[dim]Eval set:[/dim] {len(examples)} queries from {evalset_config.path}"
        )
    elif evalset_config.auto_generate:
        client = OllamaClient(
            config.backends.ollama.host,
            timeout_s=evalset_config.request_timeout_s,
            max_retries=config.backends.ollama.max_retries,
            retry_backoff_s=config.backends.ollama.retry_backoff_s,
        )
        try:
            generator = SyntheticEvalsetGenerator(
                client,
                evalset_config,
                seed=config.seed,
                show_progress=show_progress,
                console=console,
            )
            examples, stats = generator.generate(chunks)
        except OllamaError as exc:
            raise EvalsetError(str(exc)) from exc
        finally:
            client.close()
        source = "synthetic"
        generation = stats.to_dict()
        generated_path = output_dir / "evalset.generated.jsonl"
        save_evalset(examples, generated_path)
        source_path = generated_path
        console.print(
            f"[dim]Generated[/dim] {len(examples)} questions with "
            f"{evalset_config.ollama_model} -> {generated_path}"
        )
    else:
        raise EvalsetError(
            "No evaluation set configured. Provide evalset.path in the config or "
            "--evalset FILE, or enable synthetic generation with --auto-evalset "
            "(requires Ollama)."
        )

    validate_evalset_documents(examples, {chunk.doc_id for chunk in chunks})
    return examples, source, source_path, generation


def _prepare_corpus(
    config: RunConfig,
    *,
    console: Console,
    show_progress: bool,
) -> tuple[Any, list[Chunk], _PreparedCorpus]:
    if config.docs is None:
        raise ConfigError("No documents folder configured. Pass --docs PATH or set 'docs'.")
    if not _unique_models(config.models):
        raise ConfigError("No models configured. Pass --models or set 'models' in the config.")

    prep_start = time.perf_counter()
    ingest_start = time.perf_counter()
    ingestion = load_documents(config.docs, config.ingestion)
    ingest_seconds = time.perf_counter() - ingest_start

    chunk_start = time.perf_counter()
    chunks = chunk_documents(ingestion.documents, config.chunking)
    chunk_seconds = time.perf_counter() - chunk_start
    if not chunks:
        raise IngestionError(
            "Documents were loaded but produced no chunks. Lower chunk sizing or "
            "check that the files contain text."
        )
    console.print(
        f"[dim]Corpus:[/dim] {len(ingestion.documents)} documents, "
        f"{len(chunks)} chunks ({ingestion.total_chars:,} characters)"
    )

    examples, source, evalset_path, generation = _prepare_evalset(
        config, chunks, console=console, show_progress=show_progress
    )

    chunked_doc_ids = {chunk.doc_id for chunk in chunks}
    docs_without_chunks = sorted(
        {
            doc_id
            for example in examples
            for doc_id in example.relevant_doc_ids
            if doc_id not in chunked_doc_ids
        }
    )
    if docs_without_chunks:
        raise EvalsetError(
            "Evaluation queries reference documents that produced no chunks: "
            + ", ".join(docs_without_chunks[:5])
        )

    prep_seconds = time.perf_counter() - prep_start
    prepared = _PreparedCorpus(
        examples=examples,
        source=source,
        evalset_path=evalset_path,
        generation=generation,
        prep_seconds=prep_seconds,
        ingest_seconds=ingest_seconds,
        chunk_seconds=chunk_seconds,
    )
    return ingestion, chunks, prepared


def _embed_chunks(
    backend: EmbeddingBackend,
    chunks: Sequence[Chunk],
    cache: EmbeddingCache,
    *,
    console: Console,
    show_progress: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Embed chunks with per-text caching; return vectors and timing/cache payloads."""

    texts = [chunk.text for chunk in chunks]
    namespace = backend.cache_namespace
    total = len(texts)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    use_progress = show_progress and console.is_terminal
    task_id = None
    if use_progress:
        progress.start()
        task_id = progress.add_task(f"Embedding {backend.name}", total=total)

    index_start = time.perf_counter()
    try:
        cached = cache.get_many(namespace, texts)
        known_dimension: int | None = backend.known_dimension
        sanitized: list[np.ndarray | None] = []
        cache_hits = 0
        for position, vector in enumerate(cached):
            if vector is not None:
                if known_dimension is None:
                    known_dimension = int(vector.shape[0])
                if vector.shape[0] != known_dimension:
                    logger.warning(
                        "Ignoring cached vector with dimension %d (expected %d) for %s",
                        vector.shape[0],
                        known_dimension,
                        chunks[position].chunk_id,
                    )
                    vector = None
            if vector is not None:
                cache_hits += 1
            sanitized.append(vector)
        if task_id is not None:
            progress.update(task_id, completed=cache_hits)

        missing = [index for index, vector in enumerate(sanitized) if vector is None]
        cache_misses = len(missing)
        logger.info(
            "%s: %d cache hits, %d chunks to embed",
            backend.name,
            cache_hits,
            cache_misses,
        )

        batch_timings_ms: list[float] = []
        embedded = 0
        for offset in range(0, len(missing), backend.max_batch_size):
            batch_indices = missing[offset : offset + backend.max_batch_size]
            batch_texts = [texts[index] for index in batch_indices]
            batch_start = time.perf_counter()
            vectors = backend.embed_documents(batch_texts)
            elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_timings_ms.append(elapsed_ms)
            cache.put_many(namespace, batch_texts, vectors)
            for local_index, chunk_index in enumerate(batch_indices):
                sanitized[chunk_index] = vectors[local_index]
            embedded += len(batch_indices)
            if task_id is not None:
                progress.advance(task_id, len(batch_indices))

        embedding_seconds = sum(batch_timings_ms) / 1000.0
        if any(vector is None for vector in sanitized):
            raise BackendError(f"{backend.name} produced incomplete embeddings")
        matrix = np.stack([vector for vector in sanitized if vector is not None])
        if matrix.shape[0] != total:
            raise BackendError(
                f"{backend.name} returned {matrix.shape[0]} vectors for {total} chunks"
            )
        index_seconds = time.perf_counter() - index_start
        warm_batches = batch_timings_ms[1:] if len(batch_timings_ms) > 1 else []
        warm_summary = percentile_summary(warm_batches)
        timing = {
            "embedding_seconds": round(embedding_seconds, 4),
            "indexing_seconds": round(index_seconds, 4),
            "batches": len(batch_timings_ms),
            "embedded_chunks": embedded,
            "cached_chunks": cache_hits,
            "cold_embed_ms": _round_or_none(batch_timings_ms[0]) if batch_timings_ms else None,
            "warm_batch_p50_ms": _round_or_none(warm_summary["p50"]),
            "warm_batch_p95_ms": _round_or_none(warm_summary["p95"]),
            "warm_batch_p99_ms": _round_or_none(warm_summary["p99"]),
        }
        cache_payload = {
            "enabled": cache.enabled,
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hits / total, 4) if total else 0.0,
        }
        return matrix, timing, cache_payload
    finally:
        if use_progress:
            progress.stop()


def _query_latencies(
    backend: EmbeddingBackend,
    index: Any,
    chunks: Sequence[Chunk],
    examples: Sequence[EvalExample],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[float], list[RetrievalMetrics]]:
    per_query: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    metric_rows: list[RetrievalMetrics] = []

    for example in examples:
        query_start = time.perf_counter()
        query_vector = backend.embed_queries([example.query])[0]
        scores, positions = index.search(query_vector, top_k)
        elapsed_ms = (time.perf_counter() - query_start) * 1000.0
        latencies_ms.append(elapsed_ms)

        retrieved: list[dict[str, Any]] = []
        raw_doc_ids: list[str] = []
        for rank, position in enumerate(positions.tolist(), start=1):
            chunk = chunks[int(position)]
            score = float(scores[rank - 1])
            raw_doc_ids.append(chunk.doc_id)
            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "score": round(score, 6),
                }
            )
        ranked_doc_ids = dedupe_ranked_docs(raw_doc_ids, top_k)
        metrics = evaluate_retrieval(example.relevant_doc_ids, ranked_doc_ids, top_k)
        metric_rows.append(metrics)
        per_query.append(
            {
                "query_id": example.query_id,
                "query": example.query,
                "relevant_doc_ids": list(example.relevant_doc_ids),
                "retrieved": retrieved,
                "ranked_doc_ids": ranked_doc_ids,
                "metrics": metrics.to_dict(),
                "latency_ms": round(elapsed_ms, 3),
                "prefix_used": backend.query_prefix,
            }
        )
        if not retrieved:
            logger.warning(
                "Query %s returned no results; index has %d chunks",
                example.query_id,
                len(chunks),
            )

    if not metric_rows:
        raise EvalsetError("No queries were evaluated")
    return per_query, latencies_ms, metric_rows


def _resources_payload(
    backend: EmbeddingBackend,
    sampler: ResourceSampler,
) -> dict[str, Any]:
    ram = sampler.stop()
    try:
        runtime = backend.runtime_metadata()
    except Exception as exc:  # metadata must never invalidate a completed benchmark
        logger.warning("Backend runtime metadata failed: %s", exc)
        runtime = {}
    resources = {**ram, **runtime}
    if runtime.get("gpu_peak_allocated_mb") is not None:
        resources["peak_vram_mb"] = runtime["gpu_peak_allocated_mb"]
        resources["vram_measurement"] = "torch_cuda_peak_allocated"
    elif runtime.get("model_resident_vram_bytes") is not None:
        resources["peak_vram_mb"] = round(
            float(runtime["model_resident_vram_bytes"]) / (1024 * 1024), 2
        )
        resources["vram_measurement"] = "ollama_api_ps_resident"
    else:
        resources["peak_vram_mb"] = None
        resources["vram_measurement"] = None
    if runtime.get("model_resident_bytes") is not None:
        resources["model_resident_mb"] = round(
            float(runtime["model_resident_bytes"]) / (1024 * 1024), 2
        )
        resources["ram_measurement"] = "ollama_api_ps_resident"
    else:
        resources["model_resident_mb"] = None
    return resources


def _benchmark_model(
    spec: str,
    config: RunConfig,
    chunks: Sequence[Chunk],
    examples: Sequence[EvalExample],
    cache: EmbeddingCache,
    *,
    console: Console,
    show_progress: bool,
) -> dict[str, Any]:
    global_start = time.perf_counter()
    backend: EmbeddingBackend | None = None
    sampler = ResourceSampler()
    model_start = time.perf_counter()
    try:
        parsed = ModelSpec.parse(spec)
    except ValueError as exc:
        return {
            "spec": spec,
            "backend": spec.partition(":")[0] or None,
            "model": spec.partition(":")[2] or None,
            "status": "error",
            "error": str(exc),
            "duration_seconds": round(time.perf_counter() - global_start, 3),
        }

    try:
        backend = create_backend(spec, config)
        sampler.start()
        backend.reset_peak_memory()
        load_start = time.perf_counter()
        backend.load()
        load_ms = (time.perf_counter() - load_start) * 1000.0

        document_texts = [chunk.text for chunk in chunks]
        prepared_documents = backend.prepare_documents(document_texts)
        index_tokens = _safe_count(backend, prepared_documents, f"{spec} documents")
        prepared_queries = backend.prepare_queries([example.query for example in examples])
        query_tokens = _safe_count(backend, prepared_queries, f"{spec} queries")

        vectors, embedding_timing, cache_payload = _embed_chunks(
            backend, chunks, cache, console=console, show_progress=show_progress
        )

        index_start = time.perf_counter()
        index = build_index(vectors)
        index_build_ms = (time.perf_counter() - index_start) * 1000.0
        try:
            index_disk_bytes = index.disk_size_bytes()
        except Exception as exc:
            logger.warning("Could not measure index size on disk: %s", exc)
            index_disk_bytes = None

        per_query, latencies, metric_rows = _query_latencies(
            backend, index, chunks, examples, top_k=config.top_k
        )
        accuracy = aggregate_retrieval_metrics(metric_rows)
        total_tokens = index_tokens.tokens + query_tokens.tokens
        usage = TokenUsage(
            index_tokens=index_tokens.tokens,
            query_tokens=query_tokens.tokens,
            total_tokens=total_tokens,
            estimated=index_tokens.estimated or query_tokens.estimated,
            method=(
                index_tokens.method
                if index_tokens.method == query_tokens.method
                else f"docs: {index_tokens.method}; queries: {query_tokens.method}"
            ),
            cost_usd=round(total_tokens / 1_000_000 * config.report.cost_per_1m_tokens, 6),
            price_per_1m_tokens=config.report.cost_per_1m_tokens,
        )

        resources = _resources_payload(backend, sampler)
        total_seconds = time.perf_counter() - model_start
        embedding_seconds = float(embedding_timing["embedding_seconds"])
        indexing_seconds = float(embedding_timing["indexing_seconds"]) + index_build_ms / 1000.0
        embedded_chunks = int(embedding_timing["embedded_chunks"])
        n_chunks = len(chunks)
        throughput = {
            "chunks_per_second": round(n_chunks / indexing_seconds, 3)
            if indexing_seconds > 0
            else None,
            "embed_chunks_per_second": round(embedded_chunks / embedding_seconds, 3)
            if embedded_chunks > 0 and embedding_seconds > 0
            else None,
            "queries_per_second": (
                round(len(latencies) / (sum(latencies) / 1000.0), 3)
                if sum(latencies) > 0
                else None
            ),
        }

        return {
            "spec": spec,
            "backend": parsed.backend,
            "model": parsed.model,
            "status": "ok",
            "error": None,
            "metrics_k": config.top_k,
            "accuracy": accuracy,
            "latency_ms": _latency_payload(latencies),
            "timing": {
                "model_load_ms": round(load_ms, 3),
                **embedding_timing,
                "index_build_ms": round(index_build_ms, 3),
                "indexing_seconds": round(indexing_seconds, 4),
                "total_seconds": round(total_seconds, 4),
            },
            "throughput": throughput,
            "tokens": usage.to_dict(),
            "index": {
                "backend": index.name,
                "dimension": int(index.dimension),
                "vectors": int(index.ntotal),
                "disk_size_bytes": index_disk_bytes,
                "disk_size_mb": round(index_disk_bytes / (1024 * 1024), 3)
                if index_disk_bytes is not None
                else None,
            },
            "cache": cache_payload,
            "resources": resources,
            "per_query": per_query,
        }
    except (BackendError, EvalsetError) as exc:
        logger.error("Model %s failed: %s", spec, exc)
        return {
            "spec": spec,
            "backend": parsed.backend,
            "model": parsed.model,
            "status": "error",
            "error": str(exc),
            "duration_seconds": round(time.perf_counter() - global_start, 3),
        }
    except Exception as exc:  # unexpected backend-specific bug: record and continue
        logger.exception("Unexpected failure for model %s", spec)
        return {
            "spec": spec,
            "backend": parsed.backend,
            "model": parsed.model,
            "status": "error",
            "error": f"Unexpected error: {type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - global_start, 3),
        }
    finally:
        sampler.stop()
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # pragma: no cover - cleanup should not mask results
                logger.warning("Error closing backend %s: %s", spec, exc)


def _corpus_payload(
    ingestion: Any,
    chunks: Sequence[Chunk],
    prepared: _PreparedCorpus,
    config: RunConfig,
) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    examples = prepared.examples
    payload = {
        "documents": len(ingestion.documents),
        "chunks": len(chunks),
        "chars": ingestion.total_chars,
        "avg_chunk_chars": round(float(np.mean(lengths)), 2),
        "median_chunk_chars": round(float(np.median(lengths)), 2),
        "min_chunk_chars": int(min(lengths)),
        "max_chunk_chars": int(max(lengths)),
        "documents_detail": [document.to_dict() for document in ingestion.documents],
        "skipped_files": [item.to_dict() for item in ingestion.skipped],
        "ingest_seconds": round(prepared.ingest_seconds, 4),
        "chunk_seconds": round(prepared.chunk_seconds, 4),
        "prep_seconds": round(prepared.prep_seconds, 4),
        "chunking": config.chunking.model_dump(mode="json"),
        "queries": len(examples),
        "avg_query_words": round(
            float(np.mean([len(example.query.split()) for example in examples])), 2
        ),
    }
    return payload


def run_benchmark(
    config: RunConfig,
    *,
    console: Console | None = None,
    show_progress: bool = True,
) -> BenchmarkRun:
    """Run ingestion, evaluation and reporting.

    Setup errors (bad docs, bad eval set, unreachable Ollama during generation)
    raise.  A failing model is recorded under ``status: error`` and the remaining
    models still run.
    """

    console = console or Console()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir / "config.snapshot.yaml")
    load_plugins(config.plugins)
    set_global_seed(config.seed)

    started = time.perf_counter()
    ingestion, chunks, prepared = _prepare_corpus(
        config, console=console, show_progress=show_progress
    )
    examples = prepared.examples
    models = _unique_models(config.models)
    results_models: list[dict[str, Any]] = []

    cache = EmbeddingCache(Path(config.cache.dir), enabled=config.cache.enabled)
    try:
        for spec in models:
            set_global_seed(config.seed)
            console.print(f"\n[bold]Benchmarking[/bold] {spec}")
            result = _benchmark_model(
                spec,
                config,
                chunks,
                examples,
                cache,
                console=console,
                show_progress=show_progress,
            )
            results_models.append(result)
            if result["status"] == "ok":
                accuracy = result["accuracy"]
                latency = result["latency_ms"]
                p95 = latency["p95_ms"]
                p95_text = f"{p95:.1f}" if isinstance(p95, (int, float)) else "n/a"
                console.print(
                    f"  nDCG@{config.top_k}={accuracy['ndcg']:.3f}  "
                    f"MRR={accuracy['mrr']:.3f}  "
                    f"p95={p95_text} ms"
                )
            else:
                console.print(f"[red]  failed:[/red] {result['error']}")
    finally:
        cache.close()

    total_seconds = time.perf_counter() - started
    successful = sum(1 for model in results_models if model["status"] == "ok")
    recommendation = recommend_models(results_models, config.report.priority)
    results: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "tool": {"name": "embedbench", "version": __version__},
        "config": config.model_dump(mode="json"),
        "run": {
            "seed": config.seed,
            "top_k": config.top_k,
            "priority": config.report.priority,
            "docs_dir": str(config.docs) if config.docs else None,
            "output_dir": str(output_dir),
            "models_requested": models,
            "successful_models": successful,
            "failed_models": len(results_models) - successful,
            "total_seconds": round(total_seconds, 3),
            "environment": environment_info(),
            "corpus": _corpus_payload(ingestion, chunks, prepared, config),
            "evalset": {
                "source": prepared.source,
                "path": str(prepared.evalset_path) if prepared.evalset_path else None,
                "queries": len(examples),
                "generation": prepared.generation,
                "examples": [example.to_dict() for example in examples],
            },
            "price_per_1m_tokens": config.report.cost_per_1m_tokens,
        },
        "models": results_models,
        "recommendation": recommendation,
    }
    artifacts = write_reports(results, output_dir)
    if console is not None:
        console_summary(results, console)
    logger.info("Results written to %s", artifacts["results_json"])
    return BenchmarkRun(results=results, artifacts=artifacts)
