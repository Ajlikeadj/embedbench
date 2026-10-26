"""Stress workloads built on top of the evaluator.

Three routines cover the failure modes that matter in production retrieval:

``needle_in_haystack``
    Can the model find one fact inside a 2k, 8k or 32k token context, and does
    accuracy decay with context length or insertion depth?

``chunk_size_sensitivity``
    How much quality is lost or gained when the same corpus is chunked at 128,
    256, 512 or 1024 tokens?

``query_robustness``
    How far does quality drop when users typo, truncate, paraphrase or shuffle
    their questions?  The clean dataset is the baseline.

Each routine returns a :class:`StressReport`, a flat list of cells (one per
configuration and model) that serialises to JSON and renders to Markdown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from embedbench.datasets import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_CONTEXT_SIZES,
    DEFAULT_DEPTHS,
    DEFAULT_NOISE_KINDS,
    DatasetDocument,
    RetrievalDataset,
    RetrievalQuery,
    chunk_size_sensitivity as build_chunk_datasets,
    needle_in_haystack as build_needle_datasets,
    query_variants,
)
from embedbench.errors import ReportError
from embedbench.evaluator import BenchmarkResult, Evaluator, ModelResult
from embedbench.logging_utils import get_logger
from embedbench.models import BaseEmbeddingModel
from embedbench.utils import atomic_write_text, utc_now_iso, write_json

logger = get_logger(__name__)


@dataclass(slots=True)
class StressReport:
    """Aggregated stress results.

    Attributes:
        kind: ``needle``, ``chunk-size`` or ``query-noise``.
        metric_k: Cut-off used for the headline columns and the deltas.
        settings: Generator settings for reproducibility.
        cells: One mapping per configuration and model.
        generated_at: ISO-8601 UTC timestamp.
    """

    kind: str
    metric_k: int
    settings: dict[str, Any]
    cells: list[dict[str, Any]]
    generated_at: str = field(default_factory=utc_now_iso)

    @property
    def models(self) -> list[str]:
        """Distinct model specs in first-seen order."""

        seen: list[str] = []
        for cell in self.cells:
            if cell["model"] not in seen:
                seen.append(cell["model"])
        return seen

    def successful_cells(self) -> list[dict[str, Any]]:
        """Cells whose model completed."""

        return [cell for cell in self.cells if cell.get("status") == "ok"]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload."""

        return {
            "schema_version": "2.0",
            "kind": "stress",
            "workload": self.kind,
            "generated_at": self.generated_at,
            "metric_k": self.metric_k,
            "settings": self.settings,
            "models": self.models,
            "cells": self.cells,
        }

    def write(self, output_dir: Path, *, stem: str | None = None) -> dict[str, Path]:
        """Write ``<stem>.json`` and ``<stem>.md``.

        Args:
            output_dir: Destination directory.
            stem: File stem; defaults to the workload kind.

        Returns:
            Mapping of artifact name to path.

        Raises:
            ReportError: If the directory cannot be created.
        """

        directory = Path(output_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ReportError(f"Could not create output directory {directory}: {exc}") from exc
        name = stem or self.kind
        json_path = directory / f"{name}.json"
        markdown_path = directory / f"{name}.md"
        write_json(json_path, self.to_dict())
        atomic_write_text(markdown_path, self.markdown())
        return {"json": json_path, "markdown": markdown_path}

    def markdown(self) -> str:
        """Render the workload-specific Markdown report."""

        from embedbench.reporter import format_metric, format_usd, markdown_cell

        k = self.metric_k
        headers = ["Model", "Configuration", "Docs", "Hit", "MRR", "nDCG", "p95 ms", "Cost"]
        lines = [
            f"# Stress workload: {self.kind}",
            "",
            f"Generated {self.generated_at} · headline cut-off **{k}**",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---", "---", "---:"] + ["---:"] * 5) + " |",
        ]
        for cell in self.successful_cells():
            metrics = cell.get("metrics", {})
            lines.append(
                "| {model} | {config} | {docs} | {hit} | {mrr} | {ndcg} | {p95} | {cost} |".format(
                    model=markdown_cell(cell["model"], 36),
                    config=markdown_cell(_config_label(cell.get("config", {})), 40),
                    docs=cell.get("documents", "n/a"),
                    hit=format_metric(metrics.get("hit_rate")),
                    mrr=format_metric(metrics.get("mrr")),
                    ndcg=format_metric(metrics.get("ndcg")),
                    p95=format_metric(cell.get("p95_ms"), 1),
                    cost=format_usd(cell.get("usd")),
                )
            )
        failures = [cell for cell in self.cells if cell.get("status") != "ok"]
        if failures:
            lines.extend(
                ["", "## Failures", "", "| Model | Configuration | Error |", "| --- | --- | --- |"]
            )
            for cell in failures:
                lines.append(
                    f"| {markdown_cell(cell['model'], 36)} | "
                    f"{markdown_cell(_config_label(cell.get('config', {})), 40)} | "
                    f"{markdown_cell(cell.get('error') or 'unknown')} |"
                )

        if self.kind == "needle":
            lines.extend(self._needle_depth_table())
        elif self.kind == "query-noise":
            lines.extend(self._noise_delta_table())

        lines.extend(
            [
                "",
                "## Reading the numbers",
                "",
                "- Cells share the same query set for a given configuration, so "
                "differences are attributable to context length, chunk size or "
                "query noise rather than to a different index.",
                "- Latency covers query embedding plus exact cosine search, measured per query.",
                "",
            ]
        )
        return "\n".join(lines)

    def _needle_depth_table(self) -> list[str]:
        """Depth breakdown for the needle workload, grouped per depth."""

        from embedbench.reporter import format_metric, markdown_cell

        k = self.metric_k
        rows: list[list[str]] = []
        for cell in self.successful_cells():
            queries = cell.get("queries", [])
            if not queries:
                continue
            buckets: dict[float, list[dict[str, Any]]] = {}
            for query in queries:
                depth = query.get("metadata", {}).get("depth")
                if depth is None:
                    continue
                buckets.setdefault(float(depth), []).append(query)
            context_tokens = cell.get("config", {}).get("context_tokens", "n/a")
            for depth in sorted(buckets):
                bucket = buckets[depth]
                hit = sum(
                    1 for query in bucket if query.get("metrics", {}).get("hit_rate", 0.0) > 0
                )
                ndcg = sum(query.get("metrics", {}).get("ndcg", 0.0) for query in bucket) / len(
                    bucket
                )
                rows.append(
                    [
                        markdown_cell(cell["model"], 36),
                        str(context_tokens),
                        f"{depth:.2f}",
                        f"{hit}/{len(bucket)}",
                        format_metric(ndcg),
                    ]
                )
        if not rows:
            return []
        lines = [
            "",
            f"## Depth breakdown @ {k}",
            "",
            "| Model | Context tokens | Depth | Queries hit | Mean nDCG |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return lines

    def _noise_delta_table(self) -> list[str]:
        """Quality deltas relative to the clean baseline."""

        from embedbench.reporter import format_metric, markdown_cell

        baseline = {
            cell["model"]: cell.get("metrics", {}).get("ndcg", 0.0)
            for cell in self.successful_cells()
            if cell.get("config", {}).get("variant") == "clean"
        }
        rows: list[list[str]] = []
        for cell in self.successful_cells():
            variant = cell.get("config", {}).get("variant")
            if variant in (None, "clean"):
                continue
            reference = baseline.get(cell["model"])
            ndcg = cell.get("metrics", {}).get("ndcg", 0.0)
            delta = f"{ndcg - reference:+.3f}" if reference is not None else "n/a"
            retention = (
                f"{100.0 * ndcg / reference:.1f}%" if reference not in (None, 0.0) else "n/a"
            )
            rows.append(
                [
                    markdown_cell(cell["model"], 36),
                    str(variant),
                    format_metric(ndcg),
                    delta,
                    retention,
                ]
            )
        if not rows:
            return []
        lines = [
            "",
            "## Robustness vs clean baseline",
            "",
            "| Model | Variant | nDCG | Δ nDCG | Quality retained |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return lines


def _config_label(config: Mapping[str, Any]) -> str:
    parts = []
    for key in ("context_tokens", "depth", "chunk_tokens", "variant"):
        if key in config:
            value = config[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
    return ", ".join(parts) or "default"


class StressRunner:
    """Runs the three stress workloads for a fixed set of models.

    Args:
        models: Model specs or instances accepted by
            :class:`~embedbench.evaluator.Evaluator`.
        k_values: Cut-offs to evaluate for every configuration.
        metric_k: Cut-off used for headline columns and deltas; defaults to the
            largest ``k_values`` entry.
        batch_size: Overrides each model's batch size.
        index_backend: ``auto``, ``faiss`` or ``numpy``.
        show_progress: Print per-model lines while running.
        console: Rich console for progress output.
        seed: Seed for data generation and model runs.
    """

    def __init__(
        self,
        models: Sequence[str | BaseEmbeddingModel],
        *,
        k_values: Sequence[int] = (1, 5, 10),
        metric_k: int | None = None,
        batch_size: int | None = None,
        index_backend: str = "auto",
        show_progress: bool = False,
        console: Console | None = None,
        seed: int = 42,
        plugins: Sequence[str] | None = None,
    ) -> None:
        if not models:
            raise ValueError("StressRunner needs at least one model")
        self.evaluator = Evaluator(
            models,
            k_values=k_values,
            batch_size=batch_size,
            index_backend=index_backend,  # type: ignore[arg-type]
            show_progress=show_progress,
            console=console,
            seed=seed,
            plugins=plugins,
        )
        self.metric_k = metric_k or max(self.evaluator.k_values)

    # -- public workloads -------------------------------------------------

    def needle_in_haystack(
        self,
        *,
        context_sizes: Sequence[int] = DEFAULT_CONTEXT_SIZES,
        depths: Sequence[float] = DEFAULT_DEPTHS,
        chunk_size: int = 800,
        seed: int = 42,
    ) -> StressReport:
        """Evaluate retrieval of facts hidden at increasing context lengths.

        Args:
            context_sizes: Haystack sizes in estimated tokens.
            depths: Needle insertion depths as fractions of the haystack.
            chunk_size: Chunk size in characters.
            seed: Seed for filler and needle generation.

        Returns:
            A report with one cell per context size and model, carrying
            per-query records tagged with their depth.

        Raises:
            DatasetError: If generation fails for a configuration.
        """

        datasets = build_needle_datasets(
            context_sizes=context_sizes,
            depths=depths,
            chunk_size=chunk_size,
            seed=seed,
        )
        cells: list[dict[str, Any]] = []
        for context_size, dataset in datasets.items():
            result = self.evaluator.evaluate(dataset)
            logger.info(
                "Needle workload %d tokens: %d documents, %d queries",
                context_size,
                dataset.num_documents,
                dataset.num_queries,
            )
            cells.extend(
                self._cells(
                    result,
                    {"context_tokens": context_size},
                    include_queries=True,
                )
            )
        return StressReport(
            kind="needle",
            metric_k=self.metric_k,
            settings={
                "context_sizes": [int(value) for value in context_sizes],
                "depths": [float(value) for value in depths],
                "chunk_size": chunk_size,
                "seed": seed,
                "k_values": list(self.evaluator.k_values),
            },
            cells=cells,
        )

    def chunk_size_sensitivity(
        self,
        documents: Sequence[DatasetDocument],
        queries: Sequence[RetrievalQuery],
        *,
        chunk_tokens: Sequence[int] = DEFAULT_CHUNK_TOKENS,
        dataset_name: str = "chunk-size-sensitivity",
    ) -> StressReport:
        """Re-chunk a corpus at several sizes and compare retrieval quality.

        Args:
            documents: Source documents.
            queries: Queries whose relevant IDs name source documents.
            chunk_tokens: Chunk sizes in estimated tokens.
            dataset_name: Name used for generated datasets.

        Returns:
            One cell per chunk size and model.

        Raises:
            DatasetError: If chunking produces no usable configuration.
        """

        datasets = build_chunk_datasets(
            documents,
            queries,
            chunk_tokens=chunk_tokens,
            name=dataset_name,
        )
        cells: list[dict[str, Any]] = []
        for tokens, dataset in datasets.items():
            result = self.evaluator.evaluate(dataset)
            cells.extend(
                self._cells(
                    result,
                    {"chunk_tokens": tokens, "chunks": dataset.num_documents},
                    include_queries=False,
                )
            )
        return StressReport(
            kind="chunk-size",
            metric_k=self.metric_k,
            settings={
                "chunk_tokens": [int(value) for value in chunk_tokens],
                "source_documents": len(documents),
                "queries": len(queries),
                "k_values": list(self.evaluator.k_values),
            },
            cells=cells,
        )

    def query_robustness(
        self,
        dataset: RetrievalDataset,
        *,
        kinds: Sequence[str] = DEFAULT_NOISE_KINDS,
        seed: int = 42,
    ) -> StressReport:
        """Measure quality retention under degraded queries.

        Args:
            dataset: Clean source dataset.
            kinds: Degradations from
                :data:`embedbench.datasets.DEFAULT_NOISE_KINDS`.
            seed: Seed for deterministic degradation.

        Returns:
            One cell per variant and model, including the clean baseline.

        Raises:
            DatasetError: If a variant cannot be generated.
        """

        variants = query_variants(dataset, kinds, seed=seed)
        cells: list[dict[str, Any]] = []
        for variant, variant_dataset in variants.items():
            result = self.evaluator.evaluate(variant_dataset)
            cells.extend(
                self._cells(
                    result,
                    {"variant": variant},
                    include_queries=False,
                )
            )
        return StressReport(
            kind="query-noise",
            metric_k=self.metric_k,
            settings={
                "variants": list(variants),
                "queries": dataset.num_queries,
                "seed": seed,
                "k_values": list(self.evaluator.k_values),
            },
            cells=cells,
        )

    # -- internals --------------------------------------------------------

    def _cells(
        self,
        result: BenchmarkResult,
        config: Mapping[str, Any],
        *,
        include_queries: bool,
    ) -> list[dict[str, Any]]:
        cells: list[dict[str, Any]] = []
        for model in result.models:
            cells.append(self._cell(model, config, result, include_queries=include_queries))
        return cells

    def _cell(
        self,
        model: ModelResult,
        config: Mapping[str, Any],
        result: BenchmarkResult,
        *,
        include_queries: bool,
    ) -> dict[str, Any]:
        cell: dict[str, Any] = {
            "model": model.spec,
            "config": dict(config),
            "dataset": result.dataset_name,
            "status": model.status,
            "error": model.error,
            "documents": result.dataset_stats.get("documents"),
            "queries_total": result.dataset_stats.get("queries"),
            "metrics_by_k": model.metrics_by_k,
            "metrics": model.metrics_at(self.metric_k),
            "p50_ms": model.latency.get("p50_ms"),
            "p95_ms": model.latency.get("p95_ms"),
            "qps": model.latency.get("qps"),
            "usd": model.cost.get("usd"),
            "tokens": model.tokens.get("total_tokens"),
        }
        if include_queries:
            cell["queries"] = [
                {
                    "query_id": row.get("query_id"),
                    "metadata": dict(row.get("metadata") or {}),
                    "metrics": row.get("metrics_by_k", {}).get(str(self.metric_k), {}),
                    "latency_ms": row.get("latency_ms"),
                }
                for row in model.per_query
            ]
        return cell


__all__ = ["StressReport", "StressRunner"]
