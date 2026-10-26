"""Model-agnostic retrieval evaluation.

:class:`Evaluator` is the library entry point behind ``from embedbench import
Evaluator``.  It embeds a dataset with one or more models, builds an exact
cosine index, answers every query, and records quality, latency, throughput,
memory and cost in a :class:`BenchmarkResult`.

The evaluator never raises for a model-level failure: a provider that is down
or misconfigured is recorded with ``status="error"`` so the remaining models
still produce a comparable report.  Invalid datasets do raise, because there is
nothing to measure.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from embedbench import __version__
from embedbench.datasets import RetrievalDataset
from embedbench.errors import BackendError, DatasetError
from embedbench.index import IndexBackend, build_index
from embedbench.logging_utils import get_logger
from embedbench.metrics import (
    DEFAULT_K_VALUES,
    evaluate_rankings_at_k,
    latency_stats,
    mean_metrics_by_k,
    memory_footprint,
    throughput,
    warm_latency_stats,
)
from embedbench.models import BaseEmbeddingModel, create_model, load_model_plugins
from embedbench.resources import ResourceSampler, environment_info
from embedbench.utils import set_global_seed, utc_now_iso

logger = get_logger(__name__)


def _elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since a `time.perf_counter()` reading."""

    return round((time.perf_counter() - start) * 1000.0, 3)


PRIORITIES: tuple[str, ...] = ("accuracy", "speed", "balanced", "cost")

_PRIORITY_WEIGHTS: dict[str, tuple[float, float, float]] = {
    # (quality, speed, cost)
    "accuracy": (0.80, 0.15, 0.05),
    "speed": (0.20, 0.75, 0.05),
    "balanced": (0.55, 0.30, 0.15),
    "cost": (0.30, 0.10, 0.60),
}


@dataclass(slots=True)
class ModelResult:
    """Everything measured for one model on one dataset.

    Attributes:
        spec: ``provider:model`` string.
        provider: Provider name.
        model: Model name.
        status: ``"ok"`` or ``"error"``.
        error: Provider error message when ``status`` is ``"error"``.
        metrics_by_k: Mean retrieval metrics keyed by cut-off string.
        latency: Per-query latency statistics.
        latency_warm: Latency statistics without the first query.
        throughput: Documents and queries per second.
        tokens: Token usage with estimation metadata.
        cost: Cost estimate in US dollars.
        memory: Vector footprint, index size and sampled RSS/VRAM.
        timings: Wall-clock breakdown in milliseconds.
        index: Index backend, dimension and vector count.
        runtime: Provider-specific runtime metadata.
        per_query: Per-query records, including retrieved documents.
    """

    spec: str
    provider: str
    model: str
    status: str = "ok"
    error: str | None = None
    metrics_by_k: dict[str, dict[str, float]] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    latency_warm: dict[str, Any] | None = None
    throughput: dict[str, float | None] = field(default_factory=dict)
    tokens: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float | None] = field(default_factory=dict)
    index: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the model produced measurements."""

        return self.status == "ok"

    def metrics_at(self, k: int) -> dict[str, float]:
        """Return mean metrics for one cut-off.

        Args:
            k: Cut-off that was evaluated.

        Returns:
            Metric mapping, empty when that cut-off was not evaluated.
        """

        return dict(self.metrics_by_k.get(str(k), {}))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""

        return {
            "spec": self.spec,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "error": self.error,
            "metrics_by_k": self.metrics_by_k,
            "latency": self.latency,
            "latency_warm": self.latency_warm,
            "throughput": self.throughput,
            "tokens": self.tokens,
            "cost": self.cost,
            "memory": self.memory,
            "timings": self.timings,
            "index": self.index,
            "runtime": self.runtime,
            "per_query": self.per_query,
        }


@dataclass(slots=True)
class BenchmarkResult:
    """A complete evaluation: dataset summary plus one entry per model.

    Attributes:
        dataset_name: Name of the evaluated dataset.
        dataset_stats: Document/query counts from
            :meth:`embedbench.datasets.RetrievalDataset.stats`.
        k_values: Cut-offs that were evaluated.
        models: Per-model results in request order.
        settings: Evaluator settings that affect comparability.
        dataset_metadata: Generator metadata copied for reproducibility.
        generated_at: ISO-8601 UTC timestamp.
        environment: Interpreter, OS and dependency versions.
    """

    dataset_name: str
    dataset_stats: dict[str, Any]
    k_values: list[int]
    models: list[ModelResult]
    settings: dict[str, Any] = field(default_factory=dict)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=utc_now_iso)
    environment: dict[str, Any] = field(default_factory=environment_info)

    @property
    def reference_k(self) -> int:
        """Largest evaluated cut-off, used for headline numbers."""

        return max(self.k_values)

    @property
    def successful(self) -> list[ModelResult]:
        """Results for models that completed."""

        return [model for model in self.models if model.ok]

    @property
    def failed(self) -> list[ModelResult]:
        """Results for models that failed."""

        return [model for model in self.models if not model.ok]

    def best(self, k: int | None = None) -> ModelResult | None:
        """Return the model with the highest nDCG at ``k``.

        Args:
            k: Cut-off; defaults to the reference cut-off.

        Returns:
            The best successful model, or ``None`` when all failed.
        """

        cut_off = k or self.reference_k
        candidates = [model for model in self.successful if model.metrics_at(cut_off)]
        if not candidates:
            return None
        return max(candidates, key=lambda model: model.metrics_at(cut_off).get("ndcg", 0.0))

    def recommend(
        self,
        priority: str = "balanced",
        *,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Rank successful models by a weighted quality/speed/cost score.

        Each component is min-max normalised across the successful models, then
        combined with weights that depend on ``priority``.  Free local models
        score best on cost; unknown prices score neutrally.

        Args:
            priority: One of ``accuracy``, ``speed``, ``balanced`` or ``cost``.
            k: Cut-off used for the quality term.

        Returns:
            A mapping with the normalised component scores and the winner.

        Raises:
            ValueError: If ``priority`` is unknown.
        """

        if priority not in _PRIORITY_WEIGHTS:
            raise ValueError(
                f"Unknown priority {priority!r}. Choose one of: {', '.join(PRIORITIES)}."
            )
        cut_off = k or self.reference_k
        weights = _PRIORITY_WEIGHTS[priority]
        models = self.successful
        if not models:
            return {"priority": priority, "reference_k": cut_off, "ranking": [], "winner": None}

        quality = {model.spec: model.metrics_at(cut_off).get("ndcg", 0.0) for model in models}
        speed = {
            model.spec: float(model.latency.get("qps") or 0.0) for model in models if model.latency
        }
        costs = {model.spec: model.cost.get("usd") for model in models}
        paid = [usd for usd in costs.values() if usd is not None and usd > 0.0]

        def normalized(value: float, values: Sequence[float]) -> float:
            if not values:
                return 1.0
            low, high = min(values), max(values)
            if high == low:
                return 1.0
            return (value - low) / (high - low)

        quality_score = {
            spec: normalized(value, list(quality.values())) for spec, value in quality.items()
        }
        speed_score = {
            model.spec: normalized(speed[model.spec], list(speed.values()))
            if model.spec in speed
            else 0.0
            for model in models
        }
        cost_score: dict[str, float] = {}
        for model in models:
            usd = costs.get(model.spec)
            if usd is None:
                cost_score[model.spec] = 0.5
            elif usd <= 0.0:
                cost_score[model.spec] = 1.0
            else:
                cost_score[model.spec] = normalized(-usd, [-value for value in paid])

        ranking = []
        for model in models:
            total = (
                weights[0] * quality_score[model.spec]
                + weights[1] * speed_score[model.spec]
                + weights[2] * cost_score[model.spec]
            )
            ranking.append(
                {
                    "spec": model.spec,
                    "score": round(total, 4),
                    "quality": round(quality_score[model.spec], 4),
                    "speed": round(speed_score[model.spec], 4),
                    "cost": round(cost_score[model.spec], 4),
                    "ndcg": round(quality.get(model.spec, 0.0), 4),
                    "p95_ms": model.latency.get("p95_ms"),
                    "usd": costs.get(model.spec),
                    "free": costs.get(model.spec) is not None and costs[model.spec] <= 0.0,
                }
            )
        ranking.sort(key=lambda row: row["score"], reverse=True)
        return {
            "priority": priority,
            "reference_k": cut_off,
            "weights": {"quality": weights[0], "speed": weights[1], "cost": weights[2]},
            "ranking": ranking,
            "winner": ranking[0]["spec"] if ranking else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report payload."""

        best = self.best()
        return {
            "schema_version": "2.0",
            "kind": "retrieval",
            "generated_at": self.generated_at,
            "tool": {"name": "embedbench", "version": __version__},
            "dataset": {
                "name": self.dataset_name,
                "stats": self.dataset_stats,
                "metadata": self.dataset_metadata,
            },
            "settings": self.settings,
            "k_values": list(self.k_values),
            "reference_k": self.reference_k,
            "environment": self.environment,
            "models": [model.to_dict() for model in self.models],
            "summary": {
                "models_requested": len(self.models),
                "successful": len(self.successful),
                "failed": len(self.failed),
                "best_spec": best.spec if best else None,
            },
        }

    def markdown(self) -> str:
        """Render a compact Markdown summary.

        Returns:
            A Markdown document with the headline comparison table.
        """

        from embedbench.reporter import render_markdown

        return render_markdown(self)


class Evaluator:
    """Runs retrieval benchmarks across embedding models.

    Args:
        models: Model specs (``"ollama:nomic-embed-text"``) or ready-to-use
            :class:`~embedbench.models.BaseEmbeddingModel` instances.
        k_values: Cut-offs to evaluate, for example ``(1, 5, 10)``.
        batch_size: Overrides each model's batch size for this run.
        index_backend: ``auto``, ``faiss`` or ``numpy``.
        show_progress: Print one line per model as it runs.
        console: Rich console for progress output; a plain one by default.
        seed: Seed applied before each model runs.
        plugins: Dotted modules or ``.py`` files registering custom providers.

    Example:
        >>> from embedbench import Evaluator, demo_dataset
        >>> evaluator = Evaluator(["ollama:nomic-embed-text"], k_values=(1, 5))
        >>> result = evaluator.evaluate(demo_dataset())  # doctest: +SKIP
        >>> result.best().spec  # doctest: +SKIP
        'ollama:nomic-embed-text'
    """

    def __init__(
        self,
        models: Sequence[str | BaseEmbeddingModel],
        *,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        batch_size: int | None = None,
        index_backend: IndexBackend = "auto",
        show_progress: bool = True,
        console: Console | None = None,
        seed: int = 42,
        plugins: Sequence[str] | None = None,
    ) -> None:
        if not models:
            raise ValueError("Evaluator needs at least one model")
        self.models = list(models)
        self.k_values = sorted({int(k) for k in k_values})
        if not self.k_values or self.k_values[0] < 1:
            raise ValueError("k_values must contain positive cut-offs")
        self.batch_size = batch_size
        self.index_backend = index_backend
        self.show_progress = show_progress
        self.console = console or Console()
        self.seed = seed
        if plugins:
            load_model_plugins(list(plugins))

    # -- public API -------------------------------------------------------

    def evaluate(
        self,
        dataset: RetrievalDataset,
        *,
        k_values: Sequence[int] | None = None,
    ) -> BenchmarkResult:
        """Benchmark every configured model on ``dataset``.

        Args:
            dataset: Documents, queries and relevance judgements.
            k_values: Overrides the evaluator-level cut-offs for this run.

        Returns:
            The benchmark result, including failed models.

        Raises:
            DatasetError: If the dataset is empty or inconsistent.
        """

        dataset.validate()
        cut_offs = sorted({int(k) for k in (k_values or self.k_values)})
        if not cut_offs or cut_offs[0] < 1:
            raise ValueError("k_values must contain positive cut-offs")
        top_k = cut_offs[-1]

        results: list[ModelResult] = []
        for entry in self.models:
            set_global_seed(self.seed)
            model, owned = self._materialize(entry)
            try:
                result = self._evaluate_model(model, dataset, cut_offs, top_k)
            finally:
                if owned:
                    try:
                        model.close()
                    except Exception as exc:  # pragma: no cover - cleanup must not mask results
                        logger.warning("Error closing %s: %s", model.name, exc)
            results.append(result)
            self._report_progress(result, top_k)

        return BenchmarkResult(
            dataset_name=dataset.name,
            dataset_stats=dataset.stats(),
            k_values=cut_offs,
            models=results,
            settings={
                "batch_size": self.batch_size,
                "index_backend": self.index_backend,
                "seed": self.seed,
                "top_k": top_k,
                "relevance": "binary, document-level (group-de-duplicated)",
            },
            dataset_metadata=dict(dataset.metadata),
        )

    def evaluate_many(
        self,
        datasets: Mapping[str, RetrievalDataset] | Sequence[RetrievalDataset],
    ) -> list[BenchmarkResult]:
        """Evaluate several datasets in order.

        Args:
            datasets: A mapping of name to dataset, or a sequence of datasets.

        Returns:
            One result per dataset.
        """

        values = list(datasets.values()) if isinstance(datasets, Mapping) else list(datasets)
        return [self.evaluate(dataset) for dataset in values]

    # -- internals --------------------------------------------------------

    def _materialize(self, entry: str | BaseEmbeddingModel) -> tuple[BaseEmbeddingModel, bool]:
        """Return the model to run and whether the evaluator owns it."""

        if isinstance(entry, BaseEmbeddingModel):
            return entry, False
        return create_model(entry), True

    def _report_progress(self, result: ModelResult, reference_k: int) -> None:
        if not self.show_progress:
            return
        if not result.ok:
            self.console.print(f"[red]  {result.spec} failed:[/red] {result.error}")
            return
        metrics = result.metrics_at(reference_k)
        p95 = result.latency.get("p95_ms")
        p95_text = f"{p95:.1f} ms" if isinstance(p95, (int, float)) else "n/a"
        self.console.print(
            f"  [cyan]{result.spec}[/cyan] "
            f"nDCG@{reference_k}={metrics.get('ndcg', 0.0):.3f} "
            f"MRR={metrics.get('mrr', 0.0):.3f} p95={p95_text}"
        )

    def _evaluate_model(
        self,
        model: BaseEmbeddingModel,
        dataset: RetrievalDataset,
        cut_offs: list[int],
        top_k: int,
    ) -> ModelResult:
        spec = f"{model.provider_name}:{model.model_name}"
        total_start = time.perf_counter()
        sampler = ResourceSampler()
        sampler.start()
        try:
            if self.batch_size is not None:
                model.batch_size = int(self.batch_size)
            load_start = time.perf_counter()
            model.load()
            load_ms = _elapsed_ms(load_start)

            token_start = time.perf_counter()
            tokens = self._token_payload(model, dataset)
            token_ms = _elapsed_ms(token_start)

            texts = [document.text for document in dataset.documents]
            embed_start = time.perf_counter()
            vectors = model.embed_documents_matrix(texts)
            embed_ms = _elapsed_ms(embed_start)
            if vectors.shape[0] != len(texts):
                raise BackendError(
                    f"{spec} returned {vectors.shape[0]} vectors for {len(texts)} documents"
                )

            index_start = time.perf_counter()
            index = build_index(vectors, self.index_backend)
            index_ms = _elapsed_ms(index_start)
            try:
                index_disk_bytes: int | None = index.disk_size_bytes()
            except Exception as exc:  # pragma: no cover - size probing is best effort
                logger.warning("Could not measure index size for %s: %s", spec, exc)
                index_disk_bytes = None

            per_query: list[dict[str, Any]] = []
            latencies: list[float] = []
            group_rankings: list[list[str]] = []
            relevant_lists: list[list[str]] = []
            query_start = time.perf_counter()
            for query in dataset.queries:
                start = time.perf_counter()
                query_vector = model.embed_queries_matrix([query.query])[0]
                scores, positions = index.search(query_vector, top_k)
                latency = _elapsed_ms(start)
                latencies.append(latency)

                retrieved: list[dict[str, Any]] = []
                raw_doc_ids: list[str] = []
                for rank, position in enumerate(positions.tolist(), start=1):
                    document = dataset.documents[int(position)]
                    raw_doc_ids.append(document.doc_id)
                    retrieved.append(
                        {
                            "rank": rank,
                            "doc_id": document.doc_id,
                            "group_id": document.group,
                            "score": round(float(scores[rank - 1]), 6),
                        }
                    )
                ranking = dataset.group_ranking(raw_doc_ids, top_k)
                group_rankings.append(ranking)
                relevant_lists.append(list(query.relevant_doc_ids))
                per_query.append(
                    {
                        "query_id": query.query_id,
                        "query": query.query,
                        "relevant_doc_ids": list(query.relevant_doc_ids),
                        "ranked_doc_ids": ranking,
                        "retrieved": retrieved,
                        "latency_ms": round(latency, 3),
                        "metadata": dict(query.metadata),
                    }
                )
            query_total_ms = _elapsed_ms(query_start)

            metrics_by_k = mean_metrics_by_k(group_rankings, relevant_lists, cut_offs)
            metric_rows = evaluate_rankings_at_k(group_rankings, relevant_lists, cut_offs)
            rows_by_k = {str(cut_off): metrics.rows() for cut_off, metrics in metric_rows.items()}
            for position_index, row in enumerate(per_query):
                row["metrics_by_k"] = {
                    cut_off: values[position_index] for cut_off, values in rows_by_k.items()
                }

            resources = sampler.stop()
            runtime = self._runtime_payload(model, resources)
            footprint = memory_footprint(vectors.shape[0], int(vectors.shape[1]))
            memory = {
                **footprint.to_dict(),
                "index_disk_bytes": index_disk_bytes,
                "index_disk_mb": round(index_disk_bytes / (1024 * 1024), 3)
                if index_disk_bytes is not None
                else None,
                "peak_rss_mb": resources.get("ram_peak_mb"),
                "ram_delta_mb": resources.get("ram_delta_mb"),
                "peak_vram_mb": runtime.get("peak_vram_mb"),
            }
            latency_payload = latency_stats(latencies).to_dict()
            warm = warm_latency_stats(latencies)
            cost = model.cost_for_tokens(int(tokens["total_tokens"])).to_dict()

            return ModelResult(
                spec=spec,
                provider=model.provider_name,
                model=model.model_name,
                status="ok",
                error=None,
                metrics_by_k={str(k): v for k, v in metrics_by_k.items()},
                latency=latency_payload,
                latency_warm=warm.to_dict() if warm else None,
                throughput={
                    "documents_per_second": throughput(len(texts), embed_ms / 1000.0),
                    "queries_per_second": latency_payload.get("qps"),
                },
                tokens=tokens,
                cost=cost,
                memory=memory,
                timings={
                    "load_ms": load_ms,
                    "token_counting_ms": token_ms,
                    "embed_documents_ms": embed_ms,
                    "index_build_ms": index_ms,
                    "queries_ms": query_total_ms,
                    "total_ms": round((time.perf_counter() - total_start) * 1000.0, 3),
                },
                index={
                    "backend": index.name,
                    "dimension": int(index.dimension),
                    "vectors": int(index.ntotal),
                    "disk_size_bytes": index_disk_bytes,
                },
                runtime=runtime,
                per_query=per_query,
            )
        except (BackendError, DatasetError) as exc:
            logger.error("Model %s failed: %s", spec, exc)
            return ModelResult(
                spec=spec,
                provider=model.provider_name,
                model=model.model_name,
                status="error",
                error=str(exc),
                timings={"total_ms": round((time.perf_counter() - total_start) * 1000.0, 3)},
            )
        except Exception as exc:  # unexpected provider bug: record and continue
            logger.exception("Unexpected failure for %s", spec)
            return ModelResult(
                spec=spec,
                provider=model.provider_name,
                model=model.model_name,
                status="error",
                error=f"Unexpected error: {type(exc).__name__}: {exc}",
                timings={"total_ms": round((time.perf_counter() - total_start) * 1000.0, 3)},
            )
        finally:
            sampler.stop()

    def _token_payload(
        self,
        model: BaseEmbeddingModel,
        dataset: RetrievalDataset,
    ) -> dict[str, Any]:
        document_texts = model.prepare_documents([document.text for document in dataset.documents])
        query_texts = model.prepare_queries([query.query for query in dataset.queries])
        document_tokens = model.count_tokens(document_texts)
        query_tokens = model.count_tokens(query_texts)
        total = document_tokens.tokens + query_tokens.tokens
        estimate = model.cost_for_tokens(total)
        return {
            "index_tokens": document_tokens.tokens,
            "query_tokens": query_tokens.tokens,
            "total_tokens": total,
            "estimated": bool(document_tokens.estimated or query_tokens.estimated),
            "method": (
                document_tokens.method
                if document_tokens.method == query_tokens.method
                else f"docs: {document_tokens.method}; queries: {query_tokens.method}"
            ),
            "usd": estimate.usd,
            "usd_per_1m_tokens": estimate.usd_per_1m_tokens,
            "price_known": estimate.price_known,
        }

    def _runtime_payload(
        self,
        model: BaseEmbeddingModel,
        resources: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            runtime = model.runtime_metadata()
        except Exception as exc:  # metadata must never invalidate a measurement
            logger.warning("Runtime metadata failed for %s: %s", model.name, exc)
            runtime = {}
        if runtime.get("gpu_peak_allocated_mb") is not None:
            runtime["peak_vram_mb"] = runtime["gpu_peak_allocated_mb"]
            runtime["vram_measurement"] = "torch_cuda_peak_allocated"
        elif runtime.get("model_resident_vram_bytes") is not None:
            runtime["peak_vram_mb"] = round(
                float(runtime["model_resident_vram_bytes"]) / (1024 * 1024), 2
            )
            runtime["vram_measurement"] = "provider_resident_memory"
        else:
            runtime.setdefault("peak_vram_mb", None)
        runtime["ram_peak_mb"] = resources.get("ram_peak_mb")
        return runtime


__all__ = [
    "PRIORITIES",
    "BenchmarkResult",
    "Evaluator",
    "ModelResult",
]
