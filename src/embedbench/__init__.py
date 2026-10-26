"""embedbench: benchmark text embedding models for retrieval quality and cost.

Two entry points:

Python API
    .. code-block:: python

        from embedbench import Evaluator, load_corpus_dataset, write_reports

        dataset = load_corpus_dataset("./docs", "evalset.jsonl")
        result = Evaluator(
            ["ollama:nomic-embed-text", "openai:text-embedding-3-small"],
            k_values=(1, 5, 10),
        ).evaluate(dataset)
        write_reports(result, "results/")

Command line
    ``embedbench run`` keeps the file-based pipeline; ``embedbench bench`` uses
    the dataset evaluator; ``embedbench stress`` runs needle-in-a-haystack,
    chunk-size and query-noise workloads.
"""

from __future__ import annotations

__version__ = "0.2.0"

from embedbench.datasets import (
    DatasetDocument,
    RetrievalDataset,
    RetrievalQuery,
    chunk_size_sensitivity,
    demo_dataset,
    load_corpus_dataset,
    load_documents_dataset,
    load_jsonl_dataset,
    load_queries,
    needle_in_haystack,
    query_variants,
)
from embedbench.errors import (
    BackendError,
    ConfigError,
    DatasetError,
    EmbedbenchError,
    EvalsetError,
    IndexBuildError,
    IngestionError,
    ReportError,
)
from embedbench.evaluator import (
    PRIORITIES,
    BenchmarkResult,
    Evaluator,
    ModelResult,
)
from embedbench.metrics import (
    DEFAULT_K_VALUES,
    METRIC_NAMES,
    BatchRetrievalMetrics,
    CostEstimate,
    LatencyStats,
    MemoryFootprint,
    RetrievalMetrics,
    estimate_cost,
    evaluate_rankings,
    hit_rate_at_k,
    latency_stats,
    mean_metrics_by_k,
    memory_footprint,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    throughput,
)
from embedbench.models import (
    BaseEmbeddingModel,
    available_providers,
    create_model,
    pricing_catalog,
    pricing_for,
    register_model,
)
from embedbench.reporter import (
    render_console,
    render_markdown,
    write_reports,
)
from embedbench.stress import StressReport, StressRunner

__all__ = [
    "DEFAULT_K_VALUES",
    "METRIC_NAMES",
    "PRIORITIES",
    "BackendError",
    "BaseEmbeddingModel",
    "BatchRetrievalMetrics",
    "BenchmarkResult",
    "ConfigError",
    "CostEstimate",
    "DatasetDocument",
    "DatasetError",
    "EmbedbenchError",
    "EvalsetError",
    "Evaluator",
    "IndexBuildError",
    "IngestionError",
    "LatencyStats",
    "MemoryFootprint",
    "ModelResult",
    "ReportError",
    "RetrievalDataset",
    "RetrievalMetrics",
    "RetrievalQuery",
    "StressReport",
    "StressRunner",
    "__version__",
    "available_providers",
    "chunk_size_sensitivity",
    "create_model",
    "demo_dataset",
    "estimate_cost",
    "evaluate_rankings",
    "hit_rate_at_k",
    "latency_stats",
    "load_corpus_dataset",
    "load_documents_dataset",
    "load_jsonl_dataset",
    "load_queries",
    "mean_metrics_by_k",
    "memory_footprint",
    "mrr_at_k",
    "ndcg_at_k",
    "needle_in_haystack",
    "precision_at_k",
    "pricing_catalog",
    "pricing_for",
    "query_variants",
    "recall_at_k",
    "register_model",
    "render_console",
    "render_markdown",
    "throughput",
    "write_reports",
]
