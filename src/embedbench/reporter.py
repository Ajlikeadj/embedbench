"""Reporting for :class:`embedbench.evaluator.BenchmarkResult`.

Four output surfaces, all optional at the call site:

* a Rich terminal table for live feedback,
* a Markdown summary (``results.md``),
* flat CSV exports (``summary.csv`` and ``queries.csv``),
* comparative plots (``latency-vs-quality.png``, ``quality-bars.png``).

``matplotlib`` is optional; plotting degrades to a clear
:class:`~embedbench.errors.ReportError` instead of an import failure.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from embedbench import __version__
from embedbench.errors import ReportError
from embedbench.evaluator import BenchmarkResult, ModelResult
from embedbench.logging_utils import get_logger
from embedbench.metrics import METRIC_NAMES
from embedbench.utils import atomic_write_text, write_json as _write_json_file

logger = get_logger(__name__)

_REQUIRED_EXTRA = 'pip install "embedbench[plots]"'


def _short(spec: str, limit: int = 32) -> str:
    """Shorten a model spec for tables while keeping the model name readable."""

    if len(spec) <= limit:
        return spec
    return spec[: limit - 3] + "..."


def format_metric(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return f"{escape(str(value))}{suffix}"


def format_usd(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if value == 0:
            return "$0 (free)"
        if value < 0.01:
            return f"${value:.6f}"
        return f"${value:.4f}"
    return escape(str(value))


def _table_row(result: ModelResult, k: int) -> list[str]:
    metrics = result.metrics_at(k)
    latency = result.latency
    tokens = result.tokens
    memory = result.memory
    return [
        _short(result.spec),
        str(result.index.get("dimension", "n/a")),
        format_metric(metrics.get("hit_rate")),
        format_metric(metrics.get("mrr")),
        format_metric(metrics.get("ndcg")),
        format_metric(metrics.get("precision")),
        format_metric(metrics.get("recall")),
        format_metric(latency.get("p50_ms"), 1),
        format_metric(latency.get("p95_ms"), 1),
        format_metric(latency.get("qps"), 2),
        format_metric(tokens.get("total_tokens"), 0),
        format_usd(result.cost.get("usd")),
        format_metric(memory.get("peak_rss_mb"), 1),
    ]


_HEADERS = [
    "Model",
    "Dim",
    "Hit",
    "MRR",
    "nDCG",
    "P",
    "R",
    "p50 ms",
    "p95 ms",
    "QPS",
    "Tokens",
    "Cost",
    "Peak MB",
]


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def render_console(
    result: BenchmarkResult,
    console: Console | None = None,
    *,
    priority: str = "balanced",
) -> None:
    """Print a side-by-side comparison table plus the recommendation.

    Args:
        result: Benchmark result to render.
        console: Rich console; a fresh one is used by default.
        priority: Recommendation weighting passed to
            :meth:`BenchmarkResult.recommend`.
    """

    console = console or Console()
    k = result.reference_k
    stats = result.dataset_stats
    console.print(
        f"[bold]Dataset[/bold] {escape(result.dataset_name)}: "
        f"{stats.get('documents', 0)} documents, "
        f"{stats.get('queries', 0)} queries, "
        f"~{stats.get('estimated_tokens', 0):,} tokens"
    )

    table = Table(
        title=f"Retrieval comparison @ k={k}",
        header_style="bold cyan",
        box=None,
    )
    for header in _HEADERS:
        table.add_column(header, justify="right" if header not in {"Model"} else "left")
    for model in result.successful:
        table.add_row(*_table_row(model, k))
    console.print(table)

    recommendation = result.recommend(priority, k=k)
    winner = recommendation.get("winner")
    if winner:
        console.print(f"\n[bold green]Recommended ({priority})[/bold green]: {escape(str(winner))}")
        rows = recommendation.get("ranking", [])
        for position, row in enumerate(rows, start=1):
            markers = "  (recommended)" if row["spec"] == winner else ""
            console.print(
                f"  {position}. {escape(_short(row['spec'], 44))} "
                f"score={row['score']:.3f} "
                f"(quality={row['quality']:.2f}, speed={row['speed']:.2f}, "
                f"cost={row['cost']:.2f}){markers}"
            )
    if result.failed:
        console.print("\n[yellow]Failed models:[/yellow]")
        for model in result.failed:
            console.print(f"  {escape(model.spec)}: {escape(str(model.error))}")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def markdown_cell(value: Any, limit: int = 60) -> str:
    text = str(value if value is not None else "n/a")
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_markdown(result: BenchmarkResult, *, priority: str = "balanced") -> str:
    """Render a self-contained Markdown report.

    Args:
        result: Benchmark result.
        priority: Recommendation weighting.

    Returns:
        The Markdown document (no trailing file I/O).
    """

    k = result.reference_k
    stats = result.dataset_stats
    lines: list[str] = [
        f"# Embedding benchmark: {result.dataset_name}",
        "",
        f"Generated {result.generated_at} with embedbench {__version__}",
        "",
        "## Dataset",
        "",
        "| Documents | Groups | Queries | Chars | Est. tokens |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {stats.get('documents', 0)} | {stats.get('groups', 0)} | "
        f"{stats.get('queries', 0)} | {stats.get('chars', 0):,} | "
        f"{stats.get('estimated_tokens', 0):,} |",
        "",
        f"Cut-offs: {', '.join(str(value) for value in result.k_values)}; "
        f"headline cut-off: **{k}**.",
        "",
        f"## Retrieval quality @ {k}",
        "",
        "| Model | Dim | Hit rate | MRR | nDCG | Precision | Recall | p50 ms | p95 ms | p99 ms | QPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in result.successful:
        metrics = model.metrics_at(k)
        latency = model.latency
        lines.append(
            "| {model} | {dim} | {hit} | {mrr} | {ndcg} | {precision} | {recall} | "
            "{p50} | {p95} | {p99} | {qps} |".format(
                model=markdown_cell(model.spec, 44),
                dim=model.index.get("dimension", "n/a"),
                hit=format_metric(metrics.get("hit_rate")),
                mrr=format_metric(metrics.get("mrr")),
                ndcg=format_metric(metrics.get("ndcg")),
                precision=format_metric(metrics.get("precision")),
                recall=format_metric(metrics.get("recall")),
                p50=format_metric(latency.get("p50_ms"), 1),
                p95=format_metric(latency.get("p95_ms"), 1),
                p99=format_metric(latency.get("p99_ms"), 1),
                qps=format_metric(latency.get("qps"), 2),
            )
        )

    lines.extend(
        [
            "",
            "## Efficiency and cost",
            "",
            "| Model | Docs/s | Tokens | Price / 1M | Est. cost | Vector MB | Index MB | Peak RSS MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in result.successful:
        tokens = model.tokens
        memory = model.memory
        lines.append(
            "| {model} | {docs_s} | {tokens} | {price} | {cost} | {vector_mb} | "
            "{index_mb} | {rss} |".format(
                model=markdown_cell(model.spec, 44),
                docs_s=format_metric(model.throughput.get("documents_per_second"), 1),
                tokens=format_metric(tokens.get("total_tokens"), 0),
                price=format_usd(tokens.get("usd_per_1m_tokens")),
                cost=format_usd(model.cost.get("usd")),
                vector_mb=format_metric(memory.get("vector_mb"), 2),
                index_mb=format_metric(
                    memory.get("index_disk_mb")
                    if memory.get("index_disk_mb") is not None
                    else (memory.get("index_bytes") or 0) / (1024 * 1024),
                    2,
                ),
                rss=format_metric(memory.get("peak_rss_mb"), 1),
            )
        )

    recommendation = result.recommend(priority, k=k)
    lines.extend(["", f"## Recommendation ({priority})", ""])
    if recommendation.get("winner"):
        lines.append(f"**{recommendation['winner']}** ranks first with these weights:")
        weights = recommendation.get("weights", {})
        lines.append(
            f"quality {weights.get('quality')}, speed {weights.get('speed')}, "
            f"cost {weights.get('cost')}."
        )
        lines.extend(
            [
                "",
                "| Rank | Model | Score | Quality | Speed | Cost | nDCG | p95 ms | USD |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for position, row in enumerate(recommendation.get("ranking", []), start=1):
            lines.append(
                f"| {position} | {markdown_cell(row['spec'], 44)} | {row['score']:.3f} | "
                f"{row['quality']:.2f} | {row['speed']:.2f} | {row['cost']:.2f} | "
                f"{row['ndcg']:.3f} | {format_metric(row['p95_ms'], 1)} | {format_usd(row['usd'])} |"
            )
    else:
        lines.append("No successful models to rank.")

    if result.failed:
        lines.extend(["", "## Failures", "", "| Model | Error |", "| --- | --- |"])
        for model in result.failed:
            lines.append(
                f"| {markdown_cell(model.spec, 44)} | {markdown_cell(model.error or 'unknown')} |"
            )

    worst = _worst_queries(result, k, limit=10)
    if worst:
        lines.extend(
            [
                "",
                f"## Lowest-scoring queries @ {k}",
                "",
                "| Model | Query | nDCG | MRR | Latency ms |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for model, row in worst:
            metrics = row.get("metrics_by_k", {}).get(str(k), {})
            lines.append(
                f"| {markdown_cell(model.spec, 30)} | {markdown_cell(row['query'], 50)} | "
                f"{format_metric(metrics.get('ndcg'))} | {format_metric(metrics.get('mrr'))} | "
                f"{format_metric(row.get('latency_ms'), 1)} |"
            )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Relevance is binary and document-level; chunk rankings are "
            "de-duplicated to their source document before scoring.",
            "- Metrics follow standard IR definitions: nDCG uses a log2(rank+1) "
            "discount, precision uses k as its denominator.",
            "- Latency is measured per query end-to-end (embedding + index search) "
            "on a single thread; p50/p95/p99 use linear interpolation.",
            "- Cost estimates combine the provider's token count with its list "
            "price; local models report 0.",
            "",
        ]
    )
    return "\n".join(lines)


def _worst_queries(
    result: BenchmarkResult,
    k: int,
    *,
    limit: int,
) -> list[tuple[ModelResult, dict[str, Any]]]:
    rows: list[tuple[ModelResult, dict[str, Any]]] = []
    for model in result.successful:
        for row in model.per_query:
            rows.append((model, row))
    rows.sort(key=lambda item: item[1].get("metrics_by_k", {}).get(str(k), {}).get("ndcg", 0.0))
    return rows[:limit]


# ---------------------------------------------------------------------------
# CSV and JSON
# ---------------------------------------------------------------------------


def _csv_text(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def write_summary_csv(result: BenchmarkResult, path: Path, *, k: int | None = None) -> Path:
    """Write one row per model with quality, latency, memory and cost.

    Args:
        result: Benchmark result.
        path: Destination file.
        k: Cut-off for the metric columns; defaults to the reference cut-off.

    Returns:
        The path that was written.
    """

    cut_off = k or result.reference_k
    headers = [
        "spec",
        "provider",
        "model",
        "status",
        "k",
        *METRIC_NAMES,
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "qps",
        "documents_per_second",
        "index_tokens",
        "query_tokens",
        "total_tokens",
        "tokens_estimated",
        "usd_per_1m_tokens",
        "usd",
        "dimension",
        "vectors",
        "vector_mb",
        "index_disk_mb",
        "peak_rss_mb",
        "peak_vram_mb",
    ]
    rows: list[list[Any]] = []
    for model in result.models:
        metrics = model.metrics_at(cut_off)
        tokens = model.tokens
        memory = model.memory
        latency = model.latency
        rows.append(
            [
                model.spec,
                model.provider,
                model.model,
                model.status,
                cut_off,
                *(metrics.get(name, "") for name in METRIC_NAMES),
                latency.get("p50_ms", ""),
                latency.get("p90_ms", ""),
                latency.get("p95_ms", ""),
                latency.get("p99_ms", ""),
                latency.get("qps", ""),
                model.throughput.get("documents_per_second", ""),
                tokens.get("index_tokens", ""),
                tokens.get("query_tokens", ""),
                tokens.get("total_tokens", ""),
                tokens.get("estimated", ""),
                tokens.get("usd_per_1m_tokens", ""),
                model.cost.get("usd", ""),
                model.index.get("dimension", ""),
                model.index.get("vectors", ""),
                memory.get("vector_mb", ""),
                memory.get("index_disk_mb", ""),
                memory.get("peak_rss_mb", ""),
                memory.get("peak_vram_mb", ""),
            ]
        )
    target = Path(path)
    atomic_write_text(target, _csv_text(rows, headers))
    return target


def write_queries_csv(result: BenchmarkResult, path: Path, *, k: int | None = None) -> Path:
    """Write one row per query and model.

    Args:
        result: Benchmark result.
        path: Destination file.
        k: Cut-off used for the metric columns.

    Returns:
        The path that was written.
    """

    cut_off = k or result.reference_k
    headers = [
        "spec",
        "query_id",
        "query",
        "relevant_doc_ids",
        "ranked_doc_ids",
        "hit_rate",
        "mrr",
        "ndcg",
        "precision",
        "recall",
        "latency_ms",
    ]
    rows: list[list[Any]] = []
    for model in result.successful:
        for row in model.per_query:
            metrics = row.get("metrics_by_k", {}).get(str(cut_off), {})
            rows.append(
                [
                    model.spec,
                    row.get("query_id", ""),
                    row.get("query", ""),
                    "|".join(row.get("relevant_doc_ids", [])),
                    "|".join(row.get("ranked_doc_ids", [])),
                    metrics.get("hit_rate", ""),
                    metrics.get("mrr", ""),
                    metrics.get("ndcg", ""),
                    metrics.get("precision", ""),
                    metrics.get("recall", ""),
                    row.get("latency_ms", ""),
                ]
            )
    target = Path(path)
    atomic_write_text(target, _csv_text(rows, headers))
    return target


def write_json(result: BenchmarkResult, path: Path, *, priority: str = "balanced") -> Path:
    """Write the full result as JSON, including the recommendation.

    Args:
        result: Benchmark result.
        path: Destination file.
        priority: Recommendation weighting stored under ``recommendation``.

    Returns:
        The path that was written.
    """

    payload = result.to_dict()
    payload["recommendation"] = result.recommend(priority)
    target = Path(path)
    _write_json_file(target, payload)
    return target


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ReportError(f"Plotting needs matplotlib. Install it with: {_REQUIRED_EXTRA}") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_latency_vs_quality(
    result: BenchmarkResult,
    path: Path,
    *,
    k: int | None = None,
) -> Path:
    """Scatter p95 latency against nDCG, the classic trade-off view.

    Args:
        result: Benchmark result.
        path: Destination PNG.
        k: Cut-off for the quality axis.

    Returns:
        The path that was written.

    Raises:
        ReportError: If matplotlib is missing or there is nothing to plot.
    """

    plt = _matplotlib()
    cut_off = k or result.reference_k
    points = [
        (model.latency.get("p95_ms"), model.metrics_at(cut_off).get("ndcg", 0.0), model)
        for model in result.successful
    ]
    points = [item for item in points if isinstance(item[0], (int, float))]
    if not points:
        raise ReportError("No latency measurements available to plot")

    figure, axis = plt.subplots(figsize=(7.5, 4.8), dpi=140)
    xs = [float(item[0]) for item in points]
    ys = [float(item[1]) for item in points]
    axis.scatter(xs, ys, s=70, color="#2f6feb", edgecolor="white", linewidth=0.8, zorder=3)
    for x_value, y_value, model in points:
        axis.annotate(
            _short(model.spec, 26),
            (float(x_value), float(y_value)),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7.5,
            color="#333333",
        )
    axis.set_xlabel("p95 latency (ms)")
    axis.set_ylabel(f"nDCG @ {cut_off}")
    axis.set_title("Quality vs latency (up and to the left is better)")
    axis.grid(True, linestyle=":", alpha=0.45)
    axis.margins(0.15)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return target


def plot_quality_bars(
    result: BenchmarkResult,
    path: Path,
    *,
    k: int | None = None,
) -> Path:
    """Grouped bars comparing hit rate, MRR and nDCG per model.

    Args:
        result: Benchmark result.
        path: Destination PNG.
        k: Cut-off for the metric columns.

    Returns:
        The path that was written.

    Raises:
        ReportError: If matplotlib is missing or there is nothing to plot.
    """

    plt = _matplotlib()
    cut_off = k or result.reference_k
    models = [model for model in result.successful if model.metrics_at(cut_off)]
    if not models:
        raise ReportError("No retrieval metrics available to plot")

    labels = [_short(model.spec, 22) for model in models]
    series = [
        ("Hit rate", [model.metrics_at(cut_off).get("hit_rate", 0.0) for model in models]),
        ("MRR", [model.metrics_at(cut_off).get("mrr", 0.0) for model in models]),
        ("nDCG", [model.metrics_at(cut_off).get("ndcg", 0.0) for model in models]),
    ]
    figure, axis = plt.subplots(figsize=(max(6.5, 1.6 * len(models) + 3), 4.4), dpi=140)
    width = 0.8 / len(series)
    positions = range(len(models))
    colors = ("#2f6feb", "#7aa2f7", "#9ece6a")
    for offset, (name, values) in enumerate(series):
        shifted = [position + offset * width - 0.4 + width / 2 for position in positions]
        axis.bar(shifted, values, width=width, label=name, color=colors[offset % len(colors)])
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel(f"Score @ {cut_off}")
    axis.set_title("Retrieval quality by model")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(True, axis="y", linestyle=":", alpha=0.45)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return target


def plot_cost_vs_quality(
    result: BenchmarkResult,
    path: Path,
    *,
    k: int | None = None,
) -> Path:
    """Scatter estimated cost against nDCG for models with a known price.

    Args:
        result: Benchmark result.
        path: Destination PNG.
        k: Cut-off for the quality axis.

    Returns:
        The path that was written.

    Raises:
        ReportError: If matplotlib is missing or fewer than two priced models
            exist (a single point is not a comparison).
    """

    plt = _matplotlib()
    cut_off = k or result.reference_k
    points = [
        (model.cost.get("usd"), model.metrics_at(cut_off).get("ndcg", 0.0), model)
        for model in result.successful
        if model.cost.get("usd") is not None
    ]
    if len(points) < 2:
        raise ReportError("Need at least two models with known cost to plot value for money")

    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    xs = [max(float(item[0]), 1e-6) for item in points]
    ys = [float(item[1]) for item in points]
    axis.scatter(xs, ys, s=70, color="#9ece6a", edgecolor="white", linewidth=0.8, zorder=3)
    for x_value, y_value, model in points:
        axis.annotate(
            _short(model.spec, 26),
            (float(x_value), float(y_value)),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7.5,
            color="#333333",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Estimated cost (USD, log scale)")
    axis.set_ylabel(f"nDCG @ {cut_off}")
    axis.set_title("Quality vs cost")
    axis.grid(True, linestyle=":", alpha=0.45)
    axis.margins(0.15)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return target


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def write_reports(
    result: BenchmarkResult,
    output_dir: Path,
    *,
    priority: str = "balanced",
    plots: bool = True,
    markdown: bool = True,
) -> dict[str, Path]:
    """Write the standard artifact bundle for a benchmark result.

    Args:
        result: Benchmark result.
        output_dir: Directory that receives the artifacts.
        priority: Recommendation weighting stored in JSON and Markdown.
        plots: Render PNG comparisons when matplotlib is installed.
        markdown: Write ``results.md``.

    Returns:
        Mapping of artifact name to the written path.  Plot entries are absent
        when matplotlib is unavailable or there is nothing to plot.

    Raises:
        ReportError: If the output directory cannot be created.
    """

    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReportError(f"Could not create output directory {directory}: {exc}") from exc

    artifacts: dict[str, Path] = {}
    artifacts["results_json"] = write_json(result, directory / "results.json", priority=priority)
    artifacts["summary_csv"] = write_summary_csv(result, directory / "summary.csv")
    artifacts["queries_csv"] = write_queries_csv(result, directory / "queries.csv")
    if markdown:
        artifacts["results_md"] = write_markdown(
            result, directory / "results.md", priority=priority
        )
    if plots:
        for name, plotter in (
            ("latency_vs_quality_png", plot_latency_vs_quality),
            ("quality_bars_png", plot_quality_bars),
        ):
            try:
                artifacts[name] = plotter(result, directory / _plot_name(name))
            except ReportError as exc:
                logger.info("Skipping plot %s: %s", name, exc)
        try:
            artifacts["cost_vs_quality_png"] = plot_cost_vs_quality(
                result, directory / "cost-vs-quality.png"
            )
        except ReportError as exc:
            logger.debug("Skipping cost plot: %s", exc)
    return artifacts


def _plot_name(key: str) -> str:
    return {
        "latency_vs_quality_png": "latency-vs-quality.png",
        "quality_bars_png": "quality-bars.png",
        "cost_vs_quality_png": "cost-vs-quality.png",
    }[key]


def write_markdown(result: BenchmarkResult, path: Path, *, priority: str = "balanced") -> Path:
    """Write the Markdown report.

    Args:
        result: Benchmark result.
        path: Destination file.
        priority: Recommendation weighting.

    Returns:
        The path that was written.
    """

    target = Path(path)
    atomic_write_text(target, render_markdown(result, priority=priority))
    return target


__all__ = [
    "format_metric",
    "format_usd",
    "markdown_cell",
    "plot_cost_vs_quality",
    "plot_latency_vs_quality",
    "plot_quality_bars",
    "render_console",
    "render_markdown",
    "write_json",
    "write_markdown",
    "write_queries_csv",
    "write_reports",
    "write_summary_csv",
]
