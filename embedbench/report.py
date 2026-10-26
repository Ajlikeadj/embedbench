"""Console, JSON, CSV and self-contained HTML reporting."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from embedbench.errors import ReportError
from embedbench.logging_utils import get_logger
from embedbench.utils import atomic_write_text, json_default, utc_now_iso, write_json

logger = get_logger(__name__)

PRIORITIES = ("accuracy", "speed", "balanced")
PRIORITY_LABELS = {
    "accuracy": "Accuracy-first",
    "speed": "Speed-first",
    "balanced": "Balanced",
}
BALANCED_WEIGHTS = {"accuracy": 0.5, "speed": 0.3, "cost": 0.2}


def _dig(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _accuracy_value(model: Mapping[str, Any]) -> float:
    return _float_or_none(_dig(model, "accuracy", "ndcg", default=0.0)) or 0.0


def _latency_value(model: Mapping[str, Any]) -> float | None:
    latency = _dig(model, "latency_ms", "p95_ms")
    if latency is None:
        latency = _dig(model, "latency_ms", "p50_ms")
    return _float_or_none(latency)


def _cost_value(model: Mapping[str, Any]) -> float | None:
    return _float_or_none(_dig(model, "tokens", "cost_usd"))


def _min_max(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    if not items:
        return 0.0, 0.0
    return min(items), max(items)


def _normalize_up(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return (value - low) / (high - low)


def _normalize_down(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return (high - value) / (high - low)


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}{suffix}"


def _fmt_usd(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "$0"
    if abs(number) < 0.000001:
        return "<$0.000001"
    return f"${number:.6f}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def recommend_models(
    models: Sequence[Mapping[str, Any]],
    priority: str = "balanced",
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Pick the best model for each priority using normalized in-run scores."""

    if priority not in PRIORITIES:
        priority = "balanced"
    successful = [model for model in models if model.get("status") == "ok"]
    empty = {
        "priority": priority,
        "selected": None,
        "rationale": "No model completed successfully, so no recommendation can be made.",
        "by_priority": {},
        "scores": [],
        "weights": dict(BALANCED_WEIGHTS),
        "top_k": top_k,
    }
    if not successful:
        return empty

    accuracy_values = {model["spec"]: _accuracy_value(model) for model in successful}
    latency_values = {
        model["spec"]: _latency_value(model) for model in successful
    }
    cost_values = {model["spec"]: _cost_value(model) for model in successful}
    valid_latencies = [value for value in latency_values.values() if value is not None]
    valid_costs = [value for value in cost_values.values() if value is not None]
    acc_min, acc_max = _min_max(accuracy_values.values())
    lat_min, lat_max = _min_max(valid_latencies)
    cost_min, cost_max = _min_max(valid_costs)

    scores: list[dict[str, Any]] = []
    for model in successful:
        spec = model["spec"]
        accuracy = accuracy_values[spec]
        latency = latency_values[spec]
        cost = cost_values[spec]
        accuracy_score = _normalize_up(accuracy, acc_min, acc_max)
        speed_score = (
            _normalize_down(latency, lat_min, lat_max) if latency is not None else 0.0
        )
        cost_score = _normalize_down(cost, cost_min, cost_max) if cost is not None else 0.0
        balanced_score = (
            BALANCED_WEIGHTS["accuracy"] * accuracy_score
            + BALANCED_WEIGHTS["speed"] * speed_score
            + BALANCED_WEIGHTS["cost"] * cost_score
        )
        scores.append(
            {
                "spec": spec,
                "ndcg": accuracy,
                "accuracy_score": round(accuracy_score, 4),
                "speed_score": round(speed_score, 4),
                "cost_score": round(cost_score, 4),
                "balanced_score": round(balanced_score, 4),
                "p95_ms": latency,
                "cost_usd": cost,
            }
        )

    def accuracy_sort(item: Mapping[str, Any]) -> tuple[Any, ...]:
        lat = latency_values[item["spec"]]
        return (
            -accuracy_values[item["spec"]],
            -_float_or_none(_dig(item, "accuracy", "recall", default=0.0)) or 0.0,
            -_float_or_none(_dig(item, "accuracy", "mrr", default=0.0)) or 0.0,
            lat if lat is not None else math.inf,
            item["spec"],
        )

    def speed_sort(item: Mapping[str, Any]) -> tuple[Any, ...]:
        lat = latency_values[item["spec"]]
        return (
            lat if lat is not None else math.inf,
            -accuracy_values[item["spec"]],
            item["spec"],
        )

    balanced_lookup = {row["spec"]: row["balanced_score"] for row in scores}

    def balanced_sort(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -balanced_lookup[item["spec"]],
            -accuracy_values[item["spec"]],
            item["spec"],
        )

    accuracy_winner = sorted(successful, key=accuracy_sort)[0]
    speed_winner = sorted(successful, key=speed_sort)[0]
    balanced_winner = sorted(successful, key=balanced_sort)[0]

    def rationale_for(kind: str, winner: Mapping[str, Any]) -> str:
        spec = winner["spec"]
        n = _fmt(accuracy_values[spec], 3)
        recall = _fmt(_dig(winner, "accuracy", "recall"), 3)
        mrr = _fmt(_dig(winner, "accuracy", "mrr"), 3)
        p95 = _fmt(latency_values[spec], 2, " ms")
        cost = _fmt_usd(cost_values[spec])
        if kind == "accuracy":
            return (
                f"Highest nDCG@{top_k} ({n}) with recall {recall} and MRR {mrr}; "
                f"p95 query latency {p95}."
            )
        if kind == "speed":
            return (
                f"Lowest p95 query latency ({p95}) while maintaining nDCG@{top_k} "
                f"of {n}."
            )
        balanced = balanced_lookup[spec]
        return (
            f"Best weighted score ({balanced:.3f}) using 50% accuracy, 30% speed and "
            f"20% cost-equivalent: nDCG@{top_k}={n}, p95={p95}, estimated cost {cost}."
        )

    by_priority = {
        kind: {
            "priority": kind,
            "label": PRIORITY_LABELS[kind],
            "model": winner["spec"],
            "rationale": rationale_for(kind, winner),
            "ndcg": round(accuracy_values[winner["spec"]], 4),
            "p95_ms": latency_values[winner["spec"]],
            "cost_usd": cost_values[winner["spec"]],
        }
        for kind, winner in (
            ("accuracy", accuracy_winner),
            ("speed", speed_winner),
            ("balanced", balanced_winner),
        )
    }

    return {
        "priority": priority,
        "selected": by_priority[priority]["model"],
        "rationale": by_priority[priority]["rationale"],
        "by_priority": by_priority,
        "scores": sorted(scores, key=lambda row: row["spec"]),
        "weights": dict(BALANCED_WEIGHTS),
        "top_k": top_k,
    }


def _recommendation_table(results: Mapping[str, Any]) -> Table | None:
    recommendation = results.get("recommendation") or {}
    scores = recommendation.get("scores") or []
    successful = [model for model in results.get("models", []) if model.get("status") == "ok"]
    if not scores or not successful:
        return None
    table = Table(title="Recommendation scores", show_lines=False, box=None)
    table.add_column("Model", style="bold", overflow="fold", max_width=30)
    table.add_column("Acc.", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Balanced", justify="right")
    for row in scores:
        table.add_row(
            escape(str(row.get("spec", ""))),
            _fmt(row.get("accuracy_score"), 3),
            _fmt(row.get("speed_score"), 3),
            _fmt(row.get("cost_score"), 3),
            _fmt(row.get("balanced_score"), 3),
        )
    return table


def console_summary(results: Mapping[str, Any], console: Console) -> None:
    """Print accuracy, latency and resource tables plus a recommendation."""

    config = results.get("config") or {}
    run = results.get("run") or {}
    top_k = int(config.get("top_k") or run.get("top_k") or 5)
    models = results.get("models") or []
    successful = [model for model in models if model.get("status") == "ok"]
    failed = [model for model in models if model.get("status") != "ok"]

    corpus = run.get("corpus") or {}
    console.print()
    console.rule("[bold]embedbench results[/bold]")
    console.print(
        f"[dim]Corpus:[/dim] {_fmt_int(corpus.get('documents'))} documents | "
        f"{_fmt_int(corpus.get('chunks'))} chunks | "
        f"{_fmt_int(run.get('evalset', {}).get('queries'))} queries | "
        f"seed {run.get('seed', config.get('seed', 'n/a'))} | "
        f"k={top_k}"
    )

    accuracy_table = Table(
        title=f"Retrieval accuracy (k={top_k}, document-level relevance)",
        header_style="bold cyan",
        box=None,
    )
    accuracy_table.add_column("Model", style="bold", overflow="fold", max_width=30)
    accuracy_table.add_column("nDCG", justify="right")
    accuracy_table.add_column("Recall", justify="right")
    accuracy_table.add_column("Prec.", justify="right")
    accuracy_table.add_column("MRR", justify="right")
    accuracy_table.add_column("Hit", justify="right")

    performance_table = Table(
        title="Latency, indexing and throughput",
        header_style="bold cyan",
        box=None,
    )
    performance_table.add_column("Model", style="bold", overflow="fold", max_width=30)
    performance_table.add_column("p50 ms", justify="right")
    performance_table.add_column("p95 ms", justify="right")
    performance_table.add_column("p99 ms", justify="right")
    performance_table.add_column("Index s", justify="right")
    performance_table.add_column("Chunks/s", justify="right")
    performance_table.add_column("Queries/s", justify="right")

    resource_table = Table(
        title="Model footprint and cost-equivalent",
        header_style="bold cyan",
        box=None,
    )
    resource_table.add_column("Model", style="bold", overflow="fold", max_width=30)
    resource_table.add_column("Dim", justify="right")
    resource_table.add_column("Index MB", justify="right")
    resource_table.add_column("Tokens", justify="right")
    resource_table.add_column("Cost est.", justify="right")
    resource_table.add_column("RAM MB", justify="right")
    resource_table.add_column("VRAM MB", justify="right")

    for model in successful:
        accuracy = model.get("accuracy") or {}
        latency = model.get("latency_ms") or {}
        timing = model.get("timing") or {}
        throughput = model.get("throughput") or {}
        tokens = model.get("tokens") or {}
        resources = model.get("resources") or {}
        index = model.get("index") or {}
        accuracy_table.add_row(
            escape(str(model.get("spec", ""))),
            _fmt(accuracy.get("ndcg"), 3),
            _fmt(accuracy.get("recall"), 3),
            _fmt(accuracy.get("precision"), 3),
            _fmt(accuracy.get("mrr"), 3),
            _fmt(accuracy.get("hit_rate"), 3),
        )
        performance_table.add_row(
            escape(str(model.get("spec", ""))),
            _fmt(latency.get("p50_ms"), 1),
            _fmt(latency.get("p95_ms"), 1),
            _fmt(latency.get("p99_ms"), 1),
            _fmt(timing.get("indexing_seconds"), 2),
            _fmt(throughput.get("chunks_per_second"), 1),
            _fmt(throughput.get("queries_per_second"), 1),
        )
        resource_table.add_row(
            escape(str(model.get("spec", ""))),
            _fmt_int(index.get("dimension")),
            _fmt(index.get("disk_size_mb"), 3),
            _fmt_int(tokens.get("total_tokens")) + ("*" if tokens.get("estimated") else ""),
            _fmt_usd(tokens.get("cost_usd")),
            _fmt(resources.get("ram_peak_mb"), 1),
            _fmt(resources.get("peak_vram_mb"), 1),
        )

    if successful:
        console.print(accuracy_table)
        console.print(performance_table)
        console.print(resource_table)
        if any((model.get("tokens") or {}).get("estimated") for model in successful):
            console.print(
                "[dim]* Token count estimated: Ollama does not expose the model tokenizer.[/dim]"
            )
        if any(
            ((model.get("resources") or {}).get("vram_measurement") == "ollama_api_ps_resident")
            for model in successful
        ):
            console.print(
                "[dim]VRAM for Ollama is resident model size from /api/ps, not a sampled peak.[/dim]"
            )
        reused = [
            f"{model.get('spec')} {((model.get('cache') or {}).get('hit_rate') or 0) * 100:.0f}%"
            for model in successful
            if ((model.get("cache") or {}).get("hit_rate") or 0) > 0
        ]
        if reused:
            console.print(
                "[dim]Cache reuse: "
                + " | ".join(escape(item) for item in reused)
                + "; indexing throughput includes cache lookups.[/dim]"
            )
    else:
        console.print(Panel("No model completed successfully.", style="red"))

    recommendation = results.get("recommendation") or {}
    if recommendation.get("selected"):
        priority = recommendation.get("priority", "balanced")
        by_priority = recommendation.get("by_priority") or {}
        lines = [
            Text.from_markup(
                f"[bold]Selected:[/bold] {escape(str(recommendation['selected']))} "
                f"({PRIORITY_LABELS.get(priority, priority)})"
            ),
            Text(recommendation.get("rationale", "")),
        ]
        alternatives = [
            f"{PRIORITY_LABELS.get(key, key)}: {escape(str(value.get('model', 'n/a')))}"
            for key, value in by_priority.items()
            if key != priority
        ]
        if alternatives:
            lines.append(Text("Alternatives - " + " | ".join(alternatives)))
        console.print(Panel(Text("\n").join(lines), title="Recommendation", border_style="green"))
        score_table = _recommendation_table(results)
        if score_table is not None:
            console.print(score_table)

    if failed:
        failure_lines = [
            Text(f"{model.get('spec', 'unknown')}: {model.get('error', 'unknown error')}", style="red")
            for model in failed
        ]
        console.print(
            Panel(
                Text("\n").join(failure_lines),
                title=f"Failed models ({len(failed)})",
                border_style="red",
            )
        )

    artifacts = results.get("artifacts") or {}
    if artifacts:
        console.print("[dim]Artifacts:[/dim]")
        for key, path in artifacts.items():
            console.print(f"  [dim]{escape(str(key))}:[/dim] {escape(str(path))}")


_SUMMARY_COLUMNS = [
    "spec",
    "backend",
    "model",
    "status",
    "error",
    "metrics_k",
    "n_queries",
    "recall",
    "precision",
    "mrr",
    "ndcg",
    "hit_rate",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "cold_query_ms",
    "warm_p50_ms",
    "warm_p95_ms",
    "warm_p99_ms",
    "cold_embed_ms",
    "warm_batch_p50_ms",
    "warm_batch_p95_ms",
    "queries_per_second",
    "model_load_ms",
    "indexing_seconds",
    "embedding_seconds",
    "index_build_ms",
    "chunks_per_second",
    "embed_chunks_per_second",
    "index_backend",
    "index_dimension",
    "index_vectors",
    "index_disk_size_mb",
    "index_tokens",
    "query_tokens",
    "total_tokens",
    "tokens_estimated",
    "token_method",
    "cost_usd",
    "price_per_1m_tokens",
    "ram_peak_mb",
    "model_resident_mb",
    "gpu_peak_allocated_mb",
    "peak_vram_mb",
    "cache_hits",
    "cache_misses",
    "cache_hit_rate",
    "total_seconds",
]


def _summary_row(model: Mapping[str, Any], price_per_1m: Any) -> dict[str, Any]:
    accuracy = model.get("accuracy") or {}
    latency = model.get("latency_ms") or {}
    timing = model.get("timing") or {}
    throughput = model.get("throughput") or {}
    index = model.get("index") or {}
    tokens = model.get("tokens") or {}
    resources = model.get("resources") or {}
    cache = model.get("cache") or {}
    row = {
        "spec": model.get("spec"),
        "backend": model.get("backend"),
        "model": model.get("model"),
        "status": model.get("status"),
        "error": model.get("error"),
        "metrics_k": model.get("metrics_k"),
        "n_queries": accuracy.get("n_queries"),
        "recall": accuracy.get("recall"),
        "precision": accuracy.get("precision"),
        "mrr": accuracy.get("mrr"),
        "ndcg": accuracy.get("ndcg"),
        "hit_rate": accuracy.get("hit_rate"),
        "p50_ms": latency.get("p50_ms"),
        "p95_ms": latency.get("p95_ms"),
        "p99_ms": latency.get("p99_ms"),
        "cold_query_ms": latency.get("cold_query_ms"),
        "warm_p50_ms": latency.get("warm_p50_ms"),
        "warm_p95_ms": latency.get("warm_p95_ms"),
        "warm_p99_ms": latency.get("warm_p99_ms"),
        "cold_embed_ms": timing.get("cold_embed_ms"),
        "warm_batch_p50_ms": timing.get("warm_batch_p50_ms"),
        "warm_batch_p95_ms": timing.get("warm_batch_p95_ms"),
        "queries_per_second": latency.get("queries_per_second"),
        "model_load_ms": timing.get("model_load_ms"),
        "indexing_seconds": timing.get("indexing_seconds"),
        "embedding_seconds": timing.get("embedding_seconds"),
        "index_build_ms": timing.get("index_build_ms"),
        "chunks_per_second": throughput.get("chunks_per_second"),
        "embed_chunks_per_second": throughput.get("embed_chunks_per_second"),
        "index_backend": index.get("backend"),
        "index_dimension": index.get("dimension"),
        "index_vectors": index.get("vectors"),
        "index_disk_size_mb": index.get("disk_size_mb"),
        "index_tokens": tokens.get("index_tokens"),
        "query_tokens": tokens.get("query_tokens"),
        "total_tokens": tokens.get("total_tokens"),
        "tokens_estimated": tokens.get("estimated"),
        "token_method": tokens.get("method"),
        "cost_usd": tokens.get("cost_usd"),
        "price_per_1m_tokens": tokens.get("price_per_1m_tokens", price_per_1m),
        "ram_peak_mb": resources.get("ram_peak_mb"),
        "model_resident_mb": resources.get("model_resident_mb"),
        "gpu_peak_allocated_mb": resources.get("gpu_peak_allocated_mb"),
        "peak_vram_mb": resources.get("peak_vram_mb"),
        "cache_hits": cache.get("hits"),
        "cache_misses": cache.get("misses"),
        "cache_hit_rate": cache.get("hit_rate"),
        "total_seconds": timing.get("total_seconds") or model.get("duration_seconds"),
    }
    return row


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: ("" if row.get(column) is None else row.get(column)) for column in columns})
    return buffer.getvalue()


def write_summary_csv(results: Mapping[str, Any], path: Path) -> Path:
    price = _dig(results, "run", "price_per_1m_tokens", default=None)
    rows = [_summary_row(model, price) for model in results.get("models", [])]
    atomic_write_text(Path(path), _csv_text(rows, _SUMMARY_COLUMNS))
    return Path(path)


_QUERY_COLUMNS = [
    "model",
    "query_id",
    "query",
    "latency_ms",
    "recall",
    "precision",
    "mrr",
    "ndcg",
    "hit_rate",
    "retrieved_doc_ids",
    "relevant_doc_ids",
]


def write_queries_csv(results: Mapping[str, Any], path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for model in results.get("models", []):
        if model.get("status") != "ok":
            continue
        for query in model.get("per_query", []):
            metrics = query.get("metrics") or {}
            rows.append(
                {
                    "model": model.get("spec"),
                    "query_id": query.get("query_id"),
                    "query": query.get("query"),
                    "latency_ms": query.get("latency_ms"),
                    "recall": metrics.get("recall"),
                    "precision": metrics.get("precision"),
                    "mrr": metrics.get("mrr"),
                    "ndcg": metrics.get("ndcg"),
                    "hit_rate": metrics.get("hit_rate"),
                    "retrieved_doc_ids": "|".join(query.get("ranked_doc_ids") or []),
                    "relevant_doc_ids": "|".join(query.get("relevant_doc_ids") or []),
                }
            )
    atomic_write_text(Path(path), _csv_text(rows, _QUERY_COLUMNS))
    return Path(path)


def load_results(path: Path) -> dict[str, Any]:
    """Load a ``results.json`` file, accepting either the file or its directory."""

    path = Path(path)
    candidate = path / "results.json" if path.is_dir() else path
    if not candidate.is_file():
        raise ReportError(f"Results file not found: {candidate}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"Could not read results from {candidate}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ReportError(f"{candidate} is not an embedbench results file")
    return payload


def write_reports(
    results: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write results.json, CSVs and report.html; JSON is written first for safety."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "results_json": output_dir / "results.json",
        "results_csv": output_dir / "results.csv",
        "queries_csv": output_dir / "queries.csv",
        "report_html": output_dir / "report.html",
        "config_snapshot": output_dir / "config.snapshot.yaml",
    }
    results["artifacts"] = {key: str(value) for key, value in artifacts.items()}

    write_json(artifacts["results_json"], results)
    write_summary_csv(results, artifacts["results_csv"])
    write_queries_csv(results, artifacts["queries_csv"])
    try:
        write_html_report(results, artifacts["report_html"])
    except Exception:
        logger.exception("HTML report generation failed; JSON and CSV artifacts are intact")
        artifacts.pop("report_html")
        results["artifacts"].pop("report_html", None)
        write_json(artifacts["results_json"], results)
        raise
    return artifacts


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _short_spec(spec: str, limit: int = 34) -> str:
    return spec if len(spec) <= limit else spec[: limit - 1] + "…"


def _accuracy_table_html(models: Sequence[Mapping[str, Any]], top_k: int) -> str:
    successful = [model for model in models if model.get("status") == "ok"]
    if not successful:
        return '<p class="muted">No model completed successfully.</p>'
    rows = []
    for model in successful:
        accuracy = model.get("accuracy") or {}
        latency = model.get("latency_ms") or {}
        rows.append(
            "<tr>"
            f"<td><strong>{_e(model.get('spec'))}</strong></td>"
            f"<td class='num'>{_fmt(accuracy.get('ndcg'), 4)}</td>"
            f"<td class='num'>{_fmt(accuracy.get('recall'), 4)}</td>"
            f"<td class='num'>{_fmt(accuracy.get('precision'), 4)}</td>"
            f"<td class='num'>{_fmt(accuracy.get('mrr'), 4)}</td>"
            f"<td class='num'>{_fmt(accuracy.get('hit_rate'), 4)}</td>"
            f"<td class='num'>{_fmt(latency.get('p50_ms'), 2)}</td>"
            f"<td class='num'>{_fmt(latency.get('p95_ms'), 2)}</td>"
            f"<td class='num'>{_fmt(latency.get('p99_ms'), 2)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr>"
        "<th>Model</th>"
        f"<th class='num'>nDCG@{top_k}</th>"
        f"<th class='num'>Recall@{top_k}</th>"
        f"<th class='num'>Precision@{top_k}</th>"
        f"<th class='num'>MRR@{top_k}</th>"
        f"<th class='num'>Hit@{top_k}</th>"
        "<th class='num'>p50 ms</th><th class='num'>p95 ms</th><th class='num'>p99 ms</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _resource_table_html(models: Sequence[Mapping[str, Any]]) -> str:
    successful = [model for model in models if model.get("status") == "ok"]
    if not successful:
        return '<p class="muted">No resource data.</p>'
    rows = []
    for model in successful:
        timing = model.get("timing") or {}
        throughput = model.get("throughput") or {}
        index = model.get("index") or {}
        tokens = model.get("tokens") or {}
        resources = model.get("resources") or {}
        cache = model.get("cache") or {}
        rows.append(
            "<tr>"
            f"<td><strong>{_e(model.get('spec'))}</strong></td>"
            f"<td>{_e(index.get('backend'))}</td>"
            f"<td class='num'>{_fmt_int(index.get('dimension'))}</td>"
            f"<td class='num'>{_fmt(index.get('disk_size_mb'), 2)}</td>"
            f"<td class='num'>{_fmt(timing.get('indexing_seconds'), 2)}</td>"
            f"<td class='num'>{_fmt(throughput.get('chunks_per_second'), 1)}</td>"
            f"<td class='num'>{_fmt(throughput.get('queries_per_second'), 1)}</td>"
            f"<td class='num'>{_fmt_int(tokens.get('total_tokens'))}{'*' if tokens.get('estimated') else ''}</td>"
            f"<td class='num'>{_e(_fmt_usd(tokens.get('cost_usd')))}</td>"
            f"<td class='num'>{_fmt(resources.get('ram_peak_mb'), 1)}</td>"
            f"<td class='num'>{_fmt(resources.get('model_resident_mb'), 1)}</td>"
            f"<td class='num'>{_fmt(resources.get('peak_vram_mb'), 1)}</td>"
            f"<td class='num'>{_fmt(cache.get('hit_rate'), 2)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Model</th><th>Index</th><th class='num'>Dim</th>"
        "<th class='num'>Index MB</th><th class='num'>Index s</th>"
        "<th class='num'>Chunks/s</th><th class='num'>Queries/s</th>"
        "<th class='num'>Tokens</th><th class='num'>Cost est.</th>"
        "<th class='num'>Process RAM MB</th><th class='num'>Model MB</th><th class='num'>VRAM MB</th>"
        "<th class='num'>Cache hit</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _timing_table_html(models: Sequence[Mapping[str, Any]]) -> str:
    successful = [model for model in models if model.get("status") == "ok"]
    if not successful:
        return '<p class="muted">No timing data.</p>'
    rows = []
    for model in successful:
        timing = model.get("timing") or {}
        latency = model.get("latency_ms") or {}
        rows.append(
            "<tr>"
            f"<td><strong>{_e(model.get('spec'))}</strong></td>"
            f"<td class='num'>{_fmt(timing.get('model_load_ms'), 1)}</td>"
            f"<td class='num'>{_fmt(timing.get('cold_embed_ms'), 1)}</td>"
            f"<td class='num'>{_fmt(timing.get('warm_batch_p50_ms'), 1)}</td>"
            f"<td class='num'>{_fmt(timing.get('warm_batch_p95_ms'), 1)}</td>"
            f"<td class='num'>{_fmt(latency.get('cold_query_ms'), 1)}</td>"
            f"<td class='num'>{_fmt(latency.get('warm_p50_ms'), 1)}</td>"
            f"<td class='num'>{_fmt(latency.get('warm_p95_ms'), 1)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="compact">'
        "<thead><tr><th>Model</th><th class='num'>Load ms</th>"
        "<th class='num'>Cold embed ms</th><th class='num'>Warm batch p50</th>"
        "<th class='num'>Warm batch p95</th><th class='num'>Cold query ms</th>"
        "<th class='num'>Warm query p50</th><th class='num'>Warm query p95</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _recommendation_html(
    recommendation: Mapping[str, Any] | None,
    models: Sequence[Mapping[str, Any]],
    top_k: int,
) -> str:
    if not recommendation or not recommendation.get("selected"):
        return (
            '<div class="callout warn">No recommendation is available because no model '
            "completed successfully. See the failures section below.</div>"
        )
    priority = recommendation.get("priority", "balanced")
    selected = str(recommendation.get("selected"))
    cards = []
    for key in PRIORITIES:
        item = (recommendation.get("by_priority") or {}).get(key)
        if not item:
            continue
        chosen = key == priority
        chosen_badge = '<span class="chosen-badge">Selected</span>' if chosen else ""
        card_class = "priority-card chosen" if chosen else "priority-card"
        cards.append(
            f'<article class="{card_class}">'
            f'<div class="priority-top"><span class="tag">{_e(PRIORITY_LABELS[key])}</span>'
            f'{chosen_badge}</div>'
            f'<h3>{_e(item.get("model"))}</h3>'
            f'<p>{_e(item.get("rationale"))}</p>'
            f'<div class="mini-stats"><span>nDCG@{top_k} <b>{_fmt(item.get("ndcg"), 3)}</b></span>'
            f'<span>p95 <b>{_fmt(item.get("p95_ms"), 1)} ms</b></span>'
            f'<span>Cost <b>{_e(_fmt_usd(item.get("cost_usd")))}</b></span></div>'
            "</article>"
        )

    scores = recommendation.get("scores") or []
    score_rows = []
    for row in scores:
        score_rows.append(
            "<tr>"
            f"<td>{_e(row.get('spec'))}</td>"
            f"<td class='num'>{_fmt(row.get('ndcg'), 4)}</td>"
            f"<td class='num'>{_fmt(row.get('p95_ms'), 2)}</td>"
            f"<td class='num'>{_fmt(row.get('accuracy_score'), 3)}</td>"
            f"<td class='num'>{_fmt(row.get('speed_score'), 3)}</td>"
            f"<td class='num'>{_fmt(row.get('cost_score'), 3)}</td>"
            f"<td class='num'>{_fmt(row.get('balanced_score'), 3)}</td>"
            "</tr>"
        )
    score_table = (
        '<div class="table-wrap"><table class="compact">'
        "<thead><tr><th>Model</th><th class='num'>nDCG</th><th class='num'>p95 ms</th>"
        "<th class='num'>Accuracy score</th><th class='num'>Speed score</th>"
        "<th class='num'>Cost score</th><th class='num'>Balanced</th></tr></thead><tbody>"
        + "".join(score_rows)
        + "</tbody></table></div>"
    )
    return (
        f'<div class="selected-callout"><div class="selected-label">Best for '
        f"{_e(PRIORITY_LABELS.get(priority, priority))}</div>"
        f'<div class="selected-model">{_e(selected)}</div>'
        f'<p>{_e(recommendation.get("rationale"))}</p></div>'
        f'<div class="priority-grid">{"".join(cards)}</div>'
        '<p class="muted">Balanced weights: 50% accuracy (nDCG), 30% speed (p95 latency), '
        "20% cost. Scores are min–max normalized within this run, so they compare these "
        "models on this corpus only.</p>"
        + score_table
    )


def _per_query_html(models: Sequence[Mapping[str, Any]], top_k: int) -> str:
    sections = []
    for model in models:
        if model.get("status") != "ok":
            continue
        rows = []
        for query in model.get("per_query") or []:
            metrics = query.get("metrics") or {}
            retrieved = query.get("ranked_doc_ids") or []
            top_docs = "<br>".join(f"<code>{_e(doc)}</code>" for doc in retrieved[:top_k])
            top_docs_html = top_docs or '<span class="muted">none</span>'
            rows.append(
                "<tr>"
                f"<td>{_e(query.get('query_id'))}</td>"
                f"<td>{_e(query.get('query'))}</td>"
                f"<td class='num'>{_fmt(query.get('latency_ms'), 2)}</td>"
                f"<td class='num'>{_fmt(metrics.get('ndcg'), 3)}</td>"
                f"<td class='num'>{_fmt(metrics.get('mrr'), 3)}</td>"
                f"<td class='num'>{_fmt(metrics.get('hit_rate'), 1)}</td>"
                f"<td>{top_docs_html}</td>"
                "</tr>"
            )
        sections.append(
            '<details class="panel query-details">'
            f"<summary>Per-query results — {_e(model.get('spec'))} ({len(rows)} queries)</summary>"
            '<div class="table-wrap"><table class="compact">'
            "<thead><tr><th>ID</th><th>Query</th><th class='num'>Latency ms</th>"
            "<th class='num'>nDCG</th><th class='num'>MRR</th><th class='num'>Hit</th>"
            "<th>Ranked documents</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div></details>"
        )
    return "".join(sections)


def _failures_html(models: Sequence[Mapping[str, Any]]) -> str:
    failures = [model for model in models if model.get("status") != "ok"]
    if not failures:
        return ""
    items = "".join(
        f"<li><strong>{_e(model.get('spec'))}</strong><p>{_e(model.get('error'))}</p></li>"
        for model in failures
    )
    return f'<section class="panel"><h2>Failures</h2><ul class="failures">{items}</ul></section>'


def _stat_cards(results: Mapping[str, Any], successful: int, total: int) -> str:
    corpus = _dig(results, "run", "corpus", default={}) or {}
    evalset = _dig(results, "run", "evalset", default={}) or {}
    config = results.get("config") or {}
    cards = [
        ("Documents", _fmt_int(corpus.get("documents"))),
        ("Chunks", _fmt_int(corpus.get("chunks"))),
        ("Queries", _fmt_int(evalset.get("queries"))),
        ("Models", f"{successful}/{total}"),
        ("Top-k", _fmt_int(config.get("top_k"))),
        ("Seed", _fmt_int(_dig(results, "run", "seed"))),
    ]
    return "".join(
        f'<div class="stat"><span class="stat-value">{_e(value)}</span>'
        f'<span class="stat-label">{_e(label)}</span></div>'
        for label, value in cards
    )


def _environment_html(results: Mapping[str, Any]) -> str:
    environment = _dig(results, "run", "environment", default={}) or {}
    packages = environment.get("packages") or {}
    rows = [
        ("Python", environment.get("python")),
        ("Platform", environment.get("platform")),
        ("CPU cores", environment.get("cpu_count")),
        ("System RAM", f"{environment.get('total_ram_gb')} GB" if environment.get("total_ram_gb") else None),
        ("NumPy", packages.get("numpy")),
        ("FAISS", packages.get("faiss-cpu")),
        ("sentence-transformers", packages.get("sentence-transformers")),
        ("PyTorch", packages.get("torch")),
    ]
    body = "".join(
        f"<tr><th>{_e(label)}</th><td>{_e(value) if value is not None else 'n/a'}</td></tr>"
        for label, value in rows
    )
    return f'<div class="table-wrap"><table class="compact kv"><tbody>{body}</tbody></table></div>'


def _queried_docs_note(results: Mapping[str, Any]) -> str:
    corpus = _dig(results, "run", "corpus", default={}) or {}
    skipped = corpus.get("skipped_files") or []
    if not skipped:
        return '<p class="muted">No files were skipped during ingestion.</p>'
    items = "".join(f"<li><code>{_e(item.get('path'))}</code> — {_e(item.get('reason'))}</li>" for item in skipped)
    return f'<details class="inline-details"><summary>{len(skipped)} skipped file(s)</summary><ul>{items}</ul></details>'


def render_html(
    results: Mapping[str, Any],
    *,
    recommendation: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """Render the complete standalone HTML report."""

    config = results.get("config") or {}
    run = results.get("run") or {}
    models = results.get("models") or []
    top_k = int(config.get("top_k") or run.get("top_k") or 5)
    successful = [model for model in models if model.get("status") == "ok"]
    report_title = title or _dig(config, "report", "title", default="embedbench report")
    recommendation = recommendation or results.get("recommendation") or {}
    generated_at = results.get("generated_at") or utc_now_iso()
    version = _dig(results, "tool", "version", default="")
    total_models = len(models)
    balanced_weights = recommendation.get("weights") or BALANCED_WEIGHTS

    body = [
        '<header class="hero">',
        '<div><p class="eyebrow">embedbench · local retrieval evaluation</p>',
        f"<h1>{_e(report_title)}</h1>",
        f'<p class="subtitle">Comparing {total_models} embedding model(s) on your own documents. '
        f"Metrics reported at k={top_k}. Generated {_e(generated_at)} with embedbench {_e(version)}.</p></div>",
        '<div class="hero-mark" aria-hidden="true">EB</div>',
        "</header>",
        f'<section class="stat-grid">{_stat_cards(results, len(successful), total_models)}</section>',
        '<section class="panel rec-panel"><h2>Best model for your priority</h2>',
        _recommendation_html(recommendation, models, top_k),
        "</section>",
        '<section class="chart-layout">',
        '<article class="panel"><h2>Accuracy vs latency</h2>'
        '<div id="chart-scatter" class="chart"></div>'
        '<p class="caption">Up and to the left is better. p95 includes the first (cold) query.</p></article>',
        '<article class="panel"><h2>Retrieval metrics</h2>'
        '<div id="chart-metrics" class="chart"></div>'
        '<div id="legend-metrics" class="legend"></div>'
        '<p class="caption">Document-level metrics with duplicate chunks from the same document removed.</p></article>',
        '<article class="panel"><h2>Query latency by percentile</h2>'
        '<div id="chart-latency" class="chart"></div>'
        '<div id="legend-latency" class="legend"></div>'
        '<p class="caption">Query embedding plus vector search, measured per query.</p></article>',
        '<article class="panel"><h2>Estimated token cost</h2>'
        '<div id="chart-cost" class="chart"></div>'
        f'<p class="caption">Corpus + query tokens × ${_fmt(config.get("report", {}).get("cost_per_1m_tokens"), 4)} per 1M tokens '
        "(configurable; not an invoice).</p></article>",
        "</section>",
        '<section class="panel"><h2>Accuracy</h2>',
        _accuracy_table_html(models, top_k),
        "</section>",
        '<section class="panel"><h2>Cost, resources and throughput</h2>',
        _resource_table_html(models),
        '<p class="caption">* Ollama token counts are estimated because the REST API does not expose '
        "the model tokenizer. Process RAM is the benchmark process peak; Model MB and VRAM for Ollama "
        "come from /api/ps resident sizes. HF VRAM is PyTorch peak allocated memory.</p>",
        "</section>",
        '<section class="panel"><h2>Cold start vs warm timing</h2>',
        _timing_table_html(models),
        '<p class="caption">Warm values exclude the first batch/query. Cold embed is the first '
        "uncached embedding batch; it is n/a when every chunk came from cache. Model load measures "
        "backend initialization.</p>",
        "</section>",
        '<section class="panel"><h2>How to read this report</h2>',
        '<div class="notes">',
        "<p><strong>Accuracy.</strong> Recall@k is the fraction of relevant documents retrieved. "
        f"Precision@k uses k as its denominator. MRR@{top_k} uses the first relevant document. "
        "nDCG@k uses binary relevance and a log2 discount. Hit@k is 1 when any relevant document "
        "appears in the top k. Duplicate chunks from one document count once, at their first rank. "
        "A relevant document with many chunks can therefore lower precision@k.</p>",
        "<p><strong>Latency.</strong> p50/p95/p99 cover all queries. The first query is cold and is "
        "reported separately; indexing time includes cache lookup, embedding and index construction.</p>",
        "<p><strong>Balanced score.</strong> Accuracy, speed and cost are min–max normalized across "
        f"the models in this run, then weighted {int(balanced_weights.get('accuracy', 0.5) * 100)}/"
        f"{int(balanced_weights.get('speed', 0.3) * 100)}/"
        f"{int(balanced_weights.get('cost', 0.2) * 100)}. It is a decision aid, not an absolute score.</p>",
        "<p><strong>Cost.</strong> Token counts come from each model's own tokenizer where available. "
        "Cost is a hosted-API equivalent at the price configured for this run, not what the local model costs.</p>",
        "</div></section>",
        _per_query_html(models, top_k),
        _failures_html(models),
        '<section class="panel"><h2>Environment and configuration</h2>',
        _environment_html(results),
        _queried_docs_note(results),
        '<details class="inline-details"><summary>Full config snapshot</summary>'
        f"<pre>{_e(json.dumps(config, indent=2, ensure_ascii=False))}</pre></details>",
        "</section>",
        '<footer>Generated locally by embedbench. Embedding caches are keyed by model and chunk hash; '
        "a warm cache produces different indexing timings. Use the same seed and note your hardware "
        "when comparing runs.</footer>",
    ]
    body_html = "".join(body)
    data_json = json.dumps(
        results, indent=2, ensure_ascii=False, default=json_default, allow_nan=False
    )
    # Prevent query text containing </script> from breaking out of the data block.
    data_json = data_json.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    # Split on the original template tokens once so user text that happens to
    # contain a token cannot be substituted again.
    replacements = {
        "__TITLE__": _e(str(report_title)),
        "__BODY__": body_html,
        "__DATA__": data_json,
    }
    parts = _TEMPLATE_TOKEN_RE.split(_HTML_TEMPLATE)
    return "".join(replacements.get(part, part) for part in parts)


def write_html_report(
    results: Mapping[str, Any],
    path: Path,
    *,
    priority: str | None = None,
    title: str | None = None,
) -> Path:
    """Write a HTML report, optionally overriding priority and title."""

    config = results.get("config") or {}
    top_k = int(config.get("top_k") or _dig(results, "run", "top_k", default=5))
    effective_priority = priority or _dig(
        results, "recommendation", "priority", default=_dig(config, "report", "priority", default="balanced")
    )
    if effective_priority not in PRIORITIES:
        raise ReportError(f"Unknown priority {effective_priority!r}; use one of {PRIORITIES}")
    recommendation = recommend_models(
        results.get("models", []), effective_priority, top_k=top_k
    )
    payload = dict(results)
    payload["recommendation"] = recommendation
    html_text = render_html(payload, recommendation=recommendation, title=title)
    path = Path(path)
    atomic_write_text(path, html_text)
    logger.info("HTML report written to %s", path)
    return path


_TEMPLATE_TOKEN_RE = re.compile(r"(__TITLE__|__BODY__|__DATA__)")

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f4f6fb;
  --surface: #ffffff;
  --surface-2: #f8f9fd;
  --text: #182033;
  --muted: #667085;
  --border: #e1e6f0;
  --grid: #e7ebf3;
  --axis: #c5cedd;
  --accent: #335cff;
  --accent-soft: #e9eeff;
  --good: #13795b;
  --warn: #9a5b00;
  --bad: #b42318;
  --shadow: 0 10px 30px rgba(24, 32, 51, 0.06);
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1119;
    --surface: #141b27;
    --surface-2: #101722;
    --text: #e7ecf5;
    --muted: #98a5ba;
    --border: #263244;
    --grid: #222d3d;
    --axis: #3a4a61;
    --accent: #8da8ff;
    --accent-soft: #1c2b52;
    --good: #5ed3ac;
    --bad: #ff8f84;
    --shadow: none;
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; padding: 28px 22px 60px;
  background: var(--bg); color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  line-height: 1.55;
}
.hero {
  max-width: 1240px; margin: 0 auto 20px; display: flex; justify-content: space-between;
  align-items: flex-start; gap: 24px;
}
.eyebrow { margin: 0 0 8px; text-transform: uppercase; letter-spacing: .14em; font-size: 11px;
  font-weight: 700; color: var(--accent); }
h1 { margin: 0; font-size: clamp(28px, 4vw, 48px); line-height: 1.1; letter-spacing: -0.03em; }
h2 { margin: 0 0 16px; font-size: 18px; letter-spacing: -0.01em; }
h3 { margin: 12px 0 8px; font-size: 17px; overflow-wrap: anywhere; }
.subtitle { color: var(--muted); max-width: 780px; margin: 12px 0 0; }
.hero-mark {
  flex: 0 0 auto; width: 62px; height: 62px; border-radius: 18px; display: grid;
  place-items: center; font-weight: 800; letter-spacing: -0.06em; color: #fff;
  background: linear-gradient(135deg, #335cff, #7b4dff); box-shadow: var(--shadow);
}
.stat-grid { max-width: 1240px; margin: 0 auto 20px; display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 16px; box-shadow: var(--shadow); }
.stat-value { display: block; font-size: 23px; font-weight: 750; letter-spacing: -0.03em; }
.stat-label { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
  padding: 22px; margin: 0 auto 18px; max-width: 1240px; box-shadow: var(--shadow); }
.rec-panel { padding: 24px; }
.chart-layout { max-width: 1240px; margin: 0 auto; display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.chart-layout .panel { margin: 0; min-width: 0; }
.selected-callout { border: 1px solid var(--accent); background: var(--accent-soft);
  border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }
.selected-label { color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: .12em; font-weight: 700; }
.selected-model { font-size: 24px; font-weight: 780; letter-spacing: -0.02em; margin: 4px 0 8px;
  overflow-wrap: anywhere; }
.selected-callout p { margin: 0; }
.priority-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.priority-card { border: 1px solid var(--border); border-radius: 14px; padding: 16px;
  background: var(--surface-2); }
.priority-card.chosen { border-color: var(--accent); background: var(--surface); }
.priority-card h3 { font-size: 15px; }
.priority-card p { color: var(--muted); font-size: 13px; margin: 8px 0 12px; }
.priority-top { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
.tag { font-size: 10px; text-transform: uppercase; letter-spacing: .12em; font-weight: 750;
  color: var(--accent); }
.chosen-badge { font-size: 10px; font-weight: 700; color: var(--good);
  border: 1px solid var(--good); border-radius: 999px; padding: 2px 8px; }
.mini-stats { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: var(--muted); }
.mini-stats b { color: var(--text); font-variant-numeric: tabular-nums; }
.table-wrap { width: 100%; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border);
  vertical-align: top; }
th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  font-weight: 700; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: var(--surface-2); }
table.compact { font-size: 12px; }
table.kv { width: auto; min-width: 360px; }
table.kv th { width: 190px; }
code, pre { font-family: var(--mono); font-size: 12px; }
code { background: var(--surface-2); border: 1px solid var(--border); border-radius: 5px;
  padding: 1px 4px; overflow-wrap: anywhere; }
pre { background: var(--surface-2); border: 1px solid var(--border); padding: 14px;
  border-radius: 12px; overflow-x: auto; max-height: 520px; }
.chart { width: 100%; min-height: 260px; }
.chart-svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart-text { fill: var(--text); font-size: 12px; font-family: inherit; }
.chart-muted { fill: var(--muted); font-size: 11px; font-family: inherit; }
.chart-value { fill: var(--muted); font-size: 9px; font-family: inherit; }
.chart-gridline { stroke: var(--grid); stroke-width: 1; }
.chart-axis { stroke: var(--axis); stroke-width: 1.1; }
.chart-point { stroke: var(--surface); stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; color: var(--muted);
  font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.caption { color: var(--muted); font-size: 12px; margin: 10px 0 0; }
.muted { color: var(--muted); }
.notes p { margin: 0 0 10px; color: var(--muted); font-size: 13px; }
.notes strong { color: var(--text); }
.callout { border-radius: 12px; padding: 14px 16px; border: 1px solid var(--border); }
.callout.warn { color: var(--warn); background: color-mix(in srgb, var(--warn) 8%, var(--surface)); }
.failures { list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }
.failures li { border-left: 3px solid var(--bad); padding-left: 12px; }
.failures p { color: var(--muted); margin: 3px 0 0; overflow-wrap: anywhere; }
.query-details { padding: 0; overflow: hidden; }
.query-details summary { cursor: pointer; padding: 18px 22px; font-weight: 650; }
.query-details .table-wrap { padding: 0 22px 18px; }
.inline-details { margin-top: 14px; }
.inline-details summary { cursor: pointer; color: var(--accent); font-weight: 600; }
.inline-details ul { color: var(--muted); font-size: 13px; }
footer { max-width: 1240px; margin: 24px auto 0; color: var(--muted); font-size: 12px; }
@media (max-width: 980px) {
  .stat-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .chart-layout { grid-template-columns: 1fr; }
  .priority-grid { grid-template-columns: 1fr; }
}
@media (max-width: 620px) {
  body { padding: 18px 12px 40px; }
  .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hero-mark { display: none; }
  .panel { padding: 16px; border-radius: 14px; }
}
</style>
</head>
<body>
__BODY__
<script>
const DATA = __DATA__;
(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const COLORS = ["#4f7cff", "#ef6c4d", "#19a974", "#b34fd6", "#e0a400", "#2aa7c8", "#8a63ff", "#d94f70"];
  const LATENCY_COLORS = ["#93a4c8", "#4f7cff", "#1d3f9e"];
  const models = (DATA.models || []).filter((model) => model.status === "ok");
  const topK = (DATA.config && DATA.config.top_k) || (DATA.run && DATA.run.top_k) || 5;

  const esc = (value) => String(value === null || value === undefined ? "" : value);
  const fmt = (value, digits = 3) => {
    const number = Number(value);
    return value === null || value === undefined || Number.isNaN(number) ? "n/a" : number.toFixed(digits);
  };
  const shortSpec = (value, limit = 26) => {
    const text = esc(value);
    return text.length > limit ? text.slice(0, limit - 1) + "…" : text;
  };
  const colorFor = (index) => COLORS[index % COLORS.length];

  function makeSvg(id, width, height, label) {
    const host = document.getElementById(id);
    if (!host) return null;
    host.textContent = "";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("class", "chart-svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label);
    host.appendChild(svg);
    return svg;
  }

  function node(svg, tag, attrs, text) {
    const element = document.createElementNS(NS, tag);
    if (attrs) {
      Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    }
    if (text !== undefined && text !== null) element.textContent = String(text);
    svg.appendChild(element);
    return element;
  }

  function text(svg, x, y, value, className, anchor) {
    return node(svg, "text", {
      x, y, class: className || "chart-muted", "text-anchor": anchor || "start",
    }, value);
  }

  function line(svg, x1, y1, x2, y2, className) {
    return node(svg, "line", { x1, y1, x2, y2, class: className || "chart-gridline" });
  }

  function emptyChart(id, message) {
    const svg = makeSvg(id, 900, 320, message);
    if (svg) text(svg, 450, 160, message, "chart-muted", "middle");
  }

  function legend(id, items) {
    const host = document.getElementById(id);
    if (!host) return;
    host.textContent = "";
    items.forEach((item) => {
      const span = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.background = item.color;
      span.appendChild(swatch);
      span.appendChild(document.createTextNode(item.label));
      host.appendChild(span);
    });
  }

  function renderScatter() {
    if (!models.length) { emptyChart("chart-scatter", "No successful models to plot."); return; }
    const width = 900, height = 390, margin = { left: 74, right: 34, top: 24, bottom: 62 };
    const svg = makeSvg("chart-scatter", width, height, "Accuracy versus p95 query latency");
    if (!svg) return;
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const points = models.map((model, index) => ({
      model, index,
      x: Number(model.latency_ms && model.latency_ms.p95_ms),
      y: Number(model.accuracy && model.accuracy.ndcg),
      color: colorFor(index),
    })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (!points.length) {
      text(svg, width / 2, height / 2, "Latency or accuracy data is unavailable.", "chart-muted", "middle");
      return;
    }
    const maxX = Math.max(1, ...points.map((point) => point.x)) * 1.18;
    const xScale = (value) => margin.left + (value / maxX) * plotW;
    const yScale = (value) => margin.top + plotH - Math.max(0, Math.min(1, value)) * plotH;
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = tick / 4;
      const y = yScale(value);
      line(svg, margin.left, y, margin.left + plotW, y);
      text(svg, margin.left - 10, y + 4, value.toFixed(2), "chart-muted", "end");
    }
    for (let tick = 0; tick <= 5; tick += 1) {
      const value = (maxX * tick) / 5;
      const x = xScale(value);
      line(svg, x, margin.top, x, margin.top + plotH);
      text(svg, x, margin.top + plotH + 20, value >= 10 ? value.toFixed(0) : value.toFixed(1), "chart-muted", "middle");
    }
    line(svg, margin.left, margin.top, margin.left, margin.top + plotH, "chart-axis");
    line(svg, margin.left, margin.top + plotH, margin.left + plotW, margin.top + plotH, "chart-axis");
    text(svg, margin.left + plotW / 2, height - 14, "p95 query latency (ms)", "chart-text", "middle");
    node(svg, "text", {
      x: 16, y: margin.top + plotH / 2, class: "chart-text", "text-anchor": "middle",
      transform: "rotate(-90 16 " + (margin.top + plotH / 2) + ")",
    }, "nDCG@" + topK);
    points.forEach((point) => {
      const x = xScale(point.x), y = yScale(point.y);
      const circle = node(svg, "circle", {
        cx: x, cy: y, r: 7, fill: point.color, class: "chart-point",
      });
      const title = document.createElementNS(NS, "title");
      title.textContent = point.model.spec + " — nDCG@" + topK + " " + fmt(point.y, 3) + ", p95 " + fmt(point.x, 1) + " ms";
      circle.appendChild(title);
      const anchor = x > width - 220 ? "end" : "start";
      text(svg, x + (anchor === "end" ? -11 : 11), y - 10, shortSpec(point.model.spec, 22), "chart-muted", anchor);
    });
  }

  function renderMetrics() {
    if (!models.length) { emptyChart("chart-metrics", "No successful models to plot."); return; }
    const width = 900, height = 380, margin = { left: 62, right: 24, top: 22, bottom: 54 };
    const svg = makeSvg("chart-metrics", width, height, "Retrieval metrics by model");
    if (!svg) return;
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const metrics = [["recall", "Recall"], ["precision", "Precision"], ["mrr", "MRR"], ["ndcg", "nDCG"], ["hit_rate", "Hit"]];
    const groupW = plotW / metrics.length;
    const barW = Math.min(42, Math.max(5, (groupW - 16) / models.length));
    const yScale = (value) => margin.top + plotH - Math.max(0, Math.min(1, value)) * plotH;
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = tick / 4, y = yScale(value);
      line(svg, margin.left, y, margin.left + plotW, y);
      text(svg, margin.left - 9, y + 4, value.toFixed(2), "chart-muted", "end");
    }
    metrics.forEach(([key, label], metricIndex) => {
      const groupX = margin.left + metricIndex * groupW;
      const startX = groupX + (groupW - barW * models.length) / 2;
      models.forEach((model, modelIndex) => {
        const accuracy = model.accuracy || {};
        const value = Number(accuracy[key]);
        if (!Number.isFinite(value)) return;
        const x = startX + modelIndex * barW;
        const y = yScale(value);
        const rect = node(svg, "rect", {
          x: x + 1, y, width: Math.max(2, barW - 2), height: Math.max(0, margin.top + plotH - y),
          rx: 2, fill: colorFor(modelIndex), opacity: "0.9",
        });
        const title = document.createElementNS(NS, "title");
        title.textContent = model.spec + " — " + label + " " + fmt(value, 4);
        rect.appendChild(title);
        if (barW >= 18) text(svg, x + barW / 2, y - 4, fmt(value, 2), "chart-value", "middle");
      });
      text(svg, groupX + groupW / 2, height - 24, label, "chart-text", "middle");
    });
    line(svg, margin.left, margin.top + plotH, margin.left + plotW, margin.top + plotH, "chart-axis");
    legend("legend-metrics", models.map((model, index) => ({ label: shortSpec(model.spec, 30), color: colorFor(index) })));
  }

  function renderLatency() {
    if (!models.length) { emptyChart("chart-latency", "No successful models to plot."); return; }
    const width = 900, height = 380, margin = { left: 62, right: 24, top: 22, bottom: 84 };
    const svg = makeSvg("chart-latency", width, height, "Query latency percentiles");
    if (!svg) return;
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const series = [["p50_ms", "p50"], ["p95_ms", "p95"], ["p99_ms", "p99"]];
    const values = [];
    models.forEach((model) => series.forEach(([key]) => {
      const value = Number(model.latency_ms && model.latency_ms[key]);
      if (Number.isFinite(value)) values.push(value);
    }));
    const maxValue = Math.max(1, ...values) * 1.15;
    const groupW = plotW / models.length;
    const barW = Math.min(36, Math.max(6, (groupW - 20) / 3));
    const yScale = (value) => margin.top + plotH - (value / maxValue) * plotH;
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = (maxValue * tick) / 4, y = yScale(value);
      line(svg, margin.left, y, margin.left + plotW, y);
      text(svg, margin.left - 9, y + 4, value.toFixed(value >= 10 ? 0 : 1), "chart-muted", "end");
    }
    models.forEach((model, modelIndex) => {
      const groupX = margin.left + modelIndex * groupW;
      const startX = groupX + (groupW - barW * series.length) / 2;
      series.forEach(([key, label], seriesIndex) => {
        const value = Number(model.latency_ms && model.latency_ms[key]);
        if (!Number.isFinite(value)) return;
        const x = startX + seriesIndex * barW;
        const y = yScale(value);
        const rect = node(svg, "rect", {
          x: x + 1, y, width: Math.max(2, barW - 2), height: Math.max(0, margin.top + plotH - y),
          rx: 2, fill: LATENCY_COLORS[seriesIndex], opacity: "0.95",
        });
        const title = document.createElementNS(NS, "title");
        title.textContent = shortSpec(model.spec, 40) + " — " + label + " " + fmt(value, 2) + " ms";
        rect.appendChild(title);
      });
      const labelX = groupX + groupW / 2;
      node(svg, "text", {
        x: labelX, y: margin.top + plotH + 18, class: "chart-muted", "text-anchor": "end",
        transform: "rotate(-22 " + labelX + " " + (margin.top + plotH + 18) + ")",
      }, shortSpec(model.spec, 18));
    });
    line(svg, margin.left, margin.top + plotH, margin.left + plotW, margin.top + plotH, "chart-axis");
    text(svg, 15, margin.top + plotH / 2, "milliseconds", "chart-text", "middle");
    legend("legend-latency", series.map(([, label], index) => ({ label, color: LATENCY_COLORS[index] })));
  }

  function renderCost() {
    if (!models.length) { emptyChart("chart-cost", "No successful models to plot."); return; }
    const width = 900, height = 350, margin = { left: 66, right: 24, top: 28, bottom: 80 };
    const svg = makeSvg("chart-cost", width, height, "Estimated token cost by model");
    if (!svg) return;
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const values = models.map((model) => Number((model.tokens && model.tokens.cost_usd) || 0));
    const maxValue = Math.max(0.000001, ...values) * 1.2;
    const groupW = plotW / models.length;
    const barW = Math.min(74, Math.max(8, groupW * 0.55));
    const yScale = (value) => margin.top + plotH - (value / maxValue) * plotH;
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = (maxValue * tick) / 4, y = yScale(value);
      line(svg, margin.left, y, margin.left + plotW, y);
      text(svg, margin.left - 9, y + 4, "$" + value.toFixed(5), "chart-muted", "end");
    }
    models.forEach((model, modelIndex) => {
      const value = values[modelIndex];
      const x = margin.left + modelIndex * groupW + (groupW - barW) / 2;
      const y = yScale(value);
      const rect = node(svg, "rect", {
        x, y, width: barW, height: Math.max(1, margin.top + plotH - y), rx: 3,
        fill: colorFor(modelIndex), opacity: "0.9",
      });
      const title = document.createElementNS(NS, "title");
      title.textContent = model.spec + " — $" + value.toFixed(6) + " equivalent";
      rect.appendChild(title);
      text(svg, x + barW / 2, y - 5, "$" + value.toFixed(4), "chart-value", "middle");
      const labelX = margin.left + modelIndex * groupW + groupW / 2;
      node(svg, "text", {
        x: labelX, y: margin.top + plotH + 18, class: "chart-muted", "text-anchor": "end",
        transform: "rotate(-22 " + labelX + " " + (margin.top + plotH + 18) + ")",
      }, shortSpec(model.spec, 18));
    });
    line(svg, margin.left, margin.top + plotH, margin.left + plotW, margin.top + plotH, "chart-axis");
    text(svg, 16, margin.top + plotH / 2, "USD equivalent", "chart-text", "middle");
  }

  renderScatter();
  renderMetrics();
  renderLatency();
  renderCost();
})();
</script>
</body>
</html>
"""
