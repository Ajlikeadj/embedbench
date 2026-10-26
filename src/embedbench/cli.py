"""Typer command-line interface.

Commands:

``run``              file-based pipeline with HTML/JSON/CSV artifacts
``bench``            dataset evaluator: quality, latency, memory, cost
``stress needle``    needle-in-a-haystack at 2k/8k/32k tokens
``stress chunk-size`` chunk-size sensitivity sweep
``stress noisy``     query robustness against typos and paraphrases
``generate-evalset`` synthetic questions from a local Ollama model
``report``           re-render reports from a saved run
``list-models``      curated local models and Ollama install status
``providers``        registered providers, extras, API keys and pricing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from embedbench import __version__
from embedbench.backends import available_backends
from embedbench.benchmark import run_benchmark
from embedbench.catalog import CATALOG, catalog_with_status
from embedbench.chunking import chunk_documents
from embedbench.config import RunConfig, default_config_path, load_config
from embedbench.datasets import (
    demo_dataset,
    load_corpus_dataset,
    load_documents_dataset,
    load_jsonl_dataset,
    load_queries,
)
from embedbench.errors import ConfigError, EmbedbenchError
from embedbench.evalset import SyntheticEvalsetGenerator, save_evalset
from embedbench.evaluator import PRIORITIES as EVAL_PRIORITIES, Evaluator
from embedbench.ingest import load_documents
from embedbench.logging_utils import close_logging_handlers, get_logger, setup_logging
from embedbench.models import pricing_catalog, provider_metadata
from embedbench.ollama_client import OllamaClient
from embedbench.report import (
    PRIORITIES,
    console_summary,
    load_results,
    recommend_models,
    write_html_report,
)
from embedbench.reporter import render_console, write_reports
from embedbench.stress import StressReport, StressRunner
from embedbench.utils import truncate

logger = get_logger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help=(
        "Benchmark embedding models for retrieval quality, latency, memory and cost — "
        "local (Ollama, sentence-transformers) or hosted (OpenAI, Cohere, Voyage)."
    ),
)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", help="YAML config file. Defaults to ./embedbench.yaml if present."
    ),
]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logging.")]


def _console() -> Console:
    return Console()


def _load(path: Path | None, overrides: dict[str, Any]) -> RunConfig:
    config_path = Path(path) if path is not None else default_config_path()
    return load_config(config_path, overrides)


def _priority(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in PRIORITIES:
        raise ConfigError(f"Invalid priority {value!r}. Choose one of: {', '.join(PRIORITIES)}.")
    return normalized


def _run_overrides(
    *,
    docs: Path | None,
    models: str | None,
    top_k: int | None,
    evalset: Path | None,
    auto_evalset: bool | None,
    evalset_model: str | None,
    output_dir: Path | None,
    seed: int | None,
    priority: str | None,
    chunking: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    batch_size: int | None,
    cost_per_1m: float | None,
    cache: bool | None,
    cache_dir: Path | None,
    ollama_host: str | None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if docs is not None:
        overrides["docs"] = str(docs)
    if models is not None:
        specs = [item.strip() for item in models.split(",") if item.strip()]
        if not specs:
            raise ConfigError("--models must contain at least one backend:model entry")
        overrides["models"] = specs
    if top_k is not None:
        overrides["top_k"] = top_k
    if evalset is not None:
        overrides["evalset.path"] = str(evalset)
        if auto_evalset:
            overrides["evalset.auto_generate"] = False
    elif auto_evalset is not None:
        overrides["evalset.auto_generate"] = auto_evalset
    if evalset_model is not None:
        overrides["evalset.ollama_model"] = evalset_model
    if output_dir is not None:
        overrides["output_dir"] = str(output_dir)
    if seed is not None:
        overrides["seed"] = seed
    if priority is not None:
        overrides["report.priority"] = _priority(priority)
    if chunking is not None:
        overrides["chunking.strategy"] = chunking.strip().lower()
    if chunk_size is not None:
        overrides["chunking.chunk_size"] = chunk_size
    if chunk_overlap is not None:
        overrides["chunking.chunk_overlap"] = chunk_overlap
    if batch_size is not None:
        overrides["backends.hf.batch_size"] = batch_size
        overrides["backends.ollama.batch_size"] = batch_size
    if cost_per_1m is not None:
        overrides["report.cost_per_1m_tokens"] = cost_per_1m
    if cache is not None:
        overrides["cache.enabled"] = cache
    if cache_dir is not None:
        overrides["cache.dir"] = str(cache_dir)
    if ollama_host is not None:
        overrides["backends.ollama.host"] = ollama_host
    return overrides


def _prepare_logging(config: RunConfig, verbose: bool) -> None:
    try:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Could not create output directory {config.output_dir}: {exc}") from exc
    setup_logging(
        verbose=verbose,
        log_file=Path(config.output_dir) / "run.log",
        force=True,
    )


def _fail(console: Console, exc: Exception, *, verbose: bool) -> None:
    message = str(exc) or type(exc).__name__
    console.print(f"[bold red]Error:[/bold red] {escape(message)}")
    if verbose:
        logger.debug("Command failed", exc_info=True)


@app.command("run")
def run_command(
    docs: Annotated[
        Path | None, typer.Option("--docs", "-d", help="Folder of .txt/.md/.pdf/.docx/.html files.")
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated backend:model specs, e.g. ollama:nomic-embed-text,hf:BAAI/bge-small-en-v1.5.",
        ),
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", min=1, help="Number of chunks to retrieve per query.")
    ] = None,
    config_path: ConfigOption = None,
    evalset: Annotated[
        Path | None,
        typer.Option("--evalset", help="User JSONL eval set; overrides synthetic generation."),
    ] = None,
    auto_evalset: Annotated[
        bool | None,
        typer.Option(
            "--auto-evalset/--no-auto-evalset",
            help="Generate questions with Ollama when no eval set is given.",
        ),
    ] = None,
    evalset_model: Annotated[
        str | None, typer.Option("--evalset-model", help="Ollama LLM used for synthetic questions.")
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory for JSON/CSV/HTML artifacts."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", min=0, help="Random seed for reproducibility.")
    ] = None,
    priority: Annotated[
        str | None,
        typer.Option("--priority", help="Recommendation priority: accuracy, speed or balanced."),
    ] = None,
    chunking: Annotated[
        str | None,
        typer.Option("--chunking", help="Chunking strategy: fixed, sentence or recursive."),
    ] = None,
    chunk_size: Annotated[
        int | None, typer.Option("--chunk-size", min=100, help="Target chunk size in characters.")
    ] = None,
    chunk_overlap: Annotated[
        int | None, typer.Option("--chunk-overlap", min=0, help="Chunk overlap in characters.")
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", min=1, help="Embedding batch size for built-in backends."),
    ] = None,
    cost_per_1m: Annotated[
        float | None,
        typer.Option(
            "--cost-per-1m",
            min=0.0,
            help="Hosted API price per 1M tokens for cost-equivalent estimates.",
        ),
    ] = None,
    cache: Annotated[
        bool | None,
        typer.Option("--cache/--no-cache", help="Enable or disable the embedding cache."),
    ] = None,
    cache_dir: Annotated[
        Path | None, typer.Option("--cache-dir", help="Embedding cache directory.")
    ] = None,
    ollama_host: Annotated[
        str | None, typer.Option("--ollama-host", help="Ollama base URL.")
    ] = None,
    progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show progress bars.")
    ] = True,
    verbose: VerboseOption = False,
) -> None:
    """Ingest documents, evaluate models and write a full report."""

    console = _console()
    try:
        overrides = _run_overrides(
            docs=docs,
            models=models,
            top_k=top_k,
            evalset=evalset,
            auto_evalset=auto_evalset,
            evalset_model=evalset_model,
            output_dir=output_dir,
            seed=seed,
            priority=priority,
            chunking=chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            cost_per_1m=cost_per_1m,
            cache=cache,
            cache_dir=cache_dir,
            ollama_host=ollama_host,
        )
        config = _load(config_path, overrides)
        if config.docs is None:
            raise ConfigError("No documents folder. Pass --docs PATH or set 'docs' in the config.")
        if not config.models:
            raise ConfigError("No models. Pass --models or set 'models' in the config.")
        tab = Path(config.docs)
        if not tab.exists() or not tab.is_dir():
            raise ConfigError(f"Documents folder not found: {tab}")
        if config.evalset.path is not None and not Path(config.evalset.path).is_file():
            raise ConfigError(f"Evaluation set not found: {config.evalset.path}")
        _prepare_logging(config, verbose)
        console.print(
            f"[bold]embedbench[/bold] {__version__} | docs=[cyan]{escape(str(config.docs))}[/cyan] | "
            f"{len(config.models)} model(s) | top-k={config.top_k} | seed={config.seed}"
        )
        benchmark = run_benchmark(config, console=console, show_progress=progress)
        successful = int(benchmark.results["run"]["successful_models"])
        failed = int(benchmark.results["run"]["failed_models"])
        if successful == 0:
            console.print(
                Panel(
                    "All models failed. Check the messages above, then see the log at "
                    f"{Path(config.output_dir) / 'run.log'}.",
                    title="No successful models",
                    border_style="red",
                )
            )
            raise typer.Exit(1)
        if failed:
            console.print(
                f"[yellow]{failed} model(s) failed; successful results were still saved.[/yellow]"
            )
    except EmbedbenchError as exc:
        _fail(console, exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # unexpected bug or third-party failure
        if verbose:
            raise
        _fail(console, exc, verbose=False)
        console.print("[dim]Re-run with --verbose for a traceback.[/dim]")
        raise typer.Exit(1) from exc
    finally:
        close_logging_handlers()


@app.command("generate-evalset")
def generate_evalset_command(
    docs: Annotated[
        Path | None,
        typer.Option("--docs", "-d", help="Folder of documents to generate questions from."),
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Destination JSONL file.")] = Path(
        "evalset.jsonl"
    ),
    config_path: ConfigOption = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Ollama LLM used to write questions.")
    ] = None,
    questions_per_chunk: Annotated[
        int | None, typer.Option("--questions-per-chunk", min=1, max=5)
    ] = None,
    max_chunks: Annotated[
        int | None,
        typer.Option(
            "--max-chunks", min=1, help="Limit generation to a deterministic sample of chunks."
        ),
    ] = None,
    min_question_words: Annotated[int | None, typer.Option("--min-question-words", min=3)] = None,
    dedupe_threshold: Annotated[
        float | None, typer.Option("--dedupe-threshold", min=0.5, max=1.0)
    ] = None,
    chunking: Annotated[
        str | None, typer.Option("--chunking", help="fixed, sentence or recursive.")
    ] = None,
    chunk_size: Annotated[int | None, typer.Option("--chunk-size", min=100)] = None,
    chunk_overlap: Annotated[int | None, typer.Option("--chunk-overlap", min=0)] = None,
    seed: Annotated[int | None, typer.Option("--seed", min=0)] = None,
    ollama_host: Annotated[
        str | None, typer.Option("--ollama-host", help="Ollama base URL.")
    ] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
    verbose: VerboseOption = False,
) -> None:
    """Generate a synthetic evaluation set with a local Ollama LLM."""

    console = _console()
    try:
        overrides: dict[str, Any] = {}
        if docs is not None:
            overrides["docs"] = str(docs)
        if model is not None:
            overrides["evalset.ollama_model"] = model
        if questions_per_chunk is not None:
            overrides["evalset.questions_per_chunk"] = questions_per_chunk
        if max_chunks is not None:
            overrides["evalset.max_chunks"] = max_chunks
        if min_question_words is not None:
            overrides["evalset.min_question_words"] = min_question_words
        if dedupe_threshold is not None:
            overrides["evalset.dedupe_threshold"] = dedupe_threshold
        if chunking is not None:
            overrides["chunking.strategy"] = chunking.strip().lower()
        if chunk_size is not None:
            overrides["chunking.chunk_size"] = chunk_size
        if chunk_overlap is not None:
            overrides["chunking.chunk_overlap"] = chunk_overlap
        if seed is not None:
            overrides["seed"] = seed
        if ollama_host is not None:
            overrides["backends.ollama.host"] = ollama_host

        config = _load(config_path, overrides)
        if config.docs is None:
            raise ConfigError("No documents folder. Pass --docs PATH or set 'docs' in the config.")
        if not Path(config.docs).is_dir():
            raise ConfigError(f"Documents folder not found: {config.docs}")
        setup_logging(verbose=verbose, force=True)
        ingest = load_documents(Path(config.docs), config.ingestion)
        chunks = chunk_documents(ingest.documents, config.chunking)
        console.print(
            f"[bold]embedbench[/bold] {__version__} | {len(ingest.documents)} docs | "
            f"{len(chunks)} chunks | model=[cyan]{escape(config.evalset.ollama_model)}[/cyan]"
        )
        client = OllamaClient(
            config.backends.ollama.host,
            timeout_s=config.evalset.request_timeout_s,
            max_retries=config.backends.ollama.max_retries,
            retry_backoff_s=config.backends.ollama.retry_backoff_s,
        )
        try:
            generator = SyntheticEvalsetGenerator(
                client,
                config.evalset,
                seed=config.seed,
                show_progress=progress,
                console=console,
            )
            examples, stats = generator.generate(chunks)
        finally:
            client.close()
        output = Path(output)
        if output.parent and not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
        save_evalset(examples, output)

        table = Table(title="Generation summary", box=None)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Chunks considered", str(stats.chunks_available))
        table.add_row("Chunks used", str(stats.chunks_used))
        table.add_row("Chunks without valid output", str(stats.chunks_failed))
        table.add_row("Accepted questions", str(stats.accepted))
        table.add_row("Invalid model responses", str(stats.invalid_responses))
        for reason, count in stats.rejections.items():
            if count:
                table.add_row(f"Rejected: {reason}", str(count))
        console.print(table)
        console.print(f"[green]Wrote {len(examples)} questions to[/green] {escape(str(output))}")
        console.print(
            "[dim]Next:[/dim] embedbench run --docs "
            f"{escape(str(config.docs))} --evalset {escape(str(output))} --models "
            "hf:BAAI/bge-small-en-v1.5"
        )
    except EmbedbenchError as exc:
        _fail(console, exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        if verbose:
            raise
        _fail(console, exc, verbose=False)
        console.print("[dim]Re-run with --verbose for a traceback.[/dim]")
        raise typer.Exit(1) from exc


@app.command("report")
def report_command(
    results: Annotated[
        Path, typer.Option("--results", "-r", help="results.json or the run output directory.")
    ] = Path("results/results.json"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="HTML output path (default: next to results)."),
    ] = None,
    priority: Annotated[
        str | None,
        typer.Option(
            "--priority", help="Recompute recommendation for accuracy, speed or balanced."
        ),
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="Override the report title.")] = None,
    verbose: VerboseOption = False,
) -> None:
    """Re-render HTML and console reports from a saved run."""

    console = _console()
    try:
        effective_priority = _priority(priority)
        payload = load_results(Path(results))
        config = payload.get("config") or {}
        top_k = int(config.get("top_k") or 5)
        chosen_priority = (
            effective_priority
            or (payload.get("recommendation") or {}).get("priority")
            or (config.get("report") or {}).get("priority")
            or "balanced"
        )
        if chosen_priority not in PRIORITIES:
            chosen_priority = "balanced"
        recommendation = recommend_models(payload.get("models", []), chosen_priority, top_k=top_k)
        rendered = dict(payload)
        rendered["recommendation"] = recommendation
        setup_logging(verbose=verbose, force=True)
        console_summary(rendered, console)

        if output is None:
            base = Path(results)
            directory = base if base.is_dir() else base.parent
            output = directory / "report.html"
        else:
            output = Path(output)
            if (output.exists() and output.is_dir()) or output.suffix.lower() not in {
                ".html",
                ".htm",
            }:
                output = output / "report.html"
        write_html_report(
            payload,
            output,
            priority=chosen_priority,
            title=title,
        )
        console.print(f"[green]HTML report written to[/green] {escape(str(output))}")
    except EmbedbenchError as exc:
        _fail(console, exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        if verbose:
            raise
        _fail(console, exc, verbose=False)
        console.print("[dim]Re-run with --verbose for a traceback.[/dim]")
        raise typer.Exit(1) from exc


@app.command("list-models")
def list_models_command(
    ollama_host: Annotated[
        str, typer.Option("--ollama-host", help="Ollama base URL.")
    ] = "http://localhost:11434",
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List curated embedding models and Ollama installation status."""

    console = _console()
    try:
        snapshot = catalog_with_status(ollama_host)
        backends = available_backends()
    except Exception as exc:
        _fail(console, exc, verbose=False)
        raise typer.Exit(1) from exc

    if as_json:
        snapshot["backends"] = backends
        typer.echo(json.dumps(snapshot, indent=2))
        return

    table = Table(title="Curated embedding models", header_style="bold cyan", box=None)
    table.add_column("Spec", style="bold")
    table.add_column("Dim", justify="right")
    table.add_column("Installed / cache", justify="center")
    table.add_column("Notes")
    for item in CATALOG:
        if item.backend == "ollama":
            if snapshot["ollama_available"]:
                status = (
                    "yes"
                    if any(
                        (entry.get("model") == item.model and entry.get("installed"))
                        for entry in snapshot["models"]
                    )
                    else "no"
                )
            else:
                status = "?"
        else:
            status = "download"
        table.add_row(
            escape(item.spec) + (" *" if item.recommended else ""),
            str(item.dimension) if item.dimension else "?",
            status,
            escape(item.notes),
        )
    console.print(table)
    console.print(
        "[dim]* Recommended starting models. 'download' means fetched on first use.[/dim]"
    )
    if snapshot["ollama_error"]:
        console.print(
            f"[yellow]Ollama unavailable:[/yellow] {escape(truncate(snapshot['ollama_error'], 300))}"
        )
    else:
        missing = [
            entry["pull_command"]
            for entry in snapshot["models"]
            if entry.get("pull_command") and entry.get("installed") is False
        ]
        if missing:
            console.print(
                "[dim]Missing Ollama models:[/dim] " + " | ".join(escape(cmd) for cmd in missing)
            )
    console.print(f"[dim]Registered backends:[/dim] {escape(', '.join(backends))}")
    console.print(
        "[dim]Examples:[/dim] embedbench run --docs ./docs --models "
        "hf:BAAI/bge-small-en-v1.5,ollama:nomic-embed-text"
    )


# ---------------------------------------------------------------------------
# Dataset evaluator, stress workloads and provider catalogue
# ---------------------------------------------------------------------------


def _parse_int_list(value: str, *, option: str, minimum: int = 1) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ConfigError(f"--{option} must be a comma-separated list of integers") from exc
    if not values:
        raise ConfigError(f"--{option} needs at least one value")
    if any(item < minimum for item in values):
        raise ConfigError(f"--{option} values must be >= {minimum}")
    return sorted(set(values))


def _parse_float_list(value: str, *, option: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ConfigError(f"--{option} must be a comma-separated list of numbers") from exc
    if not values:
        raise ConfigError(f"--{option} needs at least one value")
    if any(not 0.0 <= item <= 1.0 for item in values):
        raise ConfigError(f"--{option} values must be between 0 and 1")
    return sorted(set(values))


def _parse_string_list(value: str, *, option: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ConfigError(f"--{option} needs at least one value")
    return values


def _parse_specs(value: str) -> list[str]:
    specs = [item.strip() for item in value.split(",") if item.strip()]
    if not specs:
        raise ConfigError("--models must contain at least one provider:model entry")
    return specs


def _eval_priority(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in EVAL_PRIORITIES:
        raise ConfigError(
            f"Invalid priority {value!r}. Choose one of: {', '.join(EVAL_PRIORITIES)}."
        )
    return normalized


def _index_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"auto", "faiss", "numpy"}:
        raise ConfigError("--index must be one of: auto, faiss, numpy")
    return normalized


def _print_stress_report(
    console: Console,
    report: StressReport,
    *,
    title: str,
) -> None:
    table = Table(title=title, box=None, header_style="bold cyan")
    table.add_column("Model")
    table.add_column("Configuration")
    table.add_column("Docs", justify="right")
    table.add_column("Hit", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("nDCG", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Cost", justify="right")
    for cell in report.successful_cells():
        metrics = cell.get("metrics", {})
        config = cell.get("config", {})
        label = ", ".join(f"{key}={value}" for key, value in config.items())
        usd = cell.get("usd")
        cost = "free" if usd == 0 else ("n/a" if usd is None else f"${usd:.6f}")
        table.add_row(
            escape(truncate(cell["model"], 34)),
            escape(label),
            str(cell.get("documents", "n/a")),
            f"{float(metrics.get('hit_rate', 0.0)):.3f}",
            f"{float(metrics.get('mrr', 0.0)):.3f}",
            f"{float(metrics.get('ndcg', 0.0)):.3f}",
            f"{float(cell['p95_ms']):.1f}"
            if isinstance(cell.get("p95_ms"), (int, float))
            else "n/a",
            cost,
        )
    console.print(table)
    failures = [cell for cell in report.cells if cell.get("status") != "ok"]
    for cell in failures:
        config = cell.get("config", {})
        label = ", ".join(f"{key}={value}" for key, value in config.items())
        console.print(
            f"[red]{escape(cell['model'])} ({escape(label)}) failed:[/red] "
            f"{escape(str(cell.get('error')))}"
        )


@app.command("bench")
def bench_command(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated provider:model specs, e.g. ollama:nomic-embed-text,openai:text-embedding-3-small.",
        ),
    ] = "",
    docs: Annotated[
        Path | None, typer.Option("--docs", "-d", help="Folder of documents to index.")
    ] = None,
    evalset: Annotated[
        Path | None,
        typer.Option("--evalset", help="JSONL eval set with queries and relevant_doc_ids."),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset", help="JSONL dataset file (documents + queries) to evaluate directly."
        ),
    ] = None,
    demo: Annotated[
        bool,
        typer.Option(
            "--demo", help="Evaluate the built-in demo dataset; no documents or eval set needed."
        ),
    ] = False,
    k: Annotated[
        str, typer.Option("--k", help="Comma-separated cut-offs to evaluate, e.g. 1,5,10.")
    ] = "1,5,10",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for results.json/.md/.csv and plots."),
    ] = Path("results"),
    priority: Annotated[
        str,
        typer.Option(
            "--priority", help="Recommendation weighting: accuracy, speed, balanced or cost."
        ),
    ] = "balanced",
    index_backend: Annotated[
        str, typer.Option("--index", help="Vector index: auto, faiss or numpy.")
    ] = "auto",
    seed: Annotated[int, typer.Option("--seed", min=0, help="Random seed.")] = 42,
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", min=1, help="Override the embedding batch size.")
    ] = None,
    save_dataset: Annotated[
        Path | None,
        typer.Option("--save-dataset", help="Write the resolved dataset as JSONL for reuse."),
    ] = None,
    plugins: Annotated[
        str | None,
        typer.Option(
            "--plugins",
            help="Comma-separated dotted modules or .py files registering custom providers.",
        ),
    ] = None,
    plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots", help="Render PNG comparisons (needs embedbench[plots])."
        ),
    ] = True,
    config_path: ConfigOption = None,
    progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show per-model progress lines.")
    ] = True,
    verbose: VerboseOption = False,
) -> None:
    """Benchmark models on a dataset: quality, latency, memory and cost."""

    console = _console()
    try:
        specs = _parse_specs(models)
        cut_offs = _parse_int_list(k, option="k")
        chosen_priority = _eval_priority(priority)
        backend = _index_backend(index_backend)

        if demo:
            evaluated = demo_dataset()
        elif dataset is not None:
            evaluated = load_jsonl_dataset(dataset)
        else:
            if docs is None or evalset is None:
                raise ConfigError("Provide --docs and --evalset, --dataset FILE, or --demo.")
            if not Path(docs).is_dir():
                raise ConfigError(f"Documents folder not found: {docs}")
            if not Path(evalset).is_file():
                raise ConfigError(f"Evaluation set not found: {evalset}")
            config = _load(config_path, {})
            evaluated = load_corpus_dataset(
                docs,
                evalset,
                ingestion=config.ingestion,
                chunking=config.chunking,
            )
        setup_logging(verbose=verbose, force=True)
        console.print(
            f"[bold]embedbench[/bold] {__version__} | dataset=[cyan]{escape(evaluated.name)}[/cyan] | "
            f"{evaluated.num_documents} documents | {evaluated.num_queries} queries | "
            f"k={','.join(str(value) for value in cut_offs)}"
        )
        evaluator = Evaluator(
            specs,
            k_values=cut_offs,
            batch_size=batch_size,
            index_backend=backend,  # type: ignore[arg-type]
            show_progress=progress,
            console=console,
            seed=seed,
            plugins=_parse_string_list(plugins, option="plugins") if plugins else None,
        )
        result = evaluator.evaluate(evaluated)

        if save_dataset is not None:
            written = evaluated.save_jsonl(Path(save_dataset))
            console.print(f"[dim]Dataset written to[/dim] {escape(str(written))}")

        render_console(result, console, priority=chosen_priority)
        artifacts = write_reports(result, output_dir, priority=chosen_priority, plots=plots)
        console.print("\n[bold]Artifacts[/bold]")
        for name, path in artifacts.items():
            console.print(f"  {name}: {escape(str(path))}")

        if not result.successful:
            console.print(
                Panel(
                    "Every model failed. Check the errors above; the JSON report was still "
                    "written for inspection.",
                    title="No successful models",
                    border_style="red",
                )
            )
            raise typer.Exit(1)
        if result.failed:
            console.print(
                f"[yellow]{len(result.failed)} model(s) failed; successful results were saved.[/yellow]"
            )
    except EmbedbenchError as exc:
        _fail(console, exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        if verbose:
            raise
        _fail(console, exc, verbose=False)
        console.print("[dim]Re-run with --verbose for a traceback.[/dim]")
        raise typer.Exit(1) from exc
    finally:
        close_logging_handlers()


stress_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Stress workloads: needle-in-a-haystack, chunk-size sweeps, query noise.",
)
app.add_typer(stress_app, name="stress")


def _stress_models(value: str) -> list[str]:
    return _parse_specs(value)


@stress_app.command("needle")
def stress_needle_command(
    models: Annotated[
        str, typer.Option("--models", "-m", help="Comma-separated provider:model specs.")
    ],
    context_sizes: Annotated[
        str, typer.Option("--context-sizes", help="Haystack sizes in tokens, e.g. 2048,8192,32768.")
    ] = "2048,8192,32768",
    depths: Annotated[
        str,
        typer.Option(
            "--depths", help="Insertion depths as fractions of the context, e.g. 0,0.5,1."
        ),
    ] = "0,0.5,1",
    chunk_size: Annotated[
        int, typer.Option("--chunk-size", min=100, help="Chunk size in characters.")
    ] = 800,
    k: Annotated[str, typer.Option("--k", help="Comma-separated cut-offs.")] = "1,5,10",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/stress"),
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    batch_size: Annotated[int | None, typer.Option("--batch-size", min=1)] = None,
    index_backend: Annotated[str, typer.Option("--index")] = "auto",
    plugins: Annotated[
        str | None,
        typer.Option(
            "--plugins", help="Comma-separated modules or .py files registering custom providers."
        ),
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """Find facts hidden at increasing context lengths and depths."""

    console = _console()
    try:
        runner = StressRunner(
            _stress_models(models),
            k_values=_parse_int_list(k, option="k"),
            batch_size=batch_size,
            index_backend=_index_backend(index_backend),
            show_progress=False,
            console=console,
            seed=seed,
            plugins=_parse_string_list(plugins, option="plugins") if plugins else None,
        )
        setup_logging(verbose=verbose, force=True)
        console.print(
            f"[bold]Needle in a haystack[/bold] | contexts={context_sizes} | depths={depths}"
        )
        report = runner.needle_in_haystack(
            context_sizes=_parse_int_list(context_sizes, option="context-sizes", minimum=64),
            depths=_parse_float_list(depths, option="depths"),
            chunk_size=chunk_size,
            seed=seed,
        )
        _print_stress_report(console, report, title="Needle-in-a-haystack")
        artifacts = report.write(output_dir, stem="needle")
        for name, path in artifacts.items():
            console.print(f"  {name}: {escape(str(path))}")
        if not report.successful_cells():
            raise typer.Exit(1)
    except EmbedbenchError as exc:
        _fail(console, exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except typer.Exit:
        raise
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from exc
    finally:
        close_logging_handlers()


@stress_app.command("chunk-size")
def stress_chunk_size_command(
    models: Annotated[
        str, typer.Option("--models", "-m", help="Comma-separated provider:model specs.")
    ],
    docs: Annotated[Path, typer.Option("--docs", "-d", help="Folder of source documents.")],
    evalset: Annotated[Path, typer.Option("--evalset", help="JSONL eval set.")],
    chunk_sizes: Annotated[
        str, typer.Option("--chunk-sizes", help="Chunk sizes in tokens, e.g. 128,256,512,1024.")
    ] = "128,256,512,1024",
    k: Annotated[str, typer.Option("--k", help="Comma-separated cut-offs.")] = "1,5,10",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/stress"),
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    batch_size: Annotated[int | None, typer.Option("--batch-size", min=1)] = None,
    index_backend: Annotated[str, typer.Option("--index")] = "auto",
    plugins: Annotated[
        str | None,
        typer.Option(
            "--plugins", help="Comma-separated modules or .py files registering custom providers."
        ),
    ] = None,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Measure retrieval quality across chunk sizes."""

    console = _console()
    try:
        if not Path(docs).is_dir():
            raise ConfigError(f"Documents folder not found: {docs}")
        if not Path(evalset).is_file():
            raise ConfigError(f"Evaluation set not found: {evalset}")
        config = _load(config_path, {})
        documents = load_documents_dataset(docs, ingestion=config.ingestion)
        queries = load_queries(evalset)
        runner = StressRunner(
            _stress_models(models),
            k_values=_parse_int_list(k, option="k"),
            batch_size=batch_size,
            index_backend=_index_backend(index_backend),
            show_progress=False,
            console=console,
            seed=seed,
            plugins=_parse_string_list(plugins, option="plugins") if plugins else None,
        )
        setup_logging(verbose=verbose, force=True)
        console.print(
            f"[bold]Chunk-size sensitivity[/bold] | {len(documents)} documents | "
            f"{len(queries)} queries | sizes={chunk_sizes}"
        )
        report = runner.chunk_size_sensitivity(
            documents,
            queries,
            chunk_tokens=_parse_int_list(chunk_sizes, option="chunk-sizes", minimum=16),
        )
        _print_stress_report(console, report, title="Chunk-size sensitivity")
        artifacts = report.write(output_dir, stem="chunk-size")
        for name, path in artifacts.items():
            console.print(f"  {name}: {escape(str(path))}")
        if not report.successful_cells():
            raise typer.Exit(1)
    except EmbedbenchError as exc:
        _fail(console, exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except typer.Exit:
        raise
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from exc
    finally:
        close_logging_handlers()


@stress_app.command("noisy")
def stress_noisy_command(
    models: Annotated[
        str, typer.Option("--models", "-m", help="Comma-separated provider:model specs.")
    ],
    docs: Annotated[Path, typer.Option("--docs", "-d", help="Folder of source documents.")],
    evalset: Annotated[Path, typer.Option("--evalset", help="JSONL eval set.")],
    noise: Annotated[
        str,
        typer.Option(
            "--noise",
            help="Comma-separated degradations: typo, case, truncate, shuffle, paraphrase, noise.",
        ),
    ] = "typo,case,truncate,shuffle,paraphrase",
    k: Annotated[str, typer.Option("--k", help="Comma-separated cut-offs.")] = "1,5,10",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results/stress"),
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    batch_size: Annotated[int | None, typer.Option("--batch-size", min=1)] = None,
    index_backend: Annotated[str, typer.Option("--index")] = "auto",
    plugins: Annotated[
        str | None,
        typer.Option(
            "--plugins", help="Comma-separated modules or .py files registering custom providers."
        ),
    ] = None,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Measure quality retention when user queries are degraded."""

    console = _console()
    try:
        if not Path(docs).is_dir():
            raise ConfigError(f"Documents folder not found: {docs}")
        if not Path(evalset).is_file():
            raise ConfigError(f"Evaluation set not found: {evalset}")
        config = _load(config_path, {})
        evaluated = load_corpus_dataset(
            docs,
            evalset,
            ingestion=config.ingestion,
            chunking=config.chunking,
        )
        runner = StressRunner(
            _stress_models(models),
            k_values=_parse_int_list(k, option="k"),
            batch_size=batch_size,
            index_backend=_index_backend(index_backend),
            show_progress=False,
            console=console,
            seed=seed,
            plugins=_parse_string_list(plugins, option="plugins") if plugins else None,
        )
        setup_logging(verbose=verbose, force=True)
        console.print(f"[bold]Query robustness[/bold] | variants={noise}")
        report = runner.query_robustness(
            evaluated,
            kinds=_parse_string_list(noise, option="noise"),
            seed=seed,
        )
        _print_stress_report(console, report, title="Query robustness")
        artifacts = report.write(output_dir, stem="query-noise")
        for name, path in artifacts.items():
            console.print(f"  {name}: {escape(str(path))}")
        if not report.successful_cells():
            raise typer.Exit(1)
    except EmbedbenchError as exc:
        _fail(console, exc, verbose=verbose)
        raise typer.Exit(1) from exc
    except typer.Exit:
        raise
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from exc
    finally:
        close_logging_handlers()


@app.command("providers")
def providers_command(
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List registered providers, install extras, API keys and pricing."""

    console = _console()
    try:
        providers = provider_metadata()
        pricing = pricing_catalog()
    except Exception as exc:
        _fail(console, exc, verbose=False)
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(json.dumps({"providers": providers, "pricing": pricing}, indent=2))
        return

    table = Table(title="Embedding providers", box=None, header_style="bold cyan")
    table.add_column("Provider", style="bold")
    table.add_column("Install extra")
    table.add_column("API key env")
    table.add_column("Price / 1M")
    table.add_column("Description")
    for item in providers:
        extra = f'pip install "embedbench[{item["extra"]}]"' if item.get("extra") else "built in"
        price = item.get("price_per_1m_tokens")
        if price is None:
            price_text = (
                "per model"
                if any(entry["provider"] == item["provider"] for entry in pricing)
                else "n/a"
            )
        elif price == 0:
            price_text = "free"
        else:
            price_text = f"${price:.2f}"
        table.add_row(
            escape(str(item["provider"])),
            escape(extra),
            escape(str(item.get("api_key_env") or "-")),
            price_text,
            escape(str(item.get("description") or "")),
        )
    console.print(table)

    pricing_table = Table(title="Hosted model list prices", box=None, header_style="bold cyan")
    pricing_table.add_column("Model", style="bold")
    pricing_table.add_column("USD / 1M tokens", justify="right")
    for item in pricing:
        pricing_table.add_row(
            escape(f"{item['provider']}:{item['model']}"),
            f"${item['usd_per_1m_tokens']:.2f}",
        )
    console.print(pricing_table)
    console.print(
        "[dim]Prices are indicative list prices; override per model with "
        "--option price_per_1m_tokens=... or backends.extra in the config.[/dim]"
    )
    console.print(
        "[dim]Examples:[/dim] embedbench bench --demo --models ollama:nomic-embed-text | "
        "embedbench bench --docs ./docs --evalset evalset.jsonl --models "
        "openai:text-embedding-3-small"
    )


def main() -> None:
    """Console-script entry point."""

    try:
        app()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
