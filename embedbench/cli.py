"""Typer command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

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
from embedbench.errors import ConfigError, EmbedbenchError
from embedbench.evalset import SyntheticEvalsetGenerator, save_evalset
from embedbench.ingest import load_documents
from embedbench.logging_utils import close_logging_handlers, get_logger, setup_logging
from embedbench.ollama_client import OllamaClient
from embedbench.report import (
    PRIORITIES,
    console_summary,
    load_results,
    recommend_models,
    write_html_report,
)
from embedbench.utils import truncate

logger = get_logger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Benchmark embedding models for local RAG retrieval on your own documents.",
)

ConfigOption = Annotated[
    Optional[Path],
    typer.Option("--config", "-c", help="YAML config file. Defaults to ./embedbench.yaml if present."),
]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logging.")]


def _console() -> Console:
    return Console()


def _load(path: Optional[Path], overrides: dict[str, Any]) -> RunConfig:
    config_path = Path(path) if path is not None else default_config_path()
    return load_config(config_path, overrides)


def _priority(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in PRIORITIES:
        raise ConfigError(
            f"Invalid priority {value!r}. Choose one of: {', '.join(PRIORITIES)}."
        )
    return normalized


def _run_overrides(
    *,
    docs: Optional[Path],
    models: Optional[str],
    top_k: Optional[int],
    evalset: Optional[Path],
    auto_evalset: Optional[bool],
    evalset_model: Optional[str],
    output_dir: Optional[Path],
    seed: Optional[int],
    priority: Optional[str],
    chunking: Optional[str],
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    batch_size: Optional[int],
    cost_per_1m: Optional[float],
    cache: Optional[bool],
    cache_dir: Optional[Path],
    ollama_host: Optional[str],
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
    docs: Annotated[Optional[Path], typer.Option("--docs", "-d", help="Folder of .txt/.md/.pdf/.docx/.html files.")] = None,
    models: Annotated[
        Optional[str],
        typer.Option("--models", "-m", help="Comma-separated backend:model specs, e.g. ollama:nomic-embed-text,hf:BAAI/bge-small-en-v1.5."),
    ] = None,
    top_k: Annotated[Optional[int], typer.Option("--top-k", min=1, help="Number of chunks to retrieve per query.")] = None,
    config_path: ConfigOption = None,
    evalset: Annotated[Optional[Path], typer.Option("--evalset", help="User JSONL eval set; overrides synthetic generation.")] = None,
    auto_evalset: Annotated[
        Optional[bool],
        typer.Option("--auto-evalset/--no-auto-evalset", help="Generate questions with Ollama when no eval set is given."),
    ] = None,
    evalset_model: Annotated[Optional[str], typer.Option("--evalset-model", help="Ollama LLM used for synthetic questions.")] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Directory for JSON/CSV/HTML artifacts.")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", min=0, help="Random seed for reproducibility.")] = None,
    priority: Annotated[Optional[str], typer.Option("--priority", help="Recommendation priority: accuracy, speed or balanced.")] = None,
    chunking: Annotated[Optional[str], typer.Option("--chunking", help="Chunking strategy: fixed, sentence or recursive.")] = None,
    chunk_size: Annotated[Optional[int], typer.Option("--chunk-size", min=100, help="Target chunk size in characters.")] = None,
    chunk_overlap: Annotated[Optional[int], typer.Option("--chunk-overlap", min=0, help="Chunk overlap in characters.")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", min=1, help="Embedding batch size for built-in backends.")] = None,
    cost_per_1m: Annotated[Optional[float], typer.Option("--cost-per-1m", min=0.0, help="Hosted API price per 1M tokens for cost-equivalent estimates.")] = None,
    cache: Annotated[Optional[bool], typer.Option("--cache/--no-cache", help="Enable or disable the embedding cache.")] = None,
    cache_dir: Annotated[Optional[Path], typer.Option("--cache-dir", help="Embedding cache directory.")] = None,
    ollama_host: Annotated[Optional[str], typer.Option("--ollama-host", help="Ollama base URL.")] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress", help="Show progress bars.")] = True,
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
    docs: Annotated[Optional[Path], typer.Option("--docs", "-d", help="Folder of documents to generate questions from.")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Destination JSONL file.")] = Path("evalset.jsonl"),
    config_path: ConfigOption = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Ollama LLM used to write questions.")] = None,
    questions_per_chunk: Annotated[Optional[int], typer.Option("--questions-per-chunk", min=1, max=5)] = None,
    max_chunks: Annotated[Optional[int], typer.Option("--max-chunks", min=1, help="Limit generation to a deterministic sample of chunks.")] = None,
    min_question_words: Annotated[Optional[int], typer.Option("--min-question-words", min=3)] = None,
    dedupe_threshold: Annotated[Optional[float], typer.Option("--dedupe-threshold", min=0.5, max=1.0)] = None,
    chunking: Annotated[Optional[str], typer.Option("--chunking", help="fixed, sentence or recursive.")] = None,
    chunk_size: Annotated[Optional[int], typer.Option("--chunk-size", min=100)] = None,
    chunk_overlap: Annotated[Optional[int], typer.Option("--chunk-overlap", min=0)] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", min=0)] = None,
    ollama_host: Annotated[Optional[str], typer.Option("--ollama-host", help="Ollama base URL.")] = None,
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
    results: Annotated[Path, typer.Option("--results", "-r", help="results.json or the run output directory.")] = Path("results/results.json"),
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="HTML output path (default: next to results).")] = None,
    priority: Annotated[Optional[str], typer.Option("--priority", help="Recompute recommendation for accuracy, speed or balanced.")] = None,
    title: Annotated[Optional[str], typer.Option("--title", help="Override the report title.")] = None,
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
            if output.exists() and output.is_dir():
                output = output / "report.html"
            elif output.suffix.lower() not in {".html", ".htm"}:
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
    ollama_host: Annotated[str, typer.Option("--ollama-host", help="Ollama base URL.")] = "http://localhost:11434",
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
                status = "yes" if any(
                    (entry.get("model") == item.model and entry.get("installed"))
                    for entry in snapshot["models"]
                ) else "no"
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
    console.print("[dim]* Recommended starting models. 'download' means fetched on first use.[/dim]")
    if snapshot["ollama_error"]:
        console.print(f"[yellow]Ollama unavailable:[/yellow] {escape(truncate(snapshot['ollama_error'], 300))}")
    else:
        missing = [
            entry["pull_command"]
            for entry in snapshot["models"]
            if entry.get("pull_command") and entry.get("installed") is False
        ]
        if missing:
            console.print("[dim]Missing Ollama models:[/dim] " + " | ".join(escape(cmd) for cmd in missing))
    console.print(f"[dim]Registered backends:[/dim] {escape(', '.join(backends))}")
    console.print(
        "[dim]Examples:[/dim] embedbench run --docs ./docs --models "
        "hf:BAAI/bge-small-en-v1.5,ollama:nomic-embed-text"
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
