"""Progress bars, multi-batch embedding and log-file probes (force a real terminal)."""
from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fake_backend import FakeEmbeddingBackend  # noqa: F401  registers "fake"

from embedbench.benchmark import run_benchmark
from embedbench.config import RunConfig

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    try:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} {detail}")
    except UnicodeEncodeError:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} (detail not printable)")
    if not cond:
        FAILS.append(f"{name} {detail}")


def make_corpus(tmp: Path, paragraphs: int = 60) -> Path:
    docs = tmp / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        body = "\n\n".join(
            f"## Section {index}-{i}\n\n"
            f"Document {index} paragraph {i} describes port {8000 + i} and buffer size "
            f"{100 * (i + 1)} bytes for service {index}. Operators restart it with ctl-{index}-{i}. "
            f"Retention is {30 + i} days and the limit is {500 * (i + 1)} events per batch. "
            * 3
            for i in range(paragraphs // 4)
        )
        (docs / f"doc{index}.md").write_text(f"# Doc {index}\n\n{body}\n", encoding="utf-8")
    return docs


def make_evalset(tmp: Path, n: int = 12) -> Path:
    ev = tmp / "evalset.jsonl"
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "query_id": f"q{i:04d}",
                    "query": f"What port does document {i % 4} paragraph {i} describe?",
                    "relevant_doc_ids": [f"doc{i % 4}.md"],
                }
            )
        )
    ev.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ev


def build_config(tmp: Path, top_k: int = 5) -> RunConfig:
    docs = make_corpus(tmp)
    ev = make_evalset(tmp)
    return RunConfig.model_validate(
        {
            "docs": str(docs),
            "models": ["fake:hash"],
            "top_k": top_k,
            "seed": 42,
            "output_dir": str(tmp / "out"),
            "chunking": {"strategy": "recursive", "chunk_size": 400, "chunk_overlap": 60, "min_chunk_chars": 40},
            "evalset": {"path": str(ev)},
            "backends": {"extra": {"fake": {"dimension": 64, "batch_size": 8}}},
            "cache": {"enabled": True, "dir": str(tmp / "cache")},
            "report": {"title": "Progress probe", "priority": "balanced", "cost_per_1m_tokens": 0.02},
        }
    )


def probe_progress(tmp: Path) -> None:
    config = build_config(tmp)
    console = Console(force_terminal=True, width=100, record=True, soft_wrap=False)
    run = run_benchmark(config, console=console, show_progress=True)
    out = console.export_text()
    check("progress run ok", run.results["run"]["successful_models"] == 1)
    model = run.results["models"][0]
    n_chunks = run.results["run"]["corpus"]["chunks"]
    check("corpus is multi-batch", n_chunks > 8, str(n_chunks))
    check("multiple batches embedded", model["timing"]["batches"] > 1, str(model["timing"]))
    check("cold embed measured", model["timing"]["cold_embed_ms"] is not None, str(model["timing"]))
    check("warm batch p50 measured", model["timing"]["warm_batch_p50_ms"] is not None, str(model["timing"]))
    check("all chunks embedded", model["timing"]["embedded_chunks"] == n_chunks, str(model["timing"]))
    check("throughput present", model["throughput"]["chunks_per_second"] > 0, str(model["throughput"]))
    check("cache miss first run", model["cache"]["misses"] == n_chunks, str(model["cache"]))

    # warm run with progress enabled
    console2 = Console(force_terminal=True, width=100, record=True)
    run2 = run_benchmark(config, console=console2, show_progress=True)
    m2 = run2.results["models"][0]
    check("warm cache full hit", m2["cache"]["hit_rate"] == 1.0, str(m2["cache"]))
    check("warm run no embedding", m2["timing"]["embedded_chunks"] == 0, str(m2["timing"]))
    check("warm run no cold embed", m2["timing"]["cold_embed_ms"] is None, str(m2["timing"]))


def probe_no_progress(tmp: Path) -> None:
    config = build_config(tmp)
    console = Console(width=88, record=True)
    run = run_benchmark(config, console=console, show_progress=False)
    check("no-progress run ok", run.results["run"]["successful_models"] == 1)
    out = console.export_text()
    check("summary printed", "embedbench results" in out, out[:300].replace("\n", " "))
    check("recommendation printed", "Selected:" in out, out[-600:].replace("\n", " "))
    check("accuracy table printed", "Retrieval accuracy" in out, out[:500].replace("\n", " "))
    check("resource table printed", "footprint" in out, out[:800].replace("\n", " "))


def probe_failed_model_reporting(tmp: Path) -> None:
    config = build_config(tmp)
    config.models = ["fake:hash", "fake:missing", "bogus"]
    config.output_dir = tmp / "out_fail"
    console = Console(width=90, record=True)
    run = run_benchmark(config, console=console, show_progress=False)
    res = run.results
    check("partial failure counts", res["run"]["successful_models"] == 1 and res["run"]["failed_models"] == 2,
          str((res["run"]["successful_models"], res["run"]["failed_models"])))
    out = console.export_text()
    check("failed panel printed", "Failed models" in out, out[-700:].replace("\n", " "))
    check("failure detail printed", "not installed" in out, out[-700:].replace("\n", " "))
    html = (tmp / "out_fail" / "report.html").read_text(encoding="utf-8")
    check("failures in html", "Failures" in html and "not installed" in html)
    check("report still renders charts", 'id="chart-scatter"' in html)
    csv_rows = (tmp / "out_fail" / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    check("csv has all models", len(csv_rows) == 4, str(len(csv_rows)))
    errors = [row.split(",")[4] for row in csv_rows[1:]]
    check("csv records errors", any(e for e in errors), str(errors))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in (probe_progress, probe_no_progress, probe_failed_model_reporting):
            print(f"\n=== {fn.__name__} ===")
            target = tmp / fn.__name__
            target.mkdir(parents=True, exist_ok=True)
            try:
                fn(target)
            except Exception:
                traceback.print_exc()
                FAILS.append(f"{fn.__name__} crashed")
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURES:")
        for item in FAILS:
            print(" -", item)
        sys.exit(1)
    print("ALL PROGRESS PROBES PASSED")
