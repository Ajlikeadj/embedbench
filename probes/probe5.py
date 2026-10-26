"""Ingestion, escaping, cache-integrity and auto-evalset probes."""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fake_backend import FakeEmbeddingBackend  # noqa: F401  registers "fake"

from embedbench.cli import app

FAILS: list[str] = []
runner = CliRunner()


def check(name: str, cond: bool, detail: str = "") -> None:
    try:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} {detail}")
    except UnicodeEncodeError:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} (detail not printable)")
    if not cond:
        FAILS.append(f"{name} {detail}")


# ---------------------------------------------------------------- ingestion
def probe_ingestion(tmp: Path) -> None:
    from embedbench.config import IngestionConfig
    from embedbench.ingest import load_documents

    docs = tmp / "docs"
    sub = docs / "guides"
    sub.mkdir(parents=True)

    (docs / "plain.txt").write_text("Plain text file with caf\u00e9 naïve content.\n", encoding="utf-8")
    # cp1252 encoded file with a smart quote / euro sign
    (docs / "legacy.txt").write_bytes("Costs \u20ac 5 – done.".encode("cp1252"))
    # UTF-16 with BOM
    (docs / "utf16.txt").write_bytes("UTF-16 encoded document body with enough words to matter.".encode("utf-16"))
    # binary with NUL bytes
    (docs / "binary.txt").write_bytes(b"hello\x00world\x00more")
    # oversized
    (docs / "huge.txt").write_text("x" * 5000, encoding="utf-8")
    # hidden file and hidden dir
    (docs / ".hidden.txt").write_text("hidden file that must be ignored\n", encoding="utf-8")
    (docs / ".hiddendir").mkdir()
    (docs / ".hiddendir" / "secret.txt").write_text("hidden dir content that must be ignored\n", encoding="utf-8")
    # skipped dir
    (docs / "node_modules").mkdir()
    (docs / "node_modules" / "dep.txt").write_text("dependency text that must be skipped\n", encoding="utf-8")
    # markdown with heading, nested doc
    (sub / "guide.md").write_text("# Install Guide\n\nRun the installer. Then configure the agent.\n", encoding="utf-8")
    # html with scripts/nav/footer
    (docs / "page.html").write_text(
        "<html><head><title>Page Title</title><style>.x{color:red}</style>"
        "<script>alert('nope')</script></head><body><nav>Nav junk</nav>"
        "<h1>Heading</h1><p>Useful body text about the gateway port.</p>"
        "<footer>Footer junk</footer><noscript>noscript junk</noscript>"
        "<svg><text>svg junk</text></svg><template>template junk</template></body></html>",
        encoding="utf-8",
    )
    # docx
    import docx

    document = docx.Document()
    document.add_heading("Report", 0)
    document.add_paragraph("First paragraph with details about retention.")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Plan"
    table.rows[0].cells[1].text = "Price"
    table.rows[1].cells[0].text = "Growth"
    table.rows[1].cells[1].text = "499"
    document.save(str(docs / "report.docx"))
    # pdf via pypdf
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # no text layer -> must be skipped
    with open(docs / "scan.pdf", "wb") as handle:
        writer.write(handle)

    config = IngestionConfig()
    result = load_documents(docs, config)
    ids = [doc.doc_id for doc in result.documents]
    skipped = {item.path: item.reason for item in result.skipped}

    check("utf8 + subdirs loaded", "plain.txt" in ids and "guides/guide.md" in ids, str(ids))
    check("cp1252 decoded", any(d.doc_id == "legacy.txt" for d in result.documents), str(ids))
    legacy = [d for d in result.documents if d.doc_id == "legacy.txt"]
    check("cp1252 euro sign", legacy and "\u20ac" in legacy[0].text, legacy[0].text if legacy else "")
    check("utf16 decoded", "utf16.txt" in ids, str(ids))
    check("binary skipped", "binary.txt" not in ids and "binary.txt" in skipped, str(skipped))
    check("binary reason", "binary" in skipped.get("binary.txt", ""), skipped.get("binary.txt", ""))
    check("hidden file ignored", ".hidden.txt" not in ids)
    check("hidden dir ignored", not any(".hiddendir" in i for i in ids))
    check("skip_dirs honoured", not any("node_modules" in i for i in ids))
    check("markdown title captured", any(d.doc_id == "guides/guide.md" and d.title == "Install Guide" for d in result.documents))
    check("md source type", any(d.doc_id == "guides/guide.md" and d.source_type == "markdown" for d in result.documents))
    check("docx loaded", "report.docx" in ids, str(ids))
    docx_doc = [d for d in result.documents if d.doc_id == "report.docx"]
    check("docx table flattened", docx_doc and "Plan | Price" in docx_doc[0].text, docx_doc[0].text if docx_doc else "")
    check("docx tables metadata", docx_doc and docx_doc[0].metadata.get("tables") == 1)
    check("blank pdf skipped", "scan.pdf" not in ids and "scan.pdf" in skipped, str(skipped))
    html_doc = [d for d in result.documents if d.doc_id == "page.html"]
    check("html loaded", bool(html_doc))
    if html_doc:
        text = html_doc[0].text
        check("html script removed", "alert" not in text, text)
        check("html nav removed", "Nav junk" not in text, text)
        check("html footer removed", "Footer junk" not in text, text)
        check("html body kept", "Useful body text" in text, text)
        check("html title", html_doc[0].title == "Page Title", str(html_doc[0].title))

    # oversized file with a lower limit
    small = IngestionConfig(max_file_bytes=1024)
    result2 = load_documents(docs, small)
    check("max_file_bytes enforced", "huge.txt" not in [d.doc_id for d in result2.documents])
    check("oversize reported", any(item.path == "huge.txt" for item in result2.skipped))

    # non-recursive discovery
    flat = IngestionConfig(recursive=False)
    result3 = load_documents(docs, flat)
    check("non-recursive skips subdirs", not any("/" in d.doc_id for d in result3.documents),
          str([d.doc_id for d in result3.documents]))

    # custom extensions, normalization
    custom = IngestionConfig(extensions=["TXT", "md", ".MD"])
    check("extensions normalized", custom.extensions == [".txt", ".md"], str(custom.extensions))
    result4 = load_documents(docs, IngestionConfig(extensions=["md"]))
    check("extension filter works", [d.doc_id for d in result4.documents] == ["guides/guide.md"],
          str([d.doc_id for d in result4.documents]))

    # empty directory
    empty = tmp / "empty"
    empty.mkdir()
    from embedbench.errors import IngestionError

    try:
        load_documents(empty, IngestionConfig())
        check("empty corpus raises", False)
    except IngestionError as exc:
        check("empty corpus raises", True, str(exc)[:80])


# ---------------------------------------------------------------- escaping
def probe_escaping(tmp: Path) -> None:
    import yaml

    from embedbench.report import render_html, write_html_report

    evil_query = '</script><script>alert("xss")</script>'
    evil_title = '<img src=x onerror=alert(1)> "quoted"'
    docs = tmp / "esc_docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "evil.md").write_text(
        "# Evil\n\nThis document talks about </script><script>alert('boom')</script> payloads. "
        "It has enough words to form several chunks for the benchmark run.\n",
        encoding="utf-8",
    )
    ev = tmp / "esc.jsonl"
    ev.write_text(
        json.dumps({"query_id": "q1", "query": evil_query, "relevant_doc_ids": ["evil.md"],
                    "metadata": {"note": "</script>meta"}}) + "\n",
        encoding="utf-8",
    )
    config = {
        "docs": str(docs),
        "models": ["fake:hash"],
        "top_k": 2,
        "output_dir": str(tmp / "esc_out"),
        "evalset": {"path": str(ev)},
        "chunking": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 40, "min_chunk_chars": 20},
        "backends": {"extra": {"fake": {"dimension": 32}}},
        "cache": {"enabled": False, "dir": str(tmp / "esc_cache")},
        "report": {"title": evil_title, "priority": "balanced"},
    }
    cfg_path = tmp / "esc.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    res = runner.invoke(app, ["run", "--config", str(cfg_path), "--no-progress"])
    check("escape run exit 0", res.exit_code == 0, res.output[-800:])
    html_path = tmp / "esc_out" / "report.html"
    if not html_path.is_file():
        check("escape report written", False)
        return
    html_text = html_path.read_text(encoding="utf-8")
    data_start = html_text.find("const DATA = ")
    markup = html_text[:data_start]

    # the embedded JSON data block must not contain a raw </script>
    start = data_start
    check("data block present", start >= 0)
    check("no script breakout in data block", html_text.count("</script>") == 1, str(html_text.count("</script>")))
    check("payload escaped in data", "</script><script>alert" not in html_text,
          str(html_text[html_text.find("xss") - 60: html_text.find("xss") + 40]))
    check("no raw img onerror in markup", "<img src=x onerror" not in markup)
    check("title escaped in head", "<img src=x" not in html_text.split("</head>")[0])
    check("escaped entities present", "&lt;/script&gt;" in html_text)
    # no external resources
    check("no external script src", "<script src=" not in html_text)
    check("no external stylesheet", "<link " not in html_text)
    check("no css import", "@import" not in html_text)

    # render_html direct with hostile data
    payload = {
        "config": {"top_k": 3, "report": {"title": "</style><script>alert(2)</script>"}},
        "run": {"corpus": {}, "evalset": {}, "seed": 1},
        "models": [{"spec": "x:y", "status": "error", "error": "</script>fail"}],
        "recommendation": {"selected": None, "by_priority": {}, "scores": [], "priority": "balanced"},
    }
    rendered = render_html(payload)
    check("failed-model render ok", "Failures" in rendered)
    check("rendered data escaped", rendered.count("</script>") == 1, str(rendered.count("</script>")))


# ---------------------------------------------------------------- cache integrity
def probe_cache_integrity(tmp: Path) -> None:
    import numpy as np

    from embedbench.cache import EmbeddingCache

    cache_dir = tmp / "integrity"
    with EmbeddingCache(cache_dir) as cache:
        namespace = "fake:hash:v1:q='':d=''"
        cache.put_many(namespace, ["a", "b"], np.ones((2, 8), dtype=np.float32))
        # poison the cache with a wrong-dimension vector for one entry
        import sqlite3

        conn = sqlite3.connect(str(cache.db_path))
        conn.execute(
            "UPDATE embeddings SET dimension = 4, vector = ? WHERE namespace = ? AND text_sha256 = ?",
            (np.ones(4, dtype=np.float32).tobytes(), namespace,
             __import__("embedbench.utils", fromlist=["sha256_hex"]).sha256_hex("a")),
        )
        conn.commit()
        conn.close()
    with EmbeddingCache(cache_dir) as cache2:
        got = cache2.get_many(namespace, ["a", "b"])
        check("corrupt entry still returned", got[0] is not None and got[0].shape[0] == 4, str(got[0]))
        check("valid entry intact", got[1] is not None and got[1].shape[0] == 8, str(got[1]))


def probe_dimension_mismatch_discarded(tmp: Path) -> None:
    """A cached vector with the wrong dimension must be re-embedded, not used."""
    import json as _json
    import yaml

    from embedbench.cache import EmbeddingCache
    from embedbench.utils import sha256_hex
    import sqlite3
    import numpy as np

    docs = tmp / "dim_docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "a.md").write_text(
        "# A\n\nAlpha content that is long enough to become at least one chunk in the corpus.\n",
        encoding="utf-8",
    )
    ev = tmp / "dim.jsonl"
    ev.write_text(json.dumps({"query_id": "q1", "query": "What is alpha?",
                              "relevant_doc_ids": ["a.md"]}) + "\n", encoding="utf-8")
    cache_dir = tmp / "dim_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # seed a wrong-dimension entry under the fake backend namespace
    conn = sqlite3.connect(str(cache_dir / "embeddings.sqlite3"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (namespace TEXT NOT NULL, text_sha256 TEXT NOT NULL,"
        " dimension INTEGER NOT NULL, vector BLOB NOT NULL, created_at TEXT NOT NULL,"
        " PRIMARY KEY (namespace, text_sha256))"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    conn.close()
    cache = EmbeddingCache(cache_dir, enabled=True)
    namespace = "fake:hash:v1:q='':d=''"
    text = "Alpha content that is long enough to become at least one chunk in the corpus."
    cache.put_many(namespace, [text], np.ones((1, 4), dtype=np.float32))
    cache.close()

    config = {
        "docs": str(docs), "models": ["fake:hash"], "top_k": 1,
        "output_dir": str(tmp / "dim_out"),
        "evalset": {"path": str(ev)},
        "chunking": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 40, "min_chunk_chars": 20},
        "backends": {"extra": {"fake": {"dimension": 16}}},
        "cache": {"enabled": True, "dir": str(cache_dir)},
    }
    cfg = tmp / "dim.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("dimension mismatch run exit 0", res.exit_code == 0, res.output[-600:])
    if (tmp / "dim_out" / "results.json").is_file():
        data = _json.loads((tmp / "dim_out" / "results.json").read_text(encoding="utf-8"))
        model = data["models"][0]
        check("mismatched vector re-embedded", model["timing"]["embedded_chunks"] >= 1, str(model["timing"]))
        check("index dimension correct", model["index"]["dimension"] == 16, str(model["index"]))
        check("model still ok", model["status"] == "ok", str(model.get("error")))


# ---------------------------------------------------------------- top_k / auto evalset
def probe_topk_larger_than_corpus(tmp: Path) -> None:
    import json as _json
    import yaml

    docs = tmp / "small_docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "one.md").write_text("# One\n\nOnly one document here with enough words for a chunk.\n", encoding="utf-8")
    ev = tmp / "small.jsonl"
    ev.write_text(json.dumps({"query_id": "q1", "query": "What is here?", "relevant_doc_ids": ["one.md"]}) + "\n",
                  encoding="utf-8")
    config = {
        "docs": str(docs), "models": ["fake:hash"], "top_k": 50,
        "output_dir": str(tmp / "small_out"),
        "evalset": {"path": str(ev)},
        "chunking": {"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 10, "min_chunk_chars": 10},
        "backends": {"extra": {"fake": {"dimension": 16}}},
        "cache": {"enabled": False, "dir": str(tmp / "small_cache")},
    }
    cfg = tmp / "small.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("top_k > corpus exit 0", res.exit_code == 0, res.output[-600:])
    if (tmp / "small_out" / "results.json").is_file():
        data = _json.loads((tmp / "small_out" / "results.json").read_text(encoding="utf-8"))
        acc = data["models"][0]["accuracy"]
        check("metrics bounded with tiny corpus", 0.0 <= acc["precision"] <= 1.0 and acc["recall"] == 1.0, str(acc))


def probe_auto_evalset(tmp: Path) -> None:
    """--evalset must win over --auto-evalset, and auto-generation writes the artifact."""
    import yaml

    docs = tmp / "auto_docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "a.md").write_text(
        "# A\n\nThe gateway listens on port 8443 and stores telemetry for thirty days.\n", encoding="utf-8"
    )
    (docs / "b.md").write_text(
        "# B\n\nBilling plans cost 499 dollars per month and allow fifty robots.\n", encoding="utf-8"
    )
    ev = tmp / "auto.jsonl"
    ev.write_text(json.dumps({"query_id": "q1", "query": "What port is used?", "relevant_doc_ids": ["a.md"]}) + "\n",
                  encoding="utf-8")
    base = {
        "docs": str(docs), "models": ["fake:hash"], "top_k": 2,
        "output_dir": str(tmp / "auto_out"),
        "evalset": {"auto_generate": True, "ollama_model": "llama3.2"},
        "chunking": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 40, "min_chunk_chars": 20},
        "backends": {"extra": {"fake": {"dimension": 16}}},
        "cache": {"enabled": False, "dir": str(tmp / "auto_cache")},
    }
    cfg = tmp / "auto.yaml"
    cfg.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    # explicit path must win over auto_generate
    res = runner.invoke(app, ["run", "--config", str(cfg), "--evalset", str(ev), "--no-progress"])
    check("explicit evalset wins over auto", res.exit_code == 0, res.output[-800:])
    out = tmp / "auto_out"
    check("no generated file when path given", not (out / "evalset.generated.jsonl").is_file())
    if (out / "results.json").is_file():
        import json as _json

        data = _json.loads((out / "results.json").read_text(encoding="utf-8"))
        check("source is jsonl", data["run"]["evalset"]["source"] == "jsonl", str(data["run"]["evalset"]["source"]))

    # auto without a path and no Ollama -> clear error, exit 1
    res2 = runner.invoke(app, ["run", "--config", str(cfg), "--no-auto-evalset", "--no-progress"])
    check("no evalset at all exits 1", res2.exit_code == 1, str(res2.exit_code))
    check("no evalset message", "No evaluation set configured" in res2.output, res2.output[-400:])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in (probe_ingestion, probe_escaping, probe_cache_integrity,
                   probe_dimension_mismatch_discarded, probe_topk_larger_than_corpus,
                   probe_auto_evalset):
            print(f"\n=== {fn.__name__} ===")
            target = tmp / fn.__name__
            target.mkdir(parents=True, exist_ok=True)
            try:
                fn(target)
            except Exception:
                traceback.print_exc()
                FAILS.append(f"{fn.__name__} crashed")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURES:")
        for item in FAILS:
            print(" -", item)
        sys.exit(1)
    print("ALL INGESTION/ESCAPE PROBES PASSED")
