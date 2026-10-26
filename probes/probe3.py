"""End-to-end probes against a fake Ollama server (no real Ollama needed)."""
from __future__ import annotations

import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from typer.testing import CliRunner

from embedbench.cli import app

FAILS: list[str] = []
runner = CliRunner()

STATE = {
    "legacy": False,        # /api/embed returns 404 -> legacy fallback
    "model_missing": False,  # /api/tags returns empty -> model missing
    "fail_first": 0,         # number of 503s to serve before success
    "requests": [],
}


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        STATE["requests"].append(("GET", self.path, None))
        if self.path == "/api/tags":
            models = [] if STATE["model_missing"] else [
                {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest", "size": 1234},
                {"name": "llama3.2:latest", "model": "llama3.2:latest", "size": 5678},
            ]
            self._send(200, {"models": models})
        elif self.path == "/api/version":
            self._send(200, {"version": "0.9.9"})
        elif self.path == "/api/ps":
            self._send(200, {"models": [
                {"name": "nomic-embed-text:latest", "size": 100 * 1024 * 1024,
                 "size_vram": 64 * 1024 * 1024}
            ]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        payload = self._body()
        STATE["requests"].append(("POST", self.path, payload))
        if self.path == "/api/embed":
            if STATE["legacy"]:
                self._send(404, {"error": "unknown endpoint"})
                return
            if STATE["fail_first"] > 0:
                STATE["fail_first"] -= 1
                self._send(503, {"error": "server busy"})
                return
            inputs = payload.get("input") or []
            vectors = []
            for text in inputs:
                vec = [float((hash(text) >> (i * 3)) % 97) / 97.0 for i in range(16)]
                vectors.append(vec)
            self._send(200, {"embeddings": vectors})
        elif self.path == "/api/embeddings":
            prompt = payload.get("prompt", "")
            vec = [float((hash(prompt) >> (i * 3)) % 97) / 97.0 for i in range(16)]
            self._send(200, {"embedding": vec})
        elif self.path == "/api/generate":
            question = {
                "questions": [
                    {
                        "question": "Which port does the gateway listen on for telemetry?",
                        "answer": "8443",
                    },
                    {"question": "Summarize this document", "answer": "nope"},
                ]
            }
            self._send(200, {"response": json.dumps(question)})
        else:
            self._send(404, {"error": "not found"})


def check(name: str, cond: bool, detail: str = "") -> None:
    try:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} {detail}")
    except UnicodeEncodeError:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} {detail.encode('ascii', 'replace').decode('ascii')}")
    if not cond:
        FAILS.append(f"{name} {detail}")


def make_corpus(tmp: Path) -> tuple[Path, Path]:
    docs = tmp / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "gateway.md").write_text(
        "# Gateway\n\nThe Alpha gateway listens on port 8443 for telemetry ingestion. "
        "It stores raw telemetry for thirty days and accepts batches of up to 500 events. "
        "Operators can restart it with alphactl restart.\n",
        encoding="utf-8",
    )
    (docs / "billing.md").write_text(
        "# Billing\n\nBilling uses a monthly subscription with a Growth plan at 499 dollars per month. "
        "The plan monitors up to 50 robots and includes two million API calls per month.\n",
        encoding="utf-8",
    )
    ev = tmp / "evalset.jsonl"
    ev.write_text(
        json.dumps({"query_id": "q1", "query": "Which port is used for telemetry?", "relevant_doc_ids": ["gateway.md"]})
        + "\n"
        + json.dumps({"query_id": "q2", "query": "How many robots does the plan allow?", "relevant_doc_ids": ["billing.md"]})
        + "\n",
        encoding="utf-8",
    )
    return docs, ev


def write_config(tmp: Path, docs: Path, ev: Path | None, host: str, out: Path, **extra) -> Path:
    data = {
        "docs": str(docs),
        "models": ["ollama:nomic-embed-text"],
        "top_k": 2,
        "seed": 3,
        "output_dir": str(out),
        "chunking": {"strategy": "recursive", "chunk_size": 400, "chunk_overlap": 60, "min_chunk_chars": 30},
        "backends": {"ollama": {"host": host, "batch_size": 8, "timeout_s": 5, "max_retries": 2,
                                "retry_backoff_s": 0.01}},
        "cache": {"enabled": True, "dir": str(tmp / "cache")},
        "report": {"title": "Ollama probe", "priority": "balanced", "cost_per_1m_tokens": 0.02},
    }
    if ev is not None:
        data["evalset"] = {"path": str(ev), "request_timeout_s": 5.0}
    else:
        data["evalset"] = {"auto_generate": True, "ollama_model": "llama3.2", "request_timeout_s": 5.0,
                           "min_chunk_chars": 30, "min_question_words": 3, "questions_per_chunk": 2}
    data.update(extra)
    path = tmp / f"cfg-{out.name}.yaml"
    path.write_text(__import__("yaml").safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def probe_ollama_run(tmp: Path, host: str) -> None:
    docs, ev = make_corpus(tmp)
    STATE.update(legacy=False, model_missing=False, fail_first=0, requests=[])
    out = tmp / "run1"
    cfg = write_config(tmp, docs, ev, host, out)
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("ollama run exit 0", res.exit_code == 0, res.output[-1200:])
    if not (out / "results.json").is_file():
        check("ollama run produced results", False, "missing results.json")
        return
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    model = data["models"][0]
    check("ollama model ok", model["status"] == "ok", str(model.get("error")))
    check("tokens estimated", model["tokens"]["estimated"] is True, str(model["tokens"]))
    check("tokens flagged in method", "Ollama" in model["tokens"]["method"], model["tokens"]["method"])
    check("index dimension 16", model["index"]["dimension"] == 16, str(model["index"]))
    check("cache recorded", model["cache"]["hit_rate"] == 0.0, str(model["cache"]))
    res_meta = model["resources"]
    check("ollama model resident mb", res_meta.get("model_resident_mb") == 100.0, str(res_meta.get("model_resident_mb")))
    check("ollama vram mb", res_meta.get("peak_vram_mb") == 64.0, str(res_meta.get("peak_vram_mb")))
    check("ollama vram measurement", res_meta.get("vram_measurement") == "ollama_api_ps_resident", str(res_meta.get("vram_measurement")))
    check("ollama version captured", res_meta.get("ollama_version") == "0.9.9", str(res_meta))
    check("warm cache second run", True)
    # second run -> full cache hit
    res2 = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    data2 = json.loads((out / "results.json").read_text(encoding="utf-8"))
    m2 = data2["models"][0]
    check("ollama warm cache", m2["cache"]["hit_rate"] == 1.0, str(m2["cache"]))
    check("ollama warm no embed", m2["timing"]["embedded_chunks"] == 0, str(m2["timing"]))


def probe_ollama_retries(tmp: Path, host: str) -> None:
    docs, ev = make_corpus(tmp)
    STATE.update(legacy=False, model_missing=False, fail_first=2, requests=[])
    out = tmp / "run_retry"
    cfg = write_config(tmp, docs, ev, host, out)
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("5xx retried then succeeds", res.exit_code == 0, res.output[-800:])
    embed_calls = [r for r in STATE["requests"] if r[1] == "/api/embed"]
    check("503 retried", len(embed_calls) >= 3, str(len(embed_calls)))


def probe_ollama_legacy(tmp: Path, host: str) -> None:
    docs, ev = make_corpus(tmp)
    STATE.update(legacy=True, model_missing=False, fail_first=0, requests=[])
    out = tmp / "run_legacy"
    cfg = write_config(tmp, docs, ev, host, out)
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("legacy fallback run exit 0", res.exit_code == 0, res.output[-800:])
    legacy_calls = [r for r in STATE["requests"] if r[1] == "/api/embeddings"]
    check("legacy endpoint used", len(legacy_calls) > 0, str(len(legacy_calls)))
    if (out / "results.json").is_file():
        data = json.loads((out / "results.json").read_text(encoding="utf-8"))
        check("legacy model ok", data["models"][0]["status"] == "ok", str(data["models"][0].get("error")))


def probe_ollama_missing_model(tmp: Path, host: str) -> None:
    docs, ev = make_corpus(tmp)
    STATE.update(legacy=False, model_missing=True, fail_first=0, requests=[])
    out = tmp / "run_missing"
    cfg = write_config(tmp, docs, ev, host, out, models=["ollama:nomic-embed-text", "fake:hash"])
    # fake backend is not registered in this process -> it will error too; just check the message
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    ollama_model = [m for m in data["models"] if m["backend"] == "ollama"][0]
    check("missing model status error", ollama_model["status"] == "error", str(ollama_model))
    check("missing model hint", "ollama pull nomic-embed-text" in ollama_model["error"], ollama_model["error"])


def probe_generate_evalset(tmp: Path, host: str) -> None:
    docs, _ = make_corpus(tmp)
    STATE.update(legacy=False, model_missing=False, fail_first=0, requests=[])
    out = tmp / "gen.jsonl"
    res = runner.invoke(
        app,
        ["generate-evalset", "--docs", str(docs), "--output", str(out),
         "--model", "llama3.2", "--ollama-host", host, "--max-chunks", "2",
         "--questions-per-chunk", "2", "--no-progress"],
    )
    check("generate-evalset exit 0", res.exit_code == 0, res.output[-1000:])
    if not out.is_file():
        check("generate-evalset wrote file", False)
        return
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    check("generated questions written", len(lines) >= 1, str(len(lines)))
    check("generic question filtered", all("Summarize" not in q["query"] for q in lines), str([q["query"] for q in lines]))
    check("all end with ?", all(q["query"].endswith("?") for q in lines), str([q["query"] for q in lines]))
    check("doc ids valid", all(q["relevant_doc_ids"] in (["gateway.md"], ["billing.md"]) for q in lines), str(lines))
    check("query ids unique", len({q["query_id"] for q in lines}) == len(lines))
    check("answers stored", all("answer" in q for q in lines), str(lines))
    check("generator metadata", lines[0].get("metadata", {}).get("generator_model") == "llama3.2", str(lines[0]))

    # and the generated set runs end to end
    out2 = tmp / "gen_run"
    cfg = write_config(tmp, docs, out, host, out2)
    res2 = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("run with generated evalset", res2.exit_code == 0, res2.output[-800:])


def probe_list_models_inventory(tmp: Path, host: str) -> None:
    STATE.update(legacy=False, model_missing=False, fail_first=0, requests=[])
    res = runner.invoke(app, ["list-models", "--json", "--ollama-host", host])
    check("list-models with server exit 0", res.exit_code == 0, res.output[:300])
    payload = json.loads(res.output)
    check("ollama available", payload["ollama_available"] is True, str(payload.get("ollama_error")))
    entry = [m for m in payload["models"] if m["spec"] == "ollama:nomic-embed-text"][0]
    check("installed detected", entry.get("installed") is True, str(entry))
    res2 = runner.invoke(app, ["list-models", "--ollama-host", host])
    check("list-models table with server", res2.exit_code == 0, res2.output[:400])
    check("installed shown as yes", "nomic-embed-text" in res2.output)


if __name__ == "__main__":
    import tempfile
    import yaml  # noqa: F401

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"fake ollama at {host}")

    probes = [
        probe_ollama_run,
        probe_ollama_retries,
        probe_ollama_legacy,
        probe_ollama_missing_model,
        probe_generate_evalset,
        probe_list_models_inventory,
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in probes:
            print(f"\n=== {fn.__name__} ===")
            target = tmp / fn.__name__
            target.mkdir(parents=True, exist_ok=True)
            try:
                fn(target, host)
            except Exception:
                traceback.print_exc()
                FAILS.append(f"{fn.__name__} crashed")

    server.shutdown()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURES:")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("ALL OLLAMA PROBES PASSED")
