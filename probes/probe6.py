"""The one documented flow still untested: `run --auto-evalset` (LLM generation inside a run)."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import traceback
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe3 import FakeOllama, STATE  # noqa: E402  fake server + state

from typer.testing import CliRunner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fake_backend import FakeEmbeddingBackend  # noqa: F401,E402

from embedbench.cli import app  # noqa: E402

FAILS: list[str] = []
runner = CliRunner()


def check(name: str, cond: bool, detail: str = "") -> None:
    try:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} {detail}")
    except UnicodeEncodeError:
        print(f"[{'ok  ' if cond else 'FAIL'}] {name} (detail not printable)")
    if not cond:
        FAILS.append(f"{name} {detail}")


def corpus(tmp: Path) -> tuple[Path, Path]:
    docs = tmp / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "gateway.md").write_text(
        "# Gateway\n\nThe Alpha gateway listens on port 8443 for telemetry ingestion. "
        "It stores raw telemetry for thirty days and accepts batches of up to 500 events. "
        "Operators restart it with alphactl restart and watch the health endpoint on 8787.\n",
        encoding="utf-8",
    )
    (docs / "billing.md").write_text(
        "# Billing\n\nBilling uses a monthly subscription with a Growth plan at 499 dollars per month. "
        "The plan monitors up to 50 robots and includes two million API calls per month. "
        "Annual subscriptions receive a 15 percent discount.\n",
        encoding="utf-8",
    )
    return docs, tmp / "out"


def write_cfg(tmp: Path, docs: Path, out: Path, extra: dict) -> Path:
    data = {
        "docs": str(docs),
        "models": ["fake:hash"],
        "top_k": 2,
        "seed": 7,
        "output_dir": str(out),
        "chunking": {"strategy": "recursive", "chunk_size": 400, "chunk_overlap": 60, "min_chunk_chars": 30},
        "evalset": {
            "auto_generate": True,
            "ollama_model": "llama3.2",
            "min_chunk_chars": 30,
            "min_question_words": 3,
            "questions_per_chunk": 2,
            "max_chunks": 2,
            "request_timeout_s": 5.0,
        },
        "backends": {"extra": {"fake": {"dimension": 32}},
                     "ollama": {"host": extra["host"], "timeout_s": 5.0, "max_retries": 1,
                                "retry_backoff_s": 0.01}},
        "cache": {"enabled": True, "dir": str(tmp / "cache")},
        "report": {"title": "Auto evalset probe", "priority": "balanced"},
    }
    data.update(extra.get("config", {}))
    path = tmp / "cfg.yaml"
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def probe_auto_generation(tmp: Path, host: str) -> None:
    docs, out = corpus(tmp)
    STATE.update(legacy=False, model_missing=False, fail_first=0, requests=[])
    cfg = write_cfg(tmp, docs, out, {"host": host})
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("auto-evalset run exit 0", res.exit_code == 0, res.output[-1500:])
    check("generated artifact written", (out / "evalset.generated.jsonl").is_file())
    if not (out / "results.json").is_file():
        check("results written", False)
        return
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    ev = data["run"]["evalset"]
    check("source is synthetic", ev["source"] == "synthetic", str(ev["source"]))
    check("generation stats present", isinstance(ev.get("generation"), dict) and ev["generation"]["accepted"] >= 1,
          str(ev.get("generation")))
    check("generated path recorded", ev["path"] and ev["path"].endswith("evalset.generated.jsonl"), str(ev["path"]))
    check("queries generated", ev["queries"] >= 1, str(ev["queries"]))
    check("run succeeded with generated set", data["run"]["successful_models"] == 1, str(data["run"]))
    check("generic questions filtered", all("Summarize" not in e["query"] for e in ev["examples"]),
          str([e["query"] for e in ev["examples"]]))
    check("questions are questions", all(e["query"].endswith("?") for e in ev["examples"]),
          str([e["query"] for e in ev["examples"]]))
    check("answers stored", all("answer" in e for e in ev["examples"]), str(ev["examples"]))
    check("source chunk recorded", all(e.get("source_chunk_id") for e in ev["examples"]), str(ev["examples"]))
    check("doc ids valid", all(e["relevant_doc_ids"] in (["gateway.md"], ["billing.md"]) for e in ev["examples"]),
          str(ev["examples"]))

    # rerun without a path: generation is deterministic with the same seed
    saved = [e["query"] for e in ev["examples"]]
    res2 = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    data2 = json.loads((out / "results.json").read_text(encoding="utf-8"))
    check("second auto run ok", res2.exit_code == 0, res2.output[-600:])
    check("generation deterministic", [e["query"] for e in data2["run"]["evalset"]["examples"]] == saved,
          str([e["query"] for e in data2["run"]["evalset"]["examples"]]))


def probe_auto_generation_down(tmp: Path, host: str) -> None:
    docs, out = corpus(tmp)
    STATE.update(legacy=False, model_missing=True, fail_first=0, requests=[])
    cfg = write_cfg(tmp, docs, tmp / "down_out", {"host": "http://127.0.0.1:1"})
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("unreachable Ollama exits 1", res.exit_code == 1, str(res.exit_code))
    check("actionable error", "Could not reach Ollama" in res.output or "ollama serve" in res.output,
          res.output[-500:])


def probe_missing_model_during_generation(tmp: Path, host: str) -> None:
    docs, out = corpus(tmp)
    STATE.update(legacy=False, model_missing=True, fail_first=0, requests=[])
    cfg = write_cfg(tmp, docs, tmp / "missing_out", {"host": host})
    res = runner.invoke(app, ["run", "--config", str(cfg), "--no-progress"])
    check("missing evalset model exits 1", res.exit_code == 1, str(res.exit_code))
    check("pull hint shown", "ollama pull llama3.2" in res.output, res.output[-500:])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"fake ollama at {host}")
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for fn in (probe_auto_generation, probe_auto_generation_down, probe_missing_model_during_generation):
                print(f"\n=== {fn.__name__} ===")
                target = tmp / fn.__name__
                target.mkdir(parents=True, exist_ok=True)
                try:
                    fn(target, host)
                except Exception:
                    traceback.print_exc()
                    FAILS.append(f"{fn.__name__} crashed")
    finally:
        server.shutdown()

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURES:")
        for item in FAILS:
            print(" -", item)
        sys.exit(1)
    print("ALL AUTO-EVALSET PROBES PASSED")
