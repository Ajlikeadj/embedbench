"""CLI behaviour probes: exit codes, error paths, artifact handling."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import yaml
from typer.testing import CliRunner

from embedbench.cli import app

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fake_backend import FakeEmbeddingBackend  # noqa: F401  (registers "fake")

FAILS: list[str] = []
ROOT = Path(__file__).resolve().parent
runner = CliRunner()


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(f"{name} {detail}")


def build(tmp: Path) -> tuple[Path, Path]:
    docs = tmp / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "alpha.md").write_text(
        "# Alpha\n\nThe Alpha gateway listens on port 8443 for telemetry. "
        "It stores raw telemetry for thirty days and accepts batches of up to 500 events.\n",
        encoding="utf-8",
    )
    (docs / "beta.md").write_text(
        "# Beta\n\nBeta uses a monthly subscription with a Growth plan at 499 dollars per month. "
        "The plan monitors up to 50 robots and includes two million API calls.\n",
        encoding="utf-8",
    )
    ev = tmp / "evalset.jsonl"
    ev.write_text(
        json.dumps(
            {"query_id": "q1", "query": "Which port does the gateway use?", "relevant_doc_ids": ["alpha.md"]}
        )
        + "\n"
        + json.dumps(
            {"query_id": "q2", "query": "How many robots does the plan support?", "relevant_doc_ids": ["beta.md"]}
        )
        + "\n",
        encoding="utf-8",
    )
    return docs, ev


def cfg_dict(docs: Path, ev: Path, tmp: Path, **extra) -> dict:
    base = {
        "docs": str(docs),
        "models": ["fake:hash"],
        "top_k": 2,
        "seed": 5,
        "output_dir": str(tmp / "out"),
        "chunking": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 50, "min_chunk_chars": 30},
        "evalset": {"path": str(ev)},
        "backends": {"extra": {"fake": {"dimension": 64}}},
        "cache": {"enabled": True, "dir": str(tmp / "cache")},
        "report": {"title": "Probe report", "priority": "balanced", "cost_per_1m_tokens": 0.02},
    }
    base.update(extra)
    return base


def write_cfg(tmp: Path, data: dict, name: str = "config.yaml") -> Path:
    path = tmp / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def probe_happy(tmp: Path) -> None:
    docs, ev = build(tmp)
    path = write_cfg(tmp, cfg_dict(docs, ev, tmp))
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("happy exit 0", res.exit_code == 0, res.output[-800:])
    out = tmp / "out"
    for name in ("results.json", "results.csv", "queries.csv", "report.html", "config.snapshot.yaml", "run.log"):
        check(f"artifact {name}", (out / name).is_file())
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    check("successful", data["run"]["successful_models"] == 1)
    check("artifacts recorded", "artifacts" in data, str(list(data.keys())))
    # queries.csv rows
    rows = (out / "queries.csv").read_text(encoding="utf-8").strip().splitlines()
    check("queries.csv rows", len(rows) == 3, str(len(rows)))
    # results.csv rows
    rrows = (out / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    check("results.csv rows", len(rrows) == 2, str(len(rrows)))


def probe_missing_docs(tmp: Path) -> None:
    _, ev = build(tmp)
    path = write_cfg(tmp, cfg_dict(tmp / "nope", ev, tmp), "missing.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("missing docs exit 1", res.exit_code == 1, str(res.exit_code))
    check("missing docs message", "Documents folder not found" in res.output, res.output[-300:])


def probe_missing_evalset(tmp: Path) -> None:
    docs, _ = build(tmp)
    data = cfg_dict(docs, tmp / "nope.jsonl", tmp)
    path = write_cfg(tmp, data, "ev.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("missing evalset exit 1", res.exit_code == 1, str(res.exit_code))
    check("missing evalset message", "Evaluation set not found" in res.output, res.output[-300:])


def probe_unknown_doc_ref(tmp: Path) -> None:
    docs, ev = build(tmp)
    ev.write_text(
        json.dumps({"query_id": "q9", "query": "What about ghost?", "relevant_doc_ids": ["ghost.md"]}) + "\n",
        encoding="utf-8",
    )
    path = write_cfg(tmp, cfg_dict(docs, ev, tmp), "ghost.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("ghost doc exit 1", res.exit_code == 1, str(res.exit_code))
    check("ghost doc message", "ghost.md" in res.output, res.output[-400:])


def probe_all_models_fail(tmp: Path) -> None:
    docs, ev = build(tmp)
    data = cfg_dict(docs, ev, tmp)
    data["models"] = ["fake:missing", "nope:spec"]
    path = write_cfg(tmp, data, "allfail.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("all models fail exit 1", res.exit_code == 1, str(res.exit_code))
    check("all fail message", "All models failed" in res.output, res.output[-400:])
    out = tmp / "allfail_out"
    data2 = cfg_dict(docs, ev, tmp, models=["fake:missing", "nope:spec"], output_dir=str(out))
    path2 = write_cfg(tmp, data2, "allfail2.yaml")
    runner.invoke(app, ["run", "--config", str(path2), "--no-progress"])
    check("failed run still writes artifacts", (out / "results.json").is_file())


def probe_invalid_spec_partial(tmp: Path) -> None:
    docs, ev = build(tmp)
    data = cfg_dict(docs, ev, tmp, models=["bad", "fake:hash"], output_dir=str(tmp / "partial_out"))
    path = write_cfg(tmp, data, "partial.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("invalid spec: other model still runs", res.exit_code == 0, res.output[-500:])
    data_json = json.loads((tmp / "partial_out" / "results.json").read_text(encoding="utf-8"))
    check("invalid spec recorded as error", data_json["run"]["failed_models"] == 1)
    check("invalid spec ok model present", data_json["run"]["successful_models"] == 1)


def probe_bad_priority(tmp: Path) -> None:
    docs, ev = build(tmp)
    path = write_cfg(tmp, cfg_dict(docs, ev, tmp), "prio.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--priority", "wrong", "--no-progress"])
    check("bad priority exit 1", res.exit_code == 1, str(res.exit_code))
    check("bad priority message", "Invalid priority" in res.output, res.output[-300:])


def probe_unknown_config_key(tmp: Path) -> None:
    docs, ev = build(tmp)
    data = cfg_dict(docs, ev, tmp)
    data["top_kk"] = 3
    path = write_cfg(tmp, data, "typo.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("typo key exit 1", res.exit_code == 1, str(res.exit_code))
    check("typo key message", "top_kk" in res.output or "Invalid configuration" in res.output, res.output[-300:])


def probe_report_command(tmp: Path) -> None:
    docs, ev = build(tmp)
    data = cfg_dict(docs, ev, tmp, output_dir=str(tmp / "rep_out"))
    path = write_cfg(tmp, data, "rep.yaml")
    runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    res = runner.invoke(app, ["report", "--results", str(tmp / "rep_out"), "--priority", "accuracy"])
    check("report dir exit 0", res.exit_code == 0, res.output[-400:])
    html = (tmp / "rep_out" / "report.html").read_text(encoding="utf-8")
    check("report reflects accuracy priority", "Best for Accuracy-first" in html, html[html.find("selected-label"): html.find("selected-label") + 200] if "selected-label" in html else "no label")

    res2 = runner.invoke(app, ["report", "--results", str(tmp / "rep_out" / "results.json"), "--output", str(tmp / "custom.html")])
    check("report file exit 0", res2.exit_code == 0, res2.output[-300:])
    check("custom html written", (tmp / "custom.html").is_file())

    # output that is a directory name that does not exist yet but has no suffix
    res3 = runner.invoke(app, ["report", "--results", str(tmp / "rep_out"), "--output", str(tmp / "outdir")])
    check("report output dir", res3.exit_code == 0, res3.output[-300:])
    check("report output dir file", (tmp / "outdir" / "report.html").is_file(), str(list((tmp / "outdir").glob("*")) if (tmp / "outdir").exists() else "missing"))

    res4 = runner.invoke(app, ["report", "--results", str(tmp / "does-not-exist")])
    check("report missing results exit 1", res4.exit_code == 1, str(res4.exit_code))


def probe_output_dir_reuse(tmp: Path) -> None:
    docs, ev = build(tmp)
    out = tmp / "reuse_out"
    data = cfg_dict(docs, ev, tmp, output_dir=str(out))
    path = write_cfg(tmp, data, "reuse.yaml")
    runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    log1 = (out / "run.log").read_text(encoding="utf-8")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("second run exit 0", res.exit_code == 0, res.output[-300:])
    log2 = (out / "run.log").read_text(encoding="utf-8")
    check("run.log recreated not appended", len(log2) <= len(log1) * 1.5, f"{len(log1)} -> {len(log2)}")


def probe_default_config_discovery(tmp: Path) -> None:
    docs, ev = build(tmp)
    data = cfg_dict(docs, ev, tmp, output_dir=str(tmp / "disc_out"))
    (tmp / "embedbench.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    import os

    old = Path.cwd()
    try:
        os.chdir(tmp)
        res = runner.invoke(app, ["run", "--no-progress"], catch_exceptions=False)
    finally:
        os.chdir(old)
    check("default config discovered", res.exit_code == 0, res.output[-500:])


def probe_evalset_validation(tmp: Path) -> None:
    docs, ev = build(tmp)
    ev.write_text(
        json.dumps({"query_id": "dup", "query": "One?", "relevant_doc_ids": ["alpha.md"]}) + "\n"
        + json.dumps({"query_id": "dup", "query": "Two?", "relevant_doc_ids": ["alpha.md"]}) + "\n",
        encoding="utf-8",
    )
    path = write_cfg(tmp, cfg_dict(docs, ev, tmp), "dup.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("duplicate query_id exit 1", res.exit_code == 1, str(res.exit_code))
    check("duplicate query_id message", "Duplicate query_id" in res.output, res.output[-300:])

    # empty query rejected
    ev.write_text(json.dumps({"query_id": "x", "query": "  ", "relevant_doc_ids": ["alpha.md"]}) + "\n", encoding="utf-8")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("empty query exit 1", res.exit_code == 1, str(res.exit_code))

    # auto ids
    ev.write_text(
        json.dumps({"query": "Which port does the gateway use?", "relevant_doc_ids": ["alpha.md"], "extra": 1}) + "\n"
        + json.dumps({"query": "How many robots are supported?", "relevant_doc_ids": ["beta.md"]}) + "\n",
        encoding="utf-8",
    )
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("auto ids run", res.exit_code == 0, res.output[-300:])
    data = json.loads((tmp / "out" / "results.json").read_text(encoding="utf-8"))
    ids = [e["query_id"] for e in data["run"]["evalset"]["examples"]]
    check("auto ids generated", ids == ["q0001", "q0002"], str(ids))
    check("unknown field preserved", data["run"]["evalset"]["examples"][0].get("metadata", {}).get("extra") == 1,
          str(data["run"]["evalset"]["examples"][0]))


def probe_no_evalset_configured(tmp: Path) -> None:
    docs, ev = build(tmp)
    data = cfg_dict(docs, ev, tmp, output_dir=str(tmp / "noev_out"))
    del data["evalset"]
    path = write_cfg(tmp, data, "noev.yaml")
    res = runner.invoke(app, ["run", "--config", str(path), "--no-progress"])
    check("no evalset exit 1", res.exit_code == 1, str(res.exit_code))
    check("no evalset message", "No evaluation set configured" in res.output, res.output[-400:])


def probe_topk_and_chunk_flags(tmp: Path) -> None:
    docs, ev = build(tmp)
    data = cfg_dict(docs, ev, tmp, output_dir=str(tmp / "flags_out"))
    path = write_cfg(tmp, data, "flags.yaml")
    res = runner.invoke(
        app,
        ["run", "--config", str(path), "--top-k", "1", "--chunking", "fixed", "--chunk-size", "120",
         "--chunk-overlap", "20", "--seed", "9", "--batch-size", "4", "--cost-per-1m", "1.5",
         "--no-cache", "--no-progress"],
    )
    check("flags run exit 0", res.exit_code == 0, res.output[-600:])
    snap = yaml.safe_load((tmp / "flags_out" / "config.snapshot.yaml").read_text(encoding="utf-8"))
    check("flag top_k applied", snap["top_k"] == 1, str(snap["top_k"]))
    check("flag chunking applied", snap["chunking"]["strategy"] == "fixed", str(snap["chunking"]))
    check("flag chunk_size applied", snap["chunking"]["chunk_size"] == 120)
    check("flag chunk_overlap applied", snap["chunking"]["chunk_overlap"] == 20)
    check("flag seed applied", snap["seed"] == 9)
    check("flag cost applied", snap["report"]["cost_per_1m_tokens"] == 1.5)
    check("flag cache off", snap["cache"]["enabled"] is False)
    res2 = runner.invoke(app, ["run", "--config", str(path), "--chunk-overlap", "500", "--no-progress"])
    check("overlap >= size rejected", res2.exit_code == 1, str(res2.exit_code))


def probe_list_models(tmp: Path) -> None:
    res = runner.invoke(app, ["list-models", "--json", "--ollama-host", "http://127.0.0.1:1"])
    check("list-models json", res.exit_code == 0, res.output[:300])
    payload = json.loads(res.output)
    check("list-models has models", len(payload["models"]) > 5, str(len(payload.get("models", []))))
    check("list-models ollama unreachable", payload["ollama_available"] is False)
    res2 = runner.invoke(app, ["list-models", "--ollama-host", "http://127.0.0.1:1"])
    check("list-models table", res2.exit_code == 0, res2.output[:300])


if __name__ == "__main__":
    import tempfile

    probes = [
        probe_happy,
        probe_missing_docs,
        probe_missing_evalset,
        probe_unknown_doc_ref,
        probe_all_models_fail,
        probe_invalid_spec_partial,
        probe_bad_priority,
        probe_unknown_config_key,
        probe_report_command,
        probe_output_dir_reuse,
        probe_evalset_validation,
        probe_no_evalset_configured,
        probe_topk_and_chunk_flags,
        probe_list_models,
        probe_default_config_discovery,
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in probes:
            print(f"\n=== {fn.__name__} ===")
            try:
                target = tmp / fn.__name__
                target.mkdir(parents=True, exist_ok=True)
                fn(target)
            except Exception:
                traceback.print_exc()
                FAILS.append(f"{fn.__name__} crashed")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURES:")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("ALL CLI PROBES PASSED")
