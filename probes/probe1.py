"""Probes for code paths with no test coverage."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name + " " + detail)


# ---- 1. NumPy index vs FAISS index agreement -------------------------------
def probe_indexes() -> None:
    from embedbench.index import build_index, FaissIndex, NumpyIndex

    rng = np.random.default_rng(0)
    for dim, n, k in [(8, 50, 5), (64, 300, 10), (3, 1, 1), (16, 17, 40)]:
        vectors = rng.normal(size=(n, dim)).astype(np.float32)
        queries = rng.normal(size=(5, dim)).astype(np.float32)
        fa = build_index(vectors, backend="faiss")
        npidx = build_index(vectors, backend="numpy")
        for qi, q in enumerate(queries):
            fs, fi = fa.search(q, k)
            ns, ni = npidx.search(q, k)
            check(
                f"index agree d={dim} n={n} k={k} q={qi}",
                list(fi) == list(ni),
                f"faiss={list(fi)} numpy={list(ni)}",
            )
            if list(fi) == list(ni):
                check(
                    f"index scores d={dim} n={n} k={k} q={qi}",
                    np.allclose(fs, ns, atol=1e-5),
                    f"{fs} vs {ns}",
                )
        check(f"index disk size d={dim}", fa.disk_size_bytes() > 0)
        check(f"numpy disk size d={dim}", npidx.disk_size_bytes() > 0)

    # sorted descending?
    vectors = rng.normal(size=(40, 8)).astype(np.float32)
    idx = build_index(vectors, backend="numpy")
    s, i = idx.search(rng.normal(size=8).astype(np.float32), 7)
    check("numpy scores descending", all(s[j] >= s[j + 1] - 1e-6 for j in range(len(s) - 1)), str(s))
    idx2 = build_index(vectors, backend="faiss")
    s2, i2 = idx2.search(rng.normal(size=8).astype(np.float32), 7)
    check("faiss scores descending", all(s2[j] >= s2[j + 1] - 1e-6 for j in range(len(s2) - 1)), str(s2))
    # duplicates / zero vectors
    zero = np.zeros((4, 8), dtype=np.float32)
    z = build_index(zero, backend="auto")
    s3, i3 = z.search(np.ones(8, dtype=np.float32), 3)
    check("zero-vector index works", len(i3) == 3, str(i3))
    # error paths
    from embedbench.errors import IndexBuildError
    for bad, expect in [
        (np.zeros((0, 4), dtype=np.float32), "empty"),
        (np.zeros(4, dtype=np.float32), "1-D"),
    ]:
        try:
            build_index(bad, backend="auto")
            check(f"build_index {expect} raises", False)
        except IndexBuildError:
            check(f"build_index {expect} raises", True)


# ---- 2. Chunking invariants ------------------------------------------------
def probe_chunking() -> None:
    from embedbench.chunking import chunk_documents
    from embedbench.config import ChunkingConfig
    from embedbench.models import Document

    text = (
        "# Title\n\n"
        + "\n\n".join(
            f"Paragraph {i}. It has sentences here. Some more text follows with commas, "
            f"semicolons; and words. " * 3
            for i in range(12)
        )
    )
    doc = Document(doc_id="a.md", path=Path("a.md"), text=text, source_type="markdown")

    for strategy in ("fixed", "sentence", "recursive"):
        cfg = ChunkingConfig(strategy=strategy, chunk_size=300, chunk_overlap=60, min_chunk_chars=10)
        chunks = chunk_documents([doc], cfg)
        check(f"{strategy}: produces chunks", len(chunks) > 1, str(len(chunks)))
        check(
            f"{strategy}: ids/ordinals sequential",
            [c.ordinal for c in chunks] == list(range(len(chunks))),
        )
        check(
            f"{strategy}: chunk_id format",
            all(c.chunk_id == f"a.md#{c.ordinal:04d}" for c in chunks),
        )
        check(
            f"{strategy}: spans ordered",
            all(chunks[i].start_char <= chunks[i].end_char for i in range(len(chunks))),
        )
        # determinism
        again = chunk_documents([doc], cfg)
        check(
            f"{strategy}: deterministic",
            [c.text for c in chunks] == [c.text for c in again],
        )
        # every chunk's text must come from the document at its declared span
        bad = [
            c.chunk_id
            for c in chunks
            if text[c.start_char : c.end_char].strip() != c.text
        ]
        check(f"{strategy}: span matches text", not bad, str(bad[:3]))
        # no chunk exceeds the configured size (except hard-split fallback)
        oversized = [c.chunk_id for c in chunks if len(c.text) > 300]
        check(f"{strategy}: size respected", not oversized, str(oversized[:3]))

    # short doc fallback: a tiny document keeps one chunk
    tiny = Document(doc_id="t.md", path=Path("t.md"), text="Tiny.", source_type="markdown")
    for strategy in ("fixed", "sentence", "recursive"):
        cfg = ChunkingConfig(strategy=strategy, chunk_size=800, chunk_overlap=10, min_chunk_chars=80)
        chunks = chunk_documents([tiny], cfg)
        check(f"{strategy}: short doc fallback", len(chunks) == 1 and chunks[0].text == "Tiny.", str(chunks))
    # empty doc
    blank = Document(doc_id="b.md", path=Path("b.md"), text="   \n  ", source_type="markdown")
    check("empty doc -> no chunks", chunk_documents([blank], ChunkingConfig()) == [])


# ---- 3. Plugin backend loading (README documented) -------------------------
def probe_plugin(tmp: Path) -> None:
    plugin = tmp / "my_backend.py"
    plugin.write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from embedbench.backends import register_backend\n"
        "from embedbench.backends.base import EmbeddingBackend\n"
        "\n"
        "@register_backend('constant')\n"
        "class ConstantBackend(EmbeddingBackend):\n"
        "    backend_name = 'constant'\n"
        "    def load(self):\n"
        "        self.embedding_dimension = 8\n"
        "    def _embed_prepared(self, texts):\n"
        "        return np.ones((len(texts), 8), dtype=np.float32)\n",
        encoding="utf-8",
    )
    from embedbench.backends import load_plugins, create_backend
    from embedbench.config import RunConfig

    cfg = RunConfig(docs=Path("."), models=["constant:demo"], plugins=[str(plugin)])
    try:
        load_plugins(cfg.plugins)
        backend = create_backend("constant:demo", cfg)
        backend.load()
        vecs = backend.embed_documents(["hello", "world"])
        check("plugin backend loads + embeds", vecs.shape == (2, 8), str(vecs.shape))
        q = backend.embed_queries(["q"])
        check("plugin query embedding normalized", np.isclose(np.linalg.norm(q), 1.0, atol=1e-5), str(np.linalg.norm(q)))
        check("plugin cache namespace", backend.cache_namespace.startswith("constant:demo"), backend.cache_namespace)
    except Exception as exc:
        traceback.print_exc()
        check("plugin backend loads + embeds", False, repr(exc))


# ---- 4. utils helpers ------------------------------------------------------
def probe_utils() -> None:
    from embedbench.utils import (
        human_bytes,
        percentile_summary,
        relative_posix,
        set_global_seed,
        sha256_hex,
        truncate,
        write_json,
        json_default,
    )

    p = percentile_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    check("percentile p50", abs(p["p50"] - 3.0) < 1e-9, str(p))
    check("percentile count", p["count"] == 5)
    empty = percentile_summary([])
    check("percentile empty", empty["p50"] is None and empty["count"] == 0)
    check("human_bytes", human_bytes(0) == "0.0 B" and human_bytes(1024) == "1.0 KB", human_bytes(1024))
    check("human_bytes none", human_bytes(None) == "n/a")
    check("truncate", truncate("a b c", 10) == "a b c")
    check("truncate long", truncate("word " * 40, 20).endswith("...") and len(truncate("word " * 40, 20)) <= 20)
    check("sha256 stable", sha256_hex("x") == sha256_hex("x"))
    set_global_seed(7)
    import random as _r

    a = _r.random()
    set_global_seed(7)
    check("seed reproducible", a == _r.random())
    out = Path("probes") / "_out.json"
    write_json(out, {"a": np.float32(1.5), "b": np.arange(3)})
    import json as _json

    check("write_json numpy", _json.loads(out.read_text()) == {"a": 1.5, "b": [0, 1, 2]})
    out.unlink()


# ---- 5. cache behaviour ----------------------------------------------------
def probe_cache(tmp: Path) -> None:
    from embedbench.cache import EmbeddingCache

    cache_dir = tmp / "cache"
    with EmbeddingCache(cache_dir) as cache:
        texts = ["one", "two", "three"]
        vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
        check("cold misses", cache.get_many("ns", texts) == [None, None, None])
        check("miss counter", cache.misses == 3, str(cache.misses))
        cache.put_many("ns", texts, vectors)
        hits = cache.get_many("ns", texts)
        check("warm hits", all(h is not None for h in hits))
        check("hit counter", cache.hits == 3, str(cache.hits))
        check("vectors round-trip", all(np.allclose(h, v) for h, v in zip(hits, vectors)))
        check("namespace isolation", cache.get_many("other", texts) == [None, None, None])
        check("order preserved", [h is not None for h in cache.get_many("ns", ["two", "one"])] == [True, True])
        # duplicate texts
        dup = cache.get_many("ns", ["one", "one", "two"])
        check("duplicate text lookup", [d is not None for d in dup] == [True, True, True])
        # mismatched shapes are refused
        cache.put_many("ns", ["four"], np.zeros((2, 4), dtype=np.float32))
        check("shape mismatch not cached", cache.get_many("ns", ["four"]) == [None])
        # disabled cache
    disabled = EmbeddingCache(tmp / "cache2", enabled=False)
    check("disabled cache misses", disabled.get_many("ns", ["a"]) == [None])
    disabled.close()


# ---- 6. config validation --------------------------------------------------
def probe_config(tmp: Path) -> None:
    import yaml
    from embedbench.config import RunConfig, load_config, config_snapshot_yaml
    from embedbench.errors import ConfigError

    good = tmp / "ok.yaml"
    good.write_text("docs: ./d\nmodels: [hf:one]\n", encoding="utf-8")
    cfg = load_config(good)
    check("defaults top_k", cfg.top_k == 5)
    check("models as list", cfg.models == ["hf:one"])

    # unknown key rejected
    bad = tmp / "bad.yaml"
    bad.write_text("docs: ./d\ntop_kk: 3\n", encoding="utf-8")
    try:
        load_config(bad)
        check("unknown key rejected", False)
    except ConfigError as exc:
        check("unknown key rejected", True, str(exc)[:60])

    # dotted overrides
    cfg2 = load_config(good, {"top_k": 7, "backends.hf.batch_size": 4, "evalset.auto_generate": True})
    check("override top_k", cfg2.top_k == 7)
    check("override nested", cfg2.backends.hf.batch_size == 4)
    check("override auto", cfg2.evalset.auto_generate is True)

    # invalid overlap
    try:
        load_config(good, {"chunking.chunk_size": 100, "chunking.chunk_overlap": 150})
        check("overlap validated", False)
    except ConfigError:
        check("overlap validated", True)

    # models as comma string
    cfg3 = load_config(good, {"models": "hf:a, ollama:b"})
    check("models string split", cfg3.models == ["hf:a", "ollama:b"], str(cfg3.models))

    # snapshot round trip
    snap = config_snapshot_yaml(cfg2)
    check("snapshot header", snap.startswith("# embedbench configuration snapshot"))
    reparsed = yaml.safe_load("\n".join(snap.splitlines()[2:]))
    check("snapshot round-trip", RunConfig.model_validate(reparsed).top_k == 7)

    # missing config file
    try:
        load_config(tmp / "nope.yaml")
        check("missing config raises", False)
    except ConfigError:
        check("missing config raises", True)


# ---- 7. metrics vs hand computation ---------------------------------------
def probe_metrics() -> None:
    from embedbench.metrics import (
        evaluate_retrieval,
        aggregate_retrieval_metrics,
        dedupe_ranked_docs,
        recall_at_k,
        precision_at_k,
        mrr_at_k,
        ndcg_at_k,
        hit_rate_at_k,
    )

    rel = ["a", "b"]
    ret = ["x", "a", "y", "b", "z"]
    check("recall", abs(recall_at_k(rel, ret, 5) - 1.0) < 1e-9)
    check("recall partial", abs(recall_at_k(rel, ret, 2) - 0.5) < 1e-9)
    check("precision", abs(precision_at_k(rel, ret, 5) - 0.4) < 1e-9)
    check("precision k", abs(precision_at_k(rel, ret, 10) - 0.2) < 1e-9)
    check("mrr", abs(mrr_at_k(rel, ret, 5) - 0.5) < 1e-9)
    # ndcg with 2 relevant, first at rank 2, second at rank 4
    import math

    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    check("ndcg", abs(ndcg_at_k(rel, ret, 5) - dcg / idcg) < 1e-9, str(ndcg_at_k(rel, ret, 5)))
    check("hit", hit_rate_at_k(rel, ret, 5) == 1.0)
    check("no hit", hit_rate_at_k(["zz"], ret, 5) == 0.0)
    # duplicates collapse at first rank
    check("dedupe", dedupe_ranked_docs(["a", "a", "b", "a", "c"], 3) == ["a", "b", "c"])
    check("dedupe limit", dedupe_ranked_docs(["a", "b"], 5) == ["a", "b"])
    # k larger than list
    check("precision >len list", abs(precision_at_k(["a"], ["a", "b"], 10) - 0.1) < 1e-9)
    rows = [evaluate_retrieval(["a"], ["a"], 1), evaluate_retrieval(["a"], ["b"], 1)]
    agg = aggregate_retrieval_metrics(rows)
    check("aggregate recall", abs(agg["recall"] - 0.5) < 1e-9, str(agg))
    check("aggregate count", agg["n_queries"] == 2)
    for fn in (recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k, hit_rate_at_k):
        for bad_k in (0, -1):
            try:
                fn(["a"], ["a"], bad_k)
                check(f"{fn.__name__} k={bad_k} raises", False)
            except ValueError:
                check(f"{fn.__name__} k={bad_k} raises", True)
        try:
            fn([], ["a"], 1)
            check(f"{fn.__name__} empty relevant raises", False)
        except ValueError:
            check(f"{fn.__name__} empty relevant raises", True)
    try:
        aggregate_retrieval_metrics([])
        check("aggregate empty raises", False)
    except ValueError:
        check("aggregate empty raises", True)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn, args in [
            (probe_indexes, ()),
            (probe_chunking, ()),
            (probe_plugin, (tmp,)),
            (probe_utils, ()),
            (probe_cache, (tmp,)),
            (probe_config, (tmp,)),
            (probe_metrics, ()),
        ]:
            print(f"\n=== {fn.__name__} ===")
            try:
                fn(*args)
            except Exception:
                traceback.print_exc()
                FAILS.append(f"{fn.__name__} crashed")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURES:")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("ALL PROBES PASSED")
