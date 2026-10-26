from __future__ import annotations

import json
import sys
import types
from typing import Any

import numpy as np
import pytest
import requests
from embedbench.backends import available_backends, create_backend, load_plugins, parse_model_spec
from embedbench.backends.base import EmbeddingBackend
from embedbench.backends.hf import HuggingFaceBackend
from embedbench.backends.ollama import OllamaBackend
from embedbench.config import RunConfig
from embedbench.errors import BackendError
from embedbench.ollama_client import OllamaClient, OllamaError


class RecordingBackend(EmbeddingBackend):
    backend_name = "recording"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self.seen: list[list[str]] = []

    def load(self) -> None:
        pass

    def _embed_prepared(self, texts: list[str]) -> np.ndarray:
        self.seen.append(list(texts))
        return np.ones((len(texts), 3), dtype=np.float32)


def test_base_interface_prefixes_and_normalization() -> None:
    backend = RecordingBackend(
        "test-model",
        {"query_prefix": "QUERY: ", "document_prefix": "PASSAGE: ", "batch_size": 8},
    )
    backend.load()

    documents = backend.embed_documents(["doc one", "doc two"])
    queries = backend.embed_queries(["what?"])

    assert backend.dimension == 3
    assert backend.known_dimension == 3
    assert backend.max_batch_size == 8
    assert backend.seen == [
        ["PASSAGE: doc one", "PASSAGE: doc two"],
        ["QUERY: what?"],
    ]
    assert documents.shape == (2, 3)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1.0)


def test_base_interface_rejects_wrong_row_count_and_non_finite() -> None:
    class ShortBackend(RecordingBackend):
        backend_name = "short"

        def _embed_prepared(self, texts: list[str]) -> np.ndarray:
            return np.ones((max(0, len(texts) - 1), 3), dtype=np.float32)

    with pytest.raises(BackendError, match="returned 1 embeddings"):
        ShortBackend("short").embed_documents(["a", "b"])

    class NanBackend(RecordingBackend):
        backend_name = "nan"

        def _embed_prepared(self, texts: list[str]) -> np.ndarray:
            return np.full((len(texts), 3), np.nan, dtype=np.float32)

    with pytest.raises(BackendError, match="non-finite"):
        NanBackend("nan").embed_documents(["a"])


def test_base_interface_rejects_empty_texts() -> None:
    backend = RecordingBackend("test")
    with pytest.raises(BackendError, match="empty"):
        backend.embed_documents(["valid", "   "])
    empty = backend.embed_queries([])
    assert empty.shape == (0, 0)


def test_heuristic_token_count() -> None:
    backend = RecordingBackend("test")
    count = backend.count_tokens(["one two three.", "four!"])
    assert count.tokens == 6  # 3 words + "." + 1 word + "!"
    assert count.estimated is True


def test_registry_and_factory() -> None:
    spec = parse_model_spec("fake:hash")
    assert (spec.backend, spec.model) == ("fake", "hash")
    assert "hf" in available_backends()
    assert "ollama" in available_backends()
    assert "fake" in available_backends()

    backend = create_backend("fake:hash", RunConfig(models=["fake:hash"]))
    backend.load()
    vectors = backend.embed_documents(["alpha beta", "gamma"])
    assert vectors.shape == (2, 128)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


@pytest.mark.parametrize("raw", ["no-colon", ":model", "backend:", ""])
def test_bad_model_specs(raw: str) -> None:
    with pytest.raises(BackendError, match="backend:model|Invalid model spec"):
        parse_model_spec(raw)


def test_unknown_backend_is_actionable() -> None:
    with pytest.raises(BackendError, match="Known backends"):
        create_backend("does-not-exist:model", RunConfig())


def test_python_file_plugin_registers_backend(tmp_path) -> None:
    plugin = tmp_path / "my_backend.py"
    plugin.write_text(
        "import numpy as np\n"
        "from embedbench.backends import register_backend\n"
        "from embedbench.backends.base import EmbeddingBackend\n\n"
        "@register_backend('plugin_echo')\n"
        "class PluginEcho(EmbeddingBackend):\n"
        "    backend_name = 'plugin_echo'\n"
        "    def load(self): pass\n"
        "    def _embed_prepared(self, texts):\n"
        "        return np.ones((len(texts), 4), dtype=np.float32)\n",
        encoding="utf-8",
    )

    load_plugins([str(plugin)])
    backend = create_backend("plugin_echo:demo", RunConfig())
    backend.load()
    assert backend.embed_queries(["hello"]).shape == (1, 4)
    assert "plugin_echo" in available_backends()


def test_ollama_backend_missing_model_suggests_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(self: OllamaClient, model: str) -> None:
        raise OllamaError(f"Model '{model}' is not installed. Run: ollama pull {model}")

    monkeypatch.setattr(OllamaClient, "require_model", missing)
    backend = OllamaBackend("nomic-embed-text", {"host": "http://example.invalid"})
    with pytest.raises(BackendError, match="ollama pull nomic-embed-text"):
        backend.load()


def test_ollama_backend_embeds_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OllamaClient, "require_model", lambda self, model: None)
    monkeypatch.setattr(
        OllamaClient,
        "embed",
        lambda self, model, texts, keep_alive=None: [[1.0, 2.0, 3.0] for _ in texts],
    )
    backend = OllamaBackend("nomic-embed-text", {"host": "http://example.invalid", "batch_size": 2})
    backend.load()
    vectors = backend.embed_documents(["one", "two"])
    assert vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    backend.close()


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, body: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = body if body is not None else json.dumps(payload or {})
        self.url = "http://example.invalid/api"
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def install_sequence(client: OllamaClient, monkeypatch: pytest.MonkeyPatch, sequence: list[Any]) -> list[str]:
    calls: list[str] = []

    def fake_request(method: str, url: str, json: Any = None, timeout: float = 0) -> FakeResponse:
        calls.append(url)
        item = sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client._session, "request", fake_request)
    return calls


def test_ollama_client_retries_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=2, retry_backoff_s=0)
    calls = install_sequence(
        client,
        monkeypatch,
        [requests.ConnectionError("boom"), FakeResponse(200, {"models": []})],
    )
    assert client.list_models() == []
    assert len(calls) == 2
    client.close()


def test_ollama_client_retries_503_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=1, retry_backoff_s=0)
    calls = install_sequence(
        client,
        monkeypatch,
        [FakeResponse(503, {"error": "loading"}), FakeResponse(200, {"models": []})],
    )
    assert client.list_models() == []
    assert len(calls) == 2
    client.close()


def test_ollama_connection_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=1, retry_backoff_s=0)
    install_sequence(
        client,
        monkeypatch,
        [requests.ConnectionError("no route"), requests.ConnectionError("no route")],
    )
    with pytest.raises(OllamaError, match="ollama serve"):
        client.list_models()
    client.close()


def test_ollama_require_model_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    install_sequence(client, monkeypatch, [FakeResponse(200, {"models": [{"name": "other:latest"}]})])
    with pytest.raises(OllamaError, match="ollama pull nomic-embed-text"):
        client.require_model("nomic-embed-text")
    client.close()


def test_ollama_require_model_accepts_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    install_sequence(client, monkeypatch, [FakeResponse(200, {"models": [{"name": "nomic-embed-text:latest"}]})])
    assert client.require_model("nomic-embed-text") is not None
    client.close()


def test_ollama_embed_falls_back_to_legacy_api(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    calls: list[str] = []

    def fake_request(method: str, url: str, json: Any = None, timeout: float = 0) -> FakeResponse:
        calls.append(url)
        if url.endswith("/api/embed"):
            return FakeResponse(404, body="404 page not found")
        prompt = json["prompt"]
        return FakeResponse(200, {"embedding": [float(len(prompt)), 2.0]})

    monkeypatch.setattr(client._session, "request", fake_request)
    vectors = client.embed("nomic-embed-text", ["ab", "abcd"])
    assert vectors == [[2.0, 2.0], [4.0, 2.0]]
    assert calls[0].endswith("/api/embed")
    assert calls[1].endswith("/api/embeddings")

    vectors_again = client.embed("nomic-embed-text", ["abc"])
    assert vectors_again == [[3.0, 2.0]]
    assert calls[-1].endswith("/api/embeddings")
    client.close()


def test_ollama_embed_missing_model_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    calls = install_sequence(
        client,
        monkeypatch,
        [FakeResponse(404, {"error": "model 'nomic-embed-text' not found"})],
    )
    with pytest.raises(OllamaError, match="ollama pull"):
        client.embed("nomic-embed-text", ["text"])
    assert len(calls) == 1
    client.close()


def test_hf_backend_with_mocked_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def __call__(self, batch: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
            max_length = kwargs.get("max_length") or 10_000
            return {"input_ids": [[1] * min(len(text), max_length) for text in batch]}

    class FakeSentenceTransformer:
        last_instance: "FakeSentenceTransformer | None" = None

        def __init__(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name
            self.kwargs = kwargs
            self.max_seq_length = 32
            self.tokenizer = FakeTokenizer()
            self.encoded: list[list[str]] = []
            FakeSentenceTransformer.last_instance = self

        def get_sentence_embedding_dimension(self) -> int:
            return 6

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            self.encoded.append(list(texts))
            return np.ones((len(texts), 6), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    backend = HuggingFaceBackend(
        "fake/model",
        {"device": "cpu", "query_prefix": "Q: ", "document_prefix": "D: ", "batch_size": 4},
    )
    backend.load()
    instance = FakeSentenceTransformer.last_instance
    assert instance is not None
    assert instance.kwargs["device"] == "cpu"
    assert backend.dimension == 6

    vectors = backend.embed_documents(["hello", "world"])
    assert vectors.shape == (2, 6)
    assert instance.encoded == [["D: hello", "D: world"]]
    assert backend.embed_queries(["what?"]).shape == (1, 6)
    assert instance.encoded[-1] == ["Q: what?"]

    count = backend.count_tokens(["abcd", "ab"])
    assert count.tokens == 6
    assert count.estimated is False
    assert count.method.startswith("tokenizer:")


def test_hf_e5_prefixes_are_applied_by_default() -> None:
    backend = HuggingFaceBackend("intfloat/e5-base-v2", {"device": "cpu"})
    assert backend.query_prefix == "query: "
    assert backend.document_prefix == "passage: "
    assert "q='query: '" in backend.cache_namespace

    other = HuggingFaceBackend("BAAI/bge-small-en-v1.5", {"device": "cpu"})
    assert other.query_prefix == ""
    assert other.document_prefix == ""
