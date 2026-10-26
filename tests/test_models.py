"""Tests for the embedding-model layer: base contract, registry and providers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from embedbench.config import RunConfig
from embedbench.errors import BackendError
from embedbench.models import (
    BaseEmbeddingModel,
    available_providers,
    create_backend,
    create_model,
    load_model_plugins,
    parse_model_spec,
    pricing_catalog,
    pricing_for,
    provider_metadata,
)
from embedbench.models.local import OllamaEmbeddingModel, SentenceTransformerEmbeddingModel
from embedbench.models.providers import (
    CohereEmbeddingModel,
    OpenAIEmbeddingModel,
    VoyageEmbeddingModel,
    is_retryable,
)
from embedbench.ollama_client import OllamaClient, OllamaError


class RecordingModel(BaseEmbeddingModel):
    """Records the exact batches it receives and returns constant vectors."""

    provider_name = "recording"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self.seen: list[list[str]] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def _embed_prepared(self, texts: list[str], kind: str = "document") -> np.ndarray:
        self.seen.append(list(texts))
        return np.ones((len(texts), 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------


def test_base_interface_prefixes_and_normalization() -> None:
    model = RecordingModel(
        "test-model",
        {"query_prefix": "QUERY: ", "document_prefix": "PASSAGE: ", "batch_size": 8},
    )
    model.load()

    documents = model.embed_documents(["doc one", "doc two"])
    queries = model.embed_queries(["what?"])

    assert model.dimension == 3
    assert model.known_dimension == 3
    assert model.max_batch_size == 8
    assert model.seen == [
        ["PASSAGE: doc one", "PASSAGE: doc two"],
        ["QUERY: what?"],
    ]
    assert isinstance(documents, list)
    assert isinstance(documents[0], list)
    matrix = np.asarray(documents, dtype=np.float32)
    assert matrix.shape == (2, 3)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(np.asarray(queries), axis=1), 1.0)


def test_batching_splits_work_and_preserves_order() -> None:
    model = RecordingModel("batched", {"batch_size": 2})
    model.load()
    texts = [f"document {index}" for index in range(5)]
    matrix = model.embed_documents_matrix(texts)

    assert matrix.shape == (5, 3)
    assert [len(batch) for batch in model.seen] == [2, 2, 1]
    assert [text for batch in model.seen for text in batch] == texts


def test_normalize_false_preserves_magnitudes() -> None:
    class Scaled(RecordingModel):
        provider_name = "scaled"

        def _embed_prepared(self, texts: list[str], kind: str = "document") -> np.ndarray:
            return np.full((len(texts), 2), 5.0, dtype=np.float32)

    model = Scaled("scaled", {"normalize": False})
    matrix = model.embed_documents_matrix(["a"])
    assert np.allclose(matrix, 5.0)


def test_base_interface_rejects_wrong_row_count_and_non_finite() -> None:
    class ShortModel(RecordingModel):
        provider_name = "short"

        def _embed_prepared(self, texts: list[str], kind: str = "document") -> np.ndarray:
            return np.ones((max(0, len(texts) - 1), 3), dtype=np.float32)

    with pytest.raises(BackendError, match="returned 1 embeddings"):
        ShortModel("short").embed_documents(["a", "b"])

    class NanModel(RecordingModel):
        provider_name = "nan"

        def _embed_prepared(self, texts: list[str], kind: str = "document") -> np.ndarray:
            return np.full((len(texts), 3), np.nan, dtype=np.float32)

    with pytest.raises(BackendError, match="non-finite"):
        NanModel("nan").embed_documents(["a"])


def test_dimension_change_between_calls_is_rejected() -> None:
    class Flaky(RecordingModel):
        provider_name = "flaky"

        def _embed_prepared(self, texts: list[str], kind: str = "document") -> np.ndarray:
            width = 3 if len(texts) == 1 and self.known_dimension is None else 4
            return np.ones((len(texts), width), dtype=np.float32)

    model = Flaky("flaky")
    model.embed_documents(["a"])
    with pytest.raises(BackendError, match="changed dimension"):
        model.embed_documents(["b"])


def test_base_interface_rejects_empty_texts() -> None:
    model = RecordingModel("test")
    with pytest.raises(BackendError, match="empty"):
        model.embed_documents(["valid", "   "])
    assert model.embed_queries([]) == []
    assert model.embed_queries_matrix([]).shape == (0, 0)


def test_heuristic_token_count() -> None:
    model = RecordingModel("test")
    count = model.count_tokens(["one two three.", "four!"])
    assert count.tokens == 6  # 3 words + "." + 1 word + "!"
    assert count.estimated is True


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(BackendError, match="batch_size"):
        RecordingModel("x", {"batch_size": 0})
    with pytest.raises(BackendError, match="price_per_1m_tokens"):
        RecordingModel("x", {"price_per_1m_tokens": -1})
    with pytest.raises(BackendError, match="max_concurrency"):
        RecordingModel("x", {"max_concurrency": 0})
    with pytest.raises(BackendError, match="cannot be empty"):
        RecordingModel("  ")


def test_context_manager_loads_and_closes() -> None:
    class Closing(RecordingModel):
        provider_name = "closing"

        def __init__(self, model_name: str, options: Any = None) -> None:
            super().__init__(model_name, options)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    model = Closing("demo")
    with model as entered:
        assert entered is model
        assert model.loaded is True
    assert model.closed is True


def test_async_embedding_returns_lists() -> None:
    import asyncio

    model = RecordingModel("async", {"batch_size": 2})
    vectors = asyncio.run(model.embed_documents_async(["a", "b", "c"]))
    assert isinstance(vectors, list)
    assert all(isinstance(vector, list) for vector in vectors)
    assert len(vectors) == 3


def test_describe_and_cache_namespace() -> None:
    model = RecordingModel("demo", {"query_prefix": "q: "})
    described = model.describe()
    assert described["spec"] == "recording:demo"
    assert described["provider"] == "recording"
    assert "q='q: '" in model.cache_namespace
    assert model.cache_namespace.startswith("recording:demo:v")


# ---------------------------------------------------------------------------
# Registry, factory and plugins
# ---------------------------------------------------------------------------


def test_registry_and_factory() -> None:
    spec = parse_model_spec("fake:hash")
    assert (spec.backend, spec.model) == ("fake", "hash")
    providers = available_providers()
    for expected in ("hf", "ollama", "openai", "cohere", "voyage", "fake"):
        assert expected in providers

    model = create_backend("fake:hash", RunConfig(models=["fake:hash"]))
    model.load()
    vectors = model.embed_documents(["alpha beta", "gamma"])
    matrix = np.asarray(vectors)
    assert matrix.shape == (2, 128)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


@pytest.mark.parametrize("raw", ["no-colon", ":model", "backend:", ""])
def test_bad_model_specs(raw: str) -> None:
    with pytest.raises(BackendError, match="backend:model|Invalid model spec"):
        parse_model_spec(raw)


def test_unknown_provider_is_actionable() -> None:
    with pytest.raises(BackendError, match="Known providers"):
        create_model("does-not-exist:model")


def test_python_file_plugin_registers_provider(tmp_path: Path) -> None:
    plugin = tmp_path / "my_provider.py"
    plugin.write_text(
        "import numpy as np\n"
        "from embedbench.models import BaseEmbeddingModel, register_model\n\n"
        "@register_model('plugin_echo')\n"
        "class PluginEcho(BaseEmbeddingModel):\n"
        "    provider_name = 'plugin_echo'\n"
        "    def load(self): pass\n"
        "    def _embed_prepared(self, texts, kind='document'):\n"
        "        return np.ones((len(texts), 4), dtype=np.float32)\n",
        encoding="utf-8",
    )

    load_model_plugins([str(plugin)])
    model = create_model("plugin_echo:demo")
    model.load()
    assert len(model.embed_queries(["hello"])[0]) == 4
    assert "plugin_echo" in available_providers()


def test_provider_metadata_and_pricing_catalog() -> None:
    rows = {row["provider"]: row for row in provider_metadata()}
    assert rows["openai"]["api_key_env"] == "OPENAI_API_KEY"
    assert rows["openai"]["extra"] == "openai"
    assert rows["hf"]["extra"] == "local"
    assert rows["ollama"]["api_key_env"] is None

    prices = {
        (row["provider"], row["model"]): row["usd_per_1m_tokens"] for row in pricing_catalog()
    }
    assert prices[("openai", "text-embedding-3-small")] == 0.02
    assert prices[("voyage", "voyage-3-large")] == 0.18
    assert pricing_for("text-embedding-3-large", "openai") == 0.13
    assert pricing_for("text-embedding-3-large-2024-01-01") == 0.13
    assert pricing_for("unknown-model", "openai") is None


def test_legacy_backends_shim_still_works() -> None:
    from embedbench.backends import EmbeddingBackend, available_backends, register_backend

    assert EmbeddingBackend is BaseEmbeddingModel
    assert "fake" in available_backends()

    @register_backend("legacy_alias")
    class LegacyModel(BaseEmbeddingModel):
        provider_name = "legacy_alias"

        def load(self) -> None:
            pass

        def _embed_prepared(self, texts: list[str], kind: str = "document") -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32)

    model = create_model("legacy_alias:demo")
    assert len(model.embed_queries(["x"])[0]) == 2


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def test_ollama_model_missing_suggests_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(self: OllamaClient, model: str) -> None:
        raise OllamaError(f"Model '{model}' is not installed. Run: ollama pull {model}")

    monkeypatch.setattr(OllamaClient, "require_model", missing)
    model = OllamaEmbeddingModel("nomic-embed-text", {"host": "http://example.invalid"})
    with pytest.raises(BackendError, match="ollama pull nomic-embed-text"):
        model.load()


def test_ollama_model_embeds_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OllamaClient, "require_model", lambda self, model: None)
    monkeypatch.setattr(
        OllamaClient,
        "embed",
        lambda self, model, texts, keep_alive=None: [[1.0, 2.0, 3.0] for _ in texts],
    )
    model = OllamaEmbeddingModel(
        "nomic-embed-text", {"host": "http://example.invalid", "batch_size": 2}
    )
    model.load()
    matrix = model.embed_documents_matrix(["one", "two"])
    assert matrix.shape == (2, 3)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert model.price_per_1m_tokens == 0.0
    model.close()


# ---------------------------------------------------------------------------
# sentence-transformers
# ---------------------------------------------------------------------------


def test_hf_model_with_mocked_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def __call__(self, batch: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
            max_length = kwargs.get("max_length") or 10_000
            return {"input_ids": [[1] * min(len(text), max_length) for text in batch]}

    class FakeSentenceTransformer:
        last_instance: FakeSentenceTransformer | None = None

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

    model = SentenceTransformerEmbeddingModel(
        "fake/model",
        {"device": "cpu", "query_prefix": "Q: ", "document_prefix": "D: ", "batch_size": 4},
    )
    model.load()
    instance = FakeSentenceTransformer.last_instance
    assert instance is not None
    assert instance.kwargs["device"] == "cpu"
    assert model.dimension == 6

    matrix = model.embed_documents_matrix(["hello", "world"])
    assert matrix.shape == (2, 6)
    assert instance.encoded == [["D: hello", "D: world"]]
    assert model.embed_queries_matrix(["what?"]).shape == (1, 6)
    assert instance.encoded[-1] == ["Q: what?"]

    count = model.count_tokens(["abcd", "ab"])
    assert count.tokens == 6
    assert count.estimated is False
    assert count.method.startswith("tokenizer:")


def test_hf_e5_prefixes_are_applied_by_default() -> None:
    model = SentenceTransformerEmbeddingModel("intfloat/e5-base-v2", {"device": "cpu"})
    assert model.query_prefix == "query: "
    assert model.document_prefix == "passage: "
    assert "q='query: '" in model.cache_namespace

    other = SentenceTransformerEmbeddingModel("BAAI/bge-small-en-v1.5", {"device": "cpu"})
    assert other.query_prefix == ""
    assert other.document_prefix == ""


# ---------------------------------------------------------------------------
# Hosted providers (fake SDK modules, no network)
# ---------------------------------------------------------------------------


class _EmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _OpenAIResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_EmbeddingItem(index, vector) for index, vector in enumerate(vectors)]


def install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"calls": [], "failures": []}

    class Embeddings:
        def create(self, **params: Any) -> _OpenAIResponse:
            state["calls"].append(params)
            if state["failures"]:
                failure = state["failures"].pop(0)
                if failure is not None:
                    raise failure
            inputs = params["input"]
            vectors = [[float(len(text)), 2.0] for text in inputs]
            return _OpenAIResponse(vectors)

    class OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            state["client_kwargs"] = kwargs
            self.embeddings = Embeddings()

    module = types.ModuleType("openai")
    module.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return state


def test_openai_provider_batches_prices_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = install_fake_openai(monkeypatch)

    model = OpenAIEmbeddingModel(
        "text-embedding-3-small",
        {"batch_size": 2, "max_retries": 0, "max_input_tokens": 4},
    )
    model.load()
    assert state["client_kwargs"]["api_key"] == "test-key"
    assert model.price_per_1m_tokens == 0.02

    vectors = model.embed_documents([f"document {index}" for index in range(5)])
    assert len(vectors) == 5
    # batch_size=2 -> 3 provider calls, each input truncated to 4 tokens
    assert len(state["calls"]) == 3
    assert all(len(call["input"]) <= 2 for call in state["calls"])
    assert all(len(text.split()) <= 4 for call in state["calls"] for text in call["input"])

    estimate = model.cost_for_tokens(1_000_000)
    assert estimate.usd == pytest.approx(0.02)
    assert estimate.price_known is True


def test_openai_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    install_fake_openai(monkeypatch)
    model = OpenAIEmbeddingModel("text-embedding-3-small")
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        model.load()


def test_openai_provider_reports_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", None)
    model = OpenAIEmbeddingModel("text-embedding-3-small")
    with pytest.raises(BackendError, match="embedbench\\[openai\\]"):
        model.load()


def test_openai_provider_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = install_fake_openai(monkeypatch)

    class RateLimited(Exception):
        status_code = 429

    state["failures"] = [RateLimited("slow down")]
    model = OpenAIEmbeddingModel(
        "text-embedding-3-small", {"max_retries": 1, "retry_backoff_s": 0.0}
    )
    model.load()
    vectors = model.embed_queries(["hello"])
    assert len(vectors) == 1
    assert len(state["calls"]) == 2


def test_openai_provider_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = install_fake_openai(monkeypatch)

    class BadRequest(Exception):
        status_code = 400

    state["failures"] = [BadRequest("bad input")]
    model = OpenAIEmbeddingModel(
        "text-embedding-3-small", {"max_retries": 3, "retry_backoff_s": 0.0}
    )
    model.load()
    with pytest.raises(BackendError, match="bad input"):
        model.embed_queries(["hello"])
    assert len(state["calls"]) == 1


def test_is_retryable_classification() -> None:
    class WithStatus(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"status {status}")
            self.status_code = status

    assert is_retryable(WithStatus(429)) is True
    assert is_retryable(WithStatus(503)) is True
    assert is_retryable(WithStatus(404)) is False
    assert is_retryable(TimeoutError("timed out")) is True
    assert is_retryable(ValueError("bad value")) is False


def install_fake_cohere(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"calls": []}

    class Embeddings:
        def __init__(self, vectors: list[list[float]]) -> None:
            self.float = vectors

    class Response:
        def __init__(self, vectors: list[list[float]]) -> None:
            self.embeddings = Embeddings(vectors)

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            state["client_kwargs"] = kwargs

        def embed(self, **params: Any) -> Response:
            state["calls"].append(params)
            return Response([[1.0, 0.0] for _ in params["texts"]])

    module = types.ModuleType("cohere")
    module.Client = Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", module)
    return state


def test_cohere_provider_uses_input_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    state = install_fake_cohere(monkeypatch)
    model = CohereEmbeddingModel("embed-english-v3.0", {"batch_size": 4, "max_retries": 0})
    model.load()

    model.embed_documents(["a", "b"])
    model.embed_queries(["what?"])

    assert [call["input_type"] for call in state["calls"]] == ["search_document", "search_query"]
    assert model.price_per_1m_tokens == 0.10


def install_fake_voyage(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"calls": []}

    class Response:
        def __init__(self, vectors: list[list[float]]) -> None:
            self.embeddings = vectors

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            state["client_kwargs"] = kwargs

        def embed(self, **params: Any) -> Response:
            state["calls"].append(params)
            return Response([[0.0, 1.0] for _ in params["texts"]])

    module = types.ModuleType("voyageai")
    module.Client = Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voyageai", module)
    return state


def test_voyage_provider_uses_input_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    state = install_fake_voyage(monkeypatch)
    model = VoyageEmbeddingModel("voyage-3-large", {"batch_size": 4, "max_retries": 0})
    model.load()

    model.embed_documents(["a"])
    model.embed_queries(["what?"])

    assert [call["input_type"] for call in state["calls"]] == ["document", "query"]
    assert model.price_per_1m_tokens == 0.18
