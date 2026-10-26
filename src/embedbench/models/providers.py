"""Hosted embedding providers: OpenAI, Cohere and Voyage AI.

Cloud SDKs are optional dependencies.  Importing this module never imports
them; the error message raised at ``load()`` names the exact extra to install,
for example ``pip install "embedbench[openai]"``.

All three providers share :class:`HTTPProviderModel`, which adds the behaviour
that matters for benchmarking hosted APIs: request chunking under provider
limits, bounded exponential backoff on transient failures, concurrent batches
in the async path, and per-model list pricing.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, TypeVar

import numpy as np

from embedbench.errors import BackendError
from embedbench.logging_utils import get_logger
from embedbench.models.base import BaseEmbeddingModel, TextKind, heuristic_token_count
from embedbench.schemas import TokenCount

logger = get_logger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

MODEL_PRICING: dict[str, float] = {
    # USD per 1M tokens, provider list prices.  Indicative values used for
    # cost estimates; override with the model option ``price_per_1m_tokens``.
    "openai:text-embedding-3-small": 0.02,
    "openai:text-embedding-3-large": 0.13,
    "openai:text-embedding-ada-002": 0.10,
    "cohere:embed-english-v3.0": 0.10,
    "cohere:embed-multilingual-v3.0": 0.10,
    "cohere:embed-english-light-v3.0": 0.10,
    "cohere:embed-multilingual-light-v3.0": 0.10,
    "cohere:embed-v4.0": 0.12,
    "voyage:voyage-3-large": 0.18,
    "voyage:voyage-3.5-large": 0.18,
    "voyage:voyage-3.5": 0.06,
    "voyage:voyage-3": 0.06,
    "voyage:voyage-3.5-lite": 0.02,
    "voyage:voyage-3-lite": 0.02,
    "voyage:voyage-code-3": 0.18,
    "voyage:voyage-large-2": 0.12,
    "voyage:voyage-2": 0.12,
    "voyage:voyage-law-2": 0.12,
}

PRICING_NOTE = (
    "Prices are provider list prices in USD per 1M tokens, cached at release time. "
    "Override with price_per_1m_tokens when your contract differs."
)


def pricing_for(model: str, provider: str | None = None) -> float | None:
    """Look up list pricing for a hosted model.

    Args:
        model: Model name, with or without a ``provider:`` prefix.
        provider: Provider name used when ``model`` has no prefix.

    Returns:
        USD per million tokens, or ``None`` when the model is not priced and no
        provider default applies.
    """

    key = model if ":" in model else f"{provider}:{model}" if provider else model
    if key in MODEL_PRICING:
        return MODEL_PRICING[key]

    if ":" in model:
        provider, _, model = model.partition(":")
    candidates: list[tuple[int, float]] = []
    for spec, price in MODEL_PRICING.items():
        known_provider, _, known_model = spec.partition(":")
        if provider and provider != known_provider:
            continue
        # Longest-prefix match so date-suffixed aliases still resolve.
        if model == known_model or model.startswith(known_model) or known_model.startswith(model):
            candidates.append((len(known_model), price))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "http_status", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(exc: BaseException) -> bool:
    """Return whether an exception from a hosted API is worth retrying.

    Args:
        exc: The exception raised by the provider SDK or HTTP layer.

    Returns:
        ``True`` for rate limits, timeouts, connection errors and 5xx codes.
    """

    status = _status_code(exc)
    if status is not None:
        return status in RETRYABLE_STATUS_CODES
    lowered = f"{type(exc).__name__}: {exc}".lower()
    markers = ("timeout", "timed out", "connection", "temporarily", "overloaded", "rate limit")
    return any(marker in lowered for marker in markers)


class HTTPProviderModel(BaseEmbeddingModel):
    """Base class for hosted embedding APIs.

    Subclasses implement :meth:`_embed_batch` and :meth:`_make_client`.

    Options:
        api_key: API key; defaults to ``env_api_key`` from the environment.
        base_url: Override the API endpoint (also enables OpenAI-compatible
            servers such as vLLM or LM Studio).
        timeout_s: Per-request timeout in seconds.
        max_retries: Retries for transient failures (default 3).
        retry_backoff_s: Base delay for exponential backoff (default 0.75).
        max_concurrency: Concurrent batches in the async path (default 4).
        dimensions: Optional output dimension reduction (provider dependent).
    """

    env_api_key: ClassVar[str] = ""
    requires_extra: ClassVar[str | None] = None
    max_inputs_per_request: ClassVar[int] = 256
    default_price_usd_per_1m: ClassVar[float | None] = None

    def __init__(self, model_name: str, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        if self._price_per_1m_tokens is None:
            self._price_per_1m_tokens = pricing_for(self.model_name, self.provider_name)

        self.api_key = str(self.options.get("api_key") or os.environ.get(self.env_api_key, ""))
        self.base_url = self.options.get("base_url")
        self.timeout_s = float(self.options.get("timeout_s", 60.0))
        self.max_retries = int(self.options.get("max_retries", 3))
        self.retry_backoff_s = float(self.options.get("retry_backoff_s", 0.75))
        self.dimensions = self.options.get("dimensions")
        self.max_input_tokens = self.options.get("max_input_tokens")
        self.max_concurrency = int(self.options.get("max_concurrency", 4))
        self._client: Any | None = None
        self._tokenizer: Any | None = None

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """Validate credentials and construct the SDK client.

        Raises:
            BackendError: If the API key is missing or the SDK is not installed.
        """

        if self._client is not None:
            return
        if not self.api_key:
            raise BackendError(
                f"{self.provider_name} requires an API key. Set the {self.env_api_key} "
                f"environment variable, pass options={{'api_key': ...}}, or choose a local "
                "provider."
            )
        self._client = self._make_client()
        logger.debug(
            "%s client ready for %s (batch=%d, concurrency=%d)",
            self.provider_name,
            self.model_name,
            self.max_batch_size,
            self.max_concurrency,
        )

    def _make_client(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        self._client = None

    # -- embedding --------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is None:
            raise BackendError(f"{self.name} used before load()")
        return self._client

    def _embed_prepared(self, texts: list[str], kind: TextKind = "document") -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        chunks = [
            texts[offset : offset + self.max_inputs_per_request]
            for offset in range(0, len(texts), self.max_inputs_per_request)
        ]
        vectors: list[list[float]] = []
        for chunk in chunks:
            vectors.extend(self._embed_batch(chunk, kind))
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape[0] != len(texts):
            raise BackendError(
                f"{self.name} returned {array.shape[0]} embeddings for {len(texts)} inputs"
            )
        return array

    def _embed_batch(self, texts: list[str], kind: TextKind) -> list[list[float]]:
        raise NotImplementedError

    def _call_with_retries(self, call: Callable[[], T], *, label: str) -> T:
        attempt = 0
        while True:
            try:
                return call()
            except BackendError:
                raise
            except Exception as exc:
                if attempt >= self.max_retries or not is_retryable(exc):
                    raise BackendError(f"{self.name} {label} failed: {exc}") from exc
                delay = self.retry_backoff_s * (2**attempt) + random.uniform(0, 0.1)
                attempt += 1
                logger.warning(
                    "%s %s failed (%s); retry %d/%d in %.2fs",
                    self.name,
                    label,
                    exc,
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)

    def _truncate(self, texts: Sequence[str]) -> list[str]:
        """Truncate inputs to the provider token limit when one is configured."""

        if not self.max_input_tokens:
            return list(texts)
        encoder = self._get_tokenizer()
        if encoder is None:
            max_chars = int(self.max_input_tokens) * 4
            return [text[:max_chars] for text in texts]
        limit = int(self.max_input_tokens)
        truncated: list[str] = []
        for text in texts:
            tokens = encoder.encode(text, disallowed_special=())
            if len(tokens) <= limit:
                truncated.append(text)
            else:
                truncated.append(encoder.decode(tokens[:limit]))
        return truncated

    def _get_tokenizer(self) -> Any | None:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            import tiktoken
        except ImportError:
            return None
        try:
            self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    # -- usage ------------------------------------------------------------

    def count_tokens(self, texts: Sequence[str]) -> TokenCount:
        """Count tokens with ``tiktoken`` when available, else heuristically."""

        encoder = self._get_tokenizer()
        if encoder is None:
            return TokenCount(
                tokens=heuristic_token_count(texts),
                estimated=True,
                method="heuristic (tiktoken not installed)",
            )
        total = 0
        for text in texts:
            total += len(encoder.encode(text, disallowed_special=()))
        return TokenCount(
            tokens=total,
            estimated=False,
            method=f"tiktoken:{self.model_name}",
        )


class OpenAIEmbeddingModel(HTTPProviderModel):
    """OpenAI ``text-embedding-3-*`` models, or any OpenAI-compatible server.

    Set ``base_url`` to point at a compatible endpoint (vLLM, LM Studio,
    Azure OpenAI gateway).  Install the SDK with
    ``pip install "embedbench[openai]"``.

    Example:
        >>> model = OpenAIEmbeddingModel("text-embedding-3-small")  # doctest: +SKIP
        >>> model.load()  # doctest: +SKIP
        >>> len(model.embed_documents(["a chunk of text"])[0])  # doctest: +SKIP
        1536
    """

    provider_name = "openai"
    env_api_key = "OPENAI_API_KEY"
    requires_extra = "openai"
    max_inputs_per_request = 2048

    def __init__(self, model_name: str, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        if self.max_input_tokens is None:
            self.max_input_tokens = 8191

    def _make_client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise BackendError(
                "The OpenAI provider requires the openai SDK. "
                'Install it with: pip install "embedbench[openai]"'
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_s,
            # embedbench retries explicitly so backoff is observable and consistent.
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = str(self.base_url)
        return openai.OpenAI(**kwargs)

    def _embed_batch(self, texts: list[str], kind: TextKind) -> list[list[float]]:
        client = self._ensure_client()
        payload = self._truncate(texts)
        params: dict[str, Any] = {
            "model": self.model_name,
            "input": payload,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            params["dimensions"] = int(self.dimensions)
        response = self._call_with_retries(
            lambda: client.embeddings.create(**params), label="embeddings.create"
        )
        data = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]


class CohereEmbeddingModel(HTTPProviderModel):
    """Cohere ``embed-*-v3.0`` models with query/document input types.

    Install the SDK with ``pip install "embedbench[cohere]"``.  Options accept
    ``document_input_type`` and ``query_input_type`` to override the defaults
    (``search_document`` / ``search_query``).
    """

    provider_name = "cohere"
    env_api_key = "COHERE_API_KEY"
    requires_extra = "cohere"
    max_inputs_per_request = 96

    def __init__(self, model_name: str, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self.document_input_type = str(self.options.get("document_input_type", "search_document"))
        self.query_input_type = str(self.options.get("query_input_type", "search_query"))
        self.truncate = str(self.options.get("truncate", "END"))

    def _make_client(self) -> Any:
        try:
            import cohere
        except ImportError as exc:
            raise BackendError(
                "The Cohere provider requires the cohere SDK. "
                'Install it with: pip install "embedbench[cohere]"'
            ) from exc
        return cohere.Client(api_key=self.api_key, timeout=self.timeout_s)

    def _embed_batch(self, texts: list[str], kind: TextKind) -> list[list[float]]:
        client = self._ensure_client()
        input_type = self.query_input_type if kind == "query" else self.document_input_type
        response = self._call_with_retries(
            lambda: client.embed(
                texts=texts,
                model=self.model_name,
                input_type=input_type,
                truncate=self.truncate,
                embedding_types=["float"],
            ),
            label="embed",
        )
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise BackendError("Cohere response contained no embeddings")
        # cohere>=5 returns an object with a `.float` attribute; older clients
        # return a plain list of vectors.
        vectors = getattr(embeddings, "float", embeddings)
        if callable(vectors):
            vectors = vectors()
        return [list(vector) for vector in vectors]


class VoyageEmbeddingModel(HTTPProviderModel):
    """Voyage AI models such as ``voyage-3-large`` and ``voyage-code-3``.

    Install the SDK with ``pip install "embedbench[voyage]"``.
    """

    provider_name = "voyage"
    env_api_key = "VOYAGE_API_KEY"
    requires_extra = "voyage"
    max_inputs_per_request = 128

    def _make_client(self) -> Any:
        try:
            import voyageai
        except ImportError as exc:
            raise BackendError(
                "The Voyage AI provider requires the voyageai SDK. "
                'Install it with: pip install "embedbench[voyage]"'
            ) from exc
        kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout_s}
        if self.base_url:
            kwargs["base_url"] = str(self.base_url)
        return voyageai.Client(**kwargs)

    def _embed_batch(self, texts: list[str], kind: TextKind) -> list[list[float]]:
        client = self._ensure_client()
        input_type = "query" if kind == "query" else "document"
        response = self._call_with_retries(
            lambda: client.embed(
                texts=texts,
                model=self.model_name,
                input_type=input_type,
                truncation=True,
            ),
            label="embed",
        )
        vectors = getattr(response, "embeddings", None)
        if vectors is None:
            raise BackendError("Voyage response contained no embeddings")
        return [list(vector) for vector in vectors]


__all__ = [
    "MODEL_PRICING",
    "PRICING_NOTE",
    "RETRYABLE_STATUS_CODES",
    "CohereEmbeddingModel",
    "HTTPProviderModel",
    "OpenAIEmbeddingModel",
    "VoyageEmbeddingModel",
    "is_retryable",
    "pricing_for",
]
