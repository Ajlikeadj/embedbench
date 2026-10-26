"""The embedding-model contract shared by every provider.

:class:`BaseEmbeddingModel` is the single interface the rest of embedbench
builds on.  It owns batching, instruction prefixes, validation, normalisation
and cost metadata so provider implementations only translate text into
vectors.

A new provider needs three things:

1. a class inheriting :class:`BaseEmbeddingModel` with ``provider_name`` set,
2. an implementation of :meth:`_embed_prepared` returning a 2-D float array,
3. registration with :func:`embedbench.models.register_model`.

Public methods return ``list[list[float]]`` (the stable, JSON-friendly API).
Internal call sites that need arithmetic use the ``*_matrix`` variants, which
return ``numpy.ndarray`` and avoid an extra conversion.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from embedbench.errors import BackendError
from embedbench.schemas import TokenCount

CACHE_FORMAT_VERSION = "2"
"""Bumped whenever embeddings for the same text would change."""

DEFAULT_BATCH_SIZE = 32
"""Chunks embedded per provider call when the caller does not override it."""

TextKind = Literal["document", "query"]
"""Whether a batch contains documents or queries; some providers need to know."""

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def heuristic_token_count(texts: Sequence[str]) -> int:
    """Approximate tokens as words plus punctuation marks.

    Args:
        texts: Input strings.

    Returns:
        Estimated token count.  Accurate to roughly +/-15% for English prose.
    """

    return sum(len(_TOKEN_RE.findall(text)) for text in texts)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving zero vectors as zeros.

    Args:
        matrix: A 1-D vector or 2-D matrix of embeddings.

    Returns:
        A contiguous float32 array with unit-norm rows where possible.
    """

    array = np.ascontiguousarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.size == 0:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class BaseEmbeddingModel(ABC):
    """Abstract embedding model with batching, prefixes and pricing metadata.

    Args:
        model_name: Provider-specific model identifier, for example
            ``text-embedding-3-small`` or ``BAAI/bge-small-en-v1.5``.
        options: Provider options. Recognised keys include ``batch_size``,
            ``query_prefix``, ``document_prefix``, ``normalize``,
            ``timeout_s``, ``max_retries`` and ``price_per_1m_tokens``.
            Each provider documents its own extra keys.

    Raises:
        BackendError: If ``model_name`` is empty or an option is invalid.
    """

    provider_name: ClassVar[str] = "base"
    """Registry name; must be unique and free of colons."""

    requires_extra: ClassVar[str | None] = None
    """Optional ``pip install embedbench[...]`` extra needed at runtime."""

    default_price_usd_per_1m: ClassVar[float | None] = None
    """Fallback price when the provider has no per-model price entry."""

    def __init__(
        self,
        model_name: str,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        name = model_name.strip()
        if not name:
            raise BackendError("Embedding model name cannot be empty")
        self.model_name = name
        self.options: dict[str, Any] = dict(options or {})

        try:
            self.batch_size = int(self.options.get("batch_size", DEFAULT_BATCH_SIZE))
        except (TypeError, ValueError) as exc:
            raise BackendError("batch_size must be an integer") from exc
        if self.batch_size < 1:
            raise BackendError("batch_size must be >= 1")

        try:
            self.max_concurrency = int(self.options.get("max_concurrency", 1))
        except (TypeError, ValueError) as exc:
            raise BackendError("max_concurrency must be an integer") from exc
        if self.max_concurrency < 1:
            raise BackendError("max_concurrency must be >= 1")

        query_prefix = self.options.get("query_prefix")
        document_prefix = self.options.get("document_prefix")
        self.query_prefix = str(query_prefix) if query_prefix else ""
        self.document_prefix = str(document_prefix) if document_prefix else ""
        self.normalize = bool(self.options.get("normalize", True))

        price = self.options.get("price_per_1m_tokens", self.default_price_usd_per_1m)
        if price is not None:
            try:
                price = float(price)
            except (TypeError, ValueError) as exc:
                raise BackendError("price_per_1m_tokens must be a number") from exc
            if price < 0:
                raise BackendError("price_per_1m_tokens must be >= 0")
        self._price_per_1m_tokens: float | None = price
        self._dimension: int | None = None

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        """``provider:model`` identifier used in reports and cache keys."""

        return f"{self.provider_name}:{self.model_name}"

    @property
    def dimension(self) -> int:
        """Embedding width.

        Raises:
            BackendError: If the width is unknown before loading or embedding.
        """

        if self._dimension is None:
            raise BackendError(
                f"Embedding dimension for {self.name} is unknown until load/first embed"
            )
        return self._dimension

    @property
    def known_dimension(self) -> int | None:
        """Embedding width when it is already known, else ``None``."""

        return self._dimension

    @property
    def max_batch_size(self) -> int:
        """Maximum number of texts sent to the provider in one call."""

        return self.batch_size

    @property
    def cache_namespace(self) -> str:
        """Identity used for embedding-cache keys.

        Prefixes, normalisation and the format version are part of the key so
        changing instructions invalidates cached vectors.
        """

        return (
            f"{self.provider_name}:{self.model_name}:v{CACHE_FORMAT_VERSION}"
            f":q={self.query_prefix!r}:d={self.document_prefix!r}"
            f":norm={self.normalize}"
        )

    @property
    def price_per_1m_tokens(self) -> float | None:
        """USD per million tokens, ``0.0`` for local models, ``None`` if unknown."""

        return self._price_per_1m_tokens

    # -- prompt preparation ----------------------------------------------

    def prepare_documents(self, texts: Sequence[str]) -> list[str]:
        """Apply the document instruction prefix.

        Args:
            texts: Raw document texts.

        Returns:
            Texts ready to send to the provider.
        """

        return [f"{self.document_prefix}{text}" for text in texts]

    def prepare_queries(self, texts: Sequence[str]) -> list[str]:
        """Apply the query instruction prefix.

        Args:
            texts: Raw query texts.

        Returns:
            Texts ready to send to the provider.
        """

        return [f"{self.query_prefix}{text}" for text in texts]

    # -- lifecycle --------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load the model or verify that a remote endpoint can serve it.

        Implementations must be idempotent: calling ``load`` twice is harmless.
        """

        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Release resources.  Safe to call multiple times."""

    def reset_peak_memory(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Reset provider-specific GPU peak counters (no-op by default)."""

    def runtime_metadata(self) -> dict[str, Any]:
        """Return device/VRAM/endpoint details for the report.

        Returns:
            A JSON-friendly mapping, empty by default.
        """

        return {}

    def __enter__(self) -> BaseEmbeddingModel:
        self.load()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    # -- embedding --------------------------------------------------------

    @abstractmethod
    def _embed_prepared(self, texts: list[str], kind: TextKind = "document") -> np.ndarray:
        """Embed already-prefixed, non-empty texts.

        Args:
            texts: Non-empty strings, already prefix-applied.
            kind: ``"document"`` or ``"query"``; providers such as Cohere and
                Voyage need this to pick the right embedding mode.

        Returns:
            A 2-D array with one row per input.
        """

        raise NotImplementedError

    @staticmethod
    def _check_texts(texts: Sequence[str]) -> None:
        for text in texts:
            if not isinstance(text, str):
                raise BackendError("Embedding inputs must be strings")
            if not text.strip():
                raise BackendError("Embedding inputs cannot be empty")

    def _validated(self, vectors: np.ndarray, expected_rows: int) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise BackendError(
                f"{self.name} returned embeddings with shape {array.shape}, expected 2-D"
            )
        if array.shape[0] != expected_rows:
            raise BackendError(
                f"{self.name} returned {array.shape[0]} embeddings for {expected_rows} inputs"
            )
        if array.shape[1] < 1:
            raise BackendError(f"{self.name} returned zero-dimensional embeddings")
        if not np.isfinite(array).all():
            raise BackendError(f"{self.name} returned non-finite embedding values")
        if self._dimension is not None and array.shape[1] != self._dimension:
            raise BackendError(
                f"{self.name} changed dimension from {self._dimension} to "
                f"{array.shape[1]} between calls"
            )
        self._dimension = int(array.shape[1])
        return normalize_rows(array) if self.normalize else np.ascontiguousarray(array)

    def _embed_with_batches(self, prepared: Sequence[str], kind: TextKind) -> np.ndarray:
        """Embed texts in ``max_batch_size`` chunks and concatenate the result."""

        batches = [
            list(prepared[offset : offset + self.max_batch_size])
            for offset in range(0, len(prepared), self.max_batch_size)
        ]
        matrices: list[np.ndarray] = []
        for batch in batches:
            vectors = self._embed_prepared(batch, kind)
            matrices.append(self._validated(vectors, len(batch)))
        if not matrices:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        if len(matrices) == 1:
            return matrices[0]
        return np.ascontiguousarray(np.vstack(matrices), dtype=np.float32)

    def embed_documents_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents and return an ``(n, dimension)`` matrix.

        Args:
            texts: Document texts; may be empty.

        Returns:
            Float32 matrix with one row per document.

        Raises:
            BackendError: If any input is invalid or the provider misbehaves.
        """

        return self._embed_matrix(texts, "document")

    def embed_queries_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Embed queries and return an ``(n, dimension)`` matrix.

        Args:
            texts: Query texts; may be empty.

        Returns:
            Float32 matrix with one row per query.

        Raises:
            BackendError: If any input is invalid or the provider misbehaves.
        """

        return self._embed_matrix(texts, "query")

    def _embed_matrix(
        self,
        texts: Sequence[str],
        kind: TextKind,
    ) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        self._check_texts(texts)
        prepared = (
            self.prepare_documents(texts) if kind == "document" else self.prepare_queries(texts)
        )
        return self._embed_with_batches(prepared, kind)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks.

        Args:
            texts: Document texts.

        Returns:
            One embedding per document, each a list of floats.
        """

        return self.embed_documents_matrix(texts).tolist()

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed search queries.

        Args:
            texts: Query texts.

        Returns:
            One embedding per query, each a list of floats.
        """

        return self.embed_queries_matrix(texts).tolist()

    async def embed_documents_async(self, texts: Sequence[str]) -> list[list[float]]:
        """Async wrapper around :meth:`embed_documents`.

        Providers that can batch concurrently override
        :meth:`_embed_async`; the default keeps the event loop free by running
        the synchronous implementation in a worker thread.

        Args:
            texts: Document texts.

        Returns:
            One embedding per document.
        """

        return await self._embed_async(texts, "document")

    async def embed_queries_async(self, texts: Sequence[str]) -> list[list[float]]:
        """Async wrapper around :meth:`embed_queries`.

        Args:
            texts: Query texts.

        Returns:
            One embedding per query.
        """

        return await self._embed_async(texts, "query")

    async def _embed_async(
        self,
        texts: Sequence[str],
        kind: TextKind,
    ) -> list[list[float]]:
        if len(texts) == 0:
            return []
        self._check_texts(texts)
        prepared = (
            self.prepare_documents(texts) if kind == "document" else self.prepare_queries(texts)
        )
        batches = [
            prepared[offset : offset + self.max_batch_size]
            for offset in range(0, len(prepared), self.max_batch_size)
        ]
        import asyncio

        semaphore = asyncio.Semaphore(self.max_concurrency)
        loop = asyncio.get_running_loop()

        async def run_batch(batch: list[str]) -> np.ndarray:
            async with semaphore:
                return await loop.run_in_executor(None, self._embed_prepared, batch, kind)

        matrices = await asyncio.gather(*(run_batch(batch) for batch in batches))
        arrays = [
            self._validated(matrix, len(batch))
            for matrix, batch in zip(matrices, batches, strict=True)
        ]
        return np.vstack(arrays).tolist()

    # -- usage and cost ---------------------------------------------------

    def count_tokens(self, texts: Sequence[str]) -> TokenCount:
        """Estimate token usage.

        Providers with a real tokenizer (sentence-transformers, OpenAI via
        ``tiktoken``) override this.

        Args:
            texts: Texts that were or will be sent to the provider.

        Returns:
            Token count plus whether it is an estimate.
        """

        return TokenCount(
            tokens=heuristic_token_count(texts),
            estimated=True,
            method="heuristic (words + punctuation)",
        )

    def cost_for_tokens(self, tokens: int) -> Any:
        """Convert a token count into a dollar estimate.

        Args:
            tokens: Total billed tokens.

        Returns:
            A :class:`embedbench.metrics.CostEstimate`.
        """

        from embedbench.metrics import estimate_cost

        return estimate_cost(tokens, self.price_per_1m_tokens)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly description used by ``embedbench providers``.

        Returns:
            Provider, model, dimension, price and extra-package details.
        """

        return {
            "spec": self.name,
            "provider": self.provider_name,
            "model": self.model_name,
            "dimension": self.known_dimension,
            "price_per_1m_tokens": self.price_per_1m_tokens,
            "requires_extra": self.requires_extra,
        }


# Backwards-compatible alias: 0.1.x called this class ``EmbeddingBackend``.
EmbeddingBackend = BaseEmbeddingModel

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_BATCH_SIZE",
    "BaseEmbeddingModel",
    "EmbeddingBackend",
    "heuristic_token_count",
    "normalize_rows",
]
