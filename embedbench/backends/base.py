"""Abstract embedding backend interface.

A new backend only needs to inherit from :class:`EmbeddingBackend`, implement
``load`` and ``_embed_prepared``, and register its class with
``embedbench.backends.register_backend``.  Batching, prefixing, validation and
normalization are handled here.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from embedbench.errors import BackendError
from embedbench.models import TokenCount

CACHE_FORMAT_VERSION = "1"
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def heuristic_token_count(texts: Sequence[str]) -> int:
    """Approximate tokens as words plus punctuation marks."""

    return sum(len(_TOKEN_RE.findall(text)) for text in texts)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows, leaving zero vectors as zeros."""

    array = np.ascontiguousarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.size == 0:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class EmbeddingBackend(ABC):
    """Common contract for all embedding providers."""

    backend_name: ClassVar[str] = "base"

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
            self.batch_size = int(self.options.get("batch_size", 32))
        except (TypeError, ValueError) as exc:
            raise BackendError("batch_size must be an integer") from exc
        if self.batch_size < 1:
            raise BackendError("batch_size must be >= 1")

        query_prefix = self.options.get("query_prefix")
        document_prefix = self.options.get("document_prefix")
        self.query_prefix = str(query_prefix) if query_prefix else ""
        self.document_prefix = str(document_prefix) if document_prefix else ""
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return f"{self.backend_name}:{self.model_name}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise BackendError(
                f"Embedding dimension for {self.name} is unknown until load/first embed"
            )
        return self._dimension

    @property
    def known_dimension(self) -> int | None:
        """Dimension when the backend can determine it at load time, else None."""

        return self._dimension

    @property
    def max_batch_size(self) -> int:
        return self.batch_size

    @property
    def cache_namespace(self) -> str:
        """Identity used for embedding-cache keys.

        Prefixes and the format version are included so changing instructions
        invalidates cached vectors.
        """

        return (
            f"{self.backend_name}:{self.model_name}:v{CACHE_FORMAT_VERSION}"
            f":q={self.query_prefix!r}:d={self.document_prefix!r}"
        )

    def prepare_documents(self, texts: Sequence[str]) -> list[str]:
        return [f"{self.document_prefix}{text}" for text in texts]

    def prepare_queries(self, texts: Sequence[str]) -> list[str]:
        return [f"{self.query_prefix}{text}" for text in texts]

    @abstractmethod
    def load(self) -> None:
        """Load the model or verify that a remote server can serve it."""

        raise NotImplementedError

    @abstractmethod
    def _embed_prepared(self, texts: list[str]) -> np.ndarray:
        """Embed already-prefixed, non-empty texts and return a 2-D array."""

        raise NotImplementedError

    def _validated(self, vectors: np.ndarray, expected_rows: int) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise BackendError(
                f"{self.name} returned embeddings with shape {array.shape}, expected 2-D"
            )
        if array.shape[0] != expected_rows:
            raise BackendError(
                f"{self.name} returned {array.shape[0]} embeddings for "
                f"{expected_rows} inputs"
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
        return normalize_rows(array)

    @staticmethod
    def _check_texts(texts: Sequence[str]) -> None:
        for text in texts:
            if not isinstance(text, str):
                raise BackendError("Embedding inputs must be strings")
            if not text.strip():
                raise BackendError("Embedding inputs cannot be empty")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed document chunks, applying any document prefix."""

        if not texts:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        self._check_texts(texts)
        prepared = self.prepare_documents(texts)
        vectors = self._embed_prepared(prepared)
        return self._validated(vectors, len(prepared))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed search queries, applying any query prefix."""

        if not texts:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        self._check_texts(texts)
        prepared = self.prepare_queries(texts)
        vectors = self._embed_prepared(prepared)
        return self._validated(vectors, len(prepared))

    def count_tokens(self, texts: Sequence[str]) -> TokenCount:
        """Estimate tokens.  Backends with a local tokenizer should override."""

        return TokenCount(
            tokens=heuristic_token_count(texts),
            estimated=True,
            method="heuristic (words + punctuation)",
        )

    def reset_peak_memory(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Reset any backend-specific GPU peak counters (no-op by default)."""

    def runtime_metadata(self) -> dict[str, Any]:
        """Return backend-specific device/VRAM/server details for the report."""

        return {}

    def close(self) -> None:  # noqa: B027 - optional lifecycle hook
        """Release resources.  Safe to call multiple times."""

    def __enter__(self) -> "EmbeddingBackend":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
