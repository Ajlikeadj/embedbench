"""Vector indexes: FAISS when available, NumPy exact search as a fallback."""

from __future__ import annotations

import io
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

from embedbench.errors import IndexBuildError
from embedbench.logging_utils import get_logger

logger = get_logger(__name__)

IndexBackend = Literal["auto", "faiss", "numpy"]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.size == 0:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class VectorIndex(ABC):
    """Minimal cosine-similarity index interface."""

    name: str

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def ntotal(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, row indices) sorted by descending similarity."""

        raise NotImplementedError

    @abstractmethod
    def disk_size_bytes(self) -> int:
        raise NotImplementedError


class FaissIndex(VectorIndex):
    """Exact inner-product index over L2-normalized vectors (cosine similarity)."""

    name = "faiss"

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        try:
            import faiss
        except Exception as exc:  # missing DLLs and unsupported platforms also matter
            raise ImportError(f"FAISS is unavailable: {exc}") from exc
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dimension)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise IndexBuildError(
                f"Expected vectors shaped (n, {self._dimension}), got {matrix.shape}"
            )
        if matrix.shape[0] == 0:
            return
        self._index.add(_normalize_rows(matrix))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if self.ntotal == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self._dimension:
            raise IndexBuildError(
                f"Query has dimension {vector.shape[1]}, index expects {self._dimension}"
            )
        count = min(k, self.ntotal)
        scores, indices = self._index.search(_normalize_rows(vector), count)
        scores = scores[0]
        indices = indices[0]
        keep = indices >= 0
        return scores[keep].astype(np.float32), indices[keep].astype(np.int64)

    def disk_size_bytes(self) -> int:
        with tempfile.TemporaryDirectory(prefix="embedbench-index-") as temp_dir:
            target = Path(temp_dir) / "index.faiss"
            self._faiss.write_index(self._index, str(target))
            return target.stat().st_size


class NumpyIndex(VectorIndex):
    """Exact NumPy fallback used when FAISS cannot be imported."""

    name = "numpy"

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._dimension = dimension
        self._vectors: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ntotal(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise IndexBuildError(
                f"Expected vectors shaped (n, {self._dimension}), got {matrix.shape}"
            )
        if matrix.shape[0] == 0:
            return
        normalized = _normalize_rows(matrix)
        if self._vectors is None:
            self._vectors = normalized
        else:
            self._vectors = np.ascontiguousarray(
                np.vstack([self._vectors, normalized]), dtype=np.float32
            )

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if self._vectors is None or self._vectors.shape[0] == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        vector = _normalize_rows(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        if vector.shape[0] != self._dimension:
            raise IndexBuildError(
                f"Query has dimension {vector.shape[0]}, index expects {self._dimension}"
            )
        similarities = self._vectors @ vector
        count = min(k, self._vectors.shape[0])
        if count < similarities.shape[0]:
            candidates = np.argpartition(-similarities, count - 1)[:count]
            order = candidates[np.argsort(-similarities[candidates], kind="stable")]
        else:
            order = np.argsort(-similarities, kind="stable")
        order = order[:count]
        return similarities[order].astype(np.float32), order.astype(np.int64)

    def disk_size_bytes(self) -> int:
        if self._vectors is None:
            return 0
        buffer = io.BytesIO()
        np.save(buffer, self._vectors, allow_pickle=False)
        return int(buffer.tell())


def build_index(vectors: np.ndarray, backend: IndexBackend = "auto") -> VectorIndex:
    """Build an exact cosine index, falling back to NumPy if FAISS is unavailable."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise IndexBuildError(
            f"Expected a non-empty 2-D embedding matrix, got shape {matrix.shape}"
        )
    dimension = int(matrix.shape[1])
    if backend not in {"auto", "faiss", "numpy"}:
        raise IndexBuildError(f"Unknown index backend: {backend!r}")

    index: VectorIndex
    if backend in {"auto", "faiss"}:
        try:
            index = FaissIndex(dimension)
        except ImportError as exc:
            if backend == "faiss":
                raise IndexBuildError(str(exc)) from exc
            logger.warning("FAISS unavailable (%s); using slower NumPy exact search", exc)
            index = NumpyIndex(dimension)
    else:
        index = NumpyIndex(dimension)

    index.add(matrix)
    logger.debug("Built %s index with %d vectors of dimension %d", index.name, index.ntotal, dimension)
    return index
