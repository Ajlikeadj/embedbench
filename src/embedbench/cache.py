"""Persistent embedding cache.

Cache keys are ``(backend namespace, SHA-256(document text))``.  The namespace
contains the backend, model name and formatting prefixes so a config change
cannot return vectors produced under different instructions.  Vectors are stored
as float32 blobs; only hashes of source text are persisted.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from embedbench.logging_utils import get_logger
from embedbench.utils import sha256_hex, utc_now_iso

logger = get_logger(__name__)

_SCHEMA_VERSION = 1
_LOOKUP_CHUNK = 500


class EmbeddingCache:
    """SQLite-backed cache with graceful degradation on I/O errors."""

    def __init__(self, cache_dir: Path, *, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._connection: sqlite3.Connection | None = None
        self._failed = False
        if enabled:
            self._open()

    @property
    def db_path(self) -> Path:
        return self.cache_dir / "embeddings.sqlite3"

    def _open(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.db_path), timeout=30.0)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    namespace TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, text_sha256)
                )
                """
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            connection.commit()
            self._connection = connection
            logger.debug("Embedding cache ready at %s", self.db_path)
        except (sqlite3.Error, OSError) as exc:
            self.enabled = False
            self._failed = True
            logger.warning("Embedding cache disabled: %s", exc)

    def _disable_after_error(self, exc: Exception) -> None:
        self.enabled = False
        self._failed = True
        logger.warning("Embedding cache disabled after %s: %s", type(exc).__name__, exc)

    def get_many(self, namespace: str, texts: Sequence[str]) -> list[np.ndarray | None]:
        """Look up vectors for texts, preserving input order."""

        if not self.enabled or self._connection is None:
            self.misses += len(texts)
            return [None] * len(texts)
        if not texts:
            return []

        hashes = [sha256_hex(text) for text in texts]
        found: dict[str, np.ndarray] = {}
        try:
            for offset in range(0, len(hashes), _LOOKUP_CHUNK):
                chunk = hashes[offset : offset + _LOOKUP_CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._connection.execute(
                    f"SELECT text_sha256, dimension, vector FROM embeddings "
                    f"WHERE namespace = ? AND text_sha256 IN ({placeholders})",
                    (namespace, *chunk),
                ).fetchall()
                for text_hash, dimension, blob in rows:
                    vector = np.frombuffer(blob, dtype=np.float32).copy()
                    if vector.shape[0] != dimension:
                        logger.warning("Ignoring corrupt cache entry for %s", text_hash[:12])
                        continue
                    found[text_hash] = vector
        except sqlite3.Error as exc:
            self._disable_after_error(exc)
            self.misses += len(texts)
            return [None] * len(texts)

        results: list[np.ndarray | None] = []
        for text_hash in hashes:
            vector = found.get(text_hash)
            results.append(vector)
            if vector is None:
                self.misses += 1
            else:
                self.hits += 1
        return results

    def put_many(self, namespace: str, texts: Sequence[str], vectors: np.ndarray) -> None:
        """Store vectors aligned with ``texts``; failures disable the cache only."""

        if not self.enabled or self._connection is None or not texts:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            logger.warning(
                "Skipping cache write: %d texts but vector matrix shape %s",
                len(texts),
                matrix.shape,
            )
            return
        now = utc_now_iso()
        dimension = int(matrix.shape[1])
        rows = [
            (
                namespace,
                sha256_hex(text),
                dimension,
                np.ascontiguousarray(matrix[index], dtype=np.float32).tobytes(),
                now,
            )
            for index, text in enumerate(texts)
        ]
        try:
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings"
                "(namespace, text_sha256, dimension, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._connection.commit()
        except sqlite3.Error as exc:
            self._disable_after_error(exc)

    def close(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(sqlite3.Error):  # pragma: no cover - close errors are harmless
                self._connection.close()
            self._connection = None

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
