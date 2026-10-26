"""Deprecated: use :mod:`embedbench.models.base` instead."""

from __future__ import annotations

from embedbench.models.base import (
    CACHE_FORMAT_VERSION,
    DEFAULT_BATCH_SIZE,
    BaseEmbeddingModel,
    EmbeddingBackend,
    TextKind,
    heuristic_token_count,
    normalize_rows,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_BATCH_SIZE",
    "BaseEmbeddingModel",
    "EmbeddingBackend",
    "TextKind",
    "heuristic_token_count",
    "normalize_rows",
]
