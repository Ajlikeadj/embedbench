"""Embedding backend registry and built-in providers."""

from __future__ import annotations

from embedbench.backends.base import EmbeddingBackend, heuristic_token_count, normalize_rows
from embedbench.backends.registry import (
    available_backends,
    create_backend,
    ensure_backends_loaded,
    load_plugins,
    parse_model_spec,
    register_backend,
)

# Built-in backend modules register themselves lazily via ensure_backends_loaded().
# Heavy optional libraries stay lazy inside each backend's load() method.

__all__ = [
    "EmbeddingBackend",
    "available_backends",
    "create_backend",
    "ensure_backends_loaded",
    "heuristic_token_count",
    "load_plugins",
    "normalize_rows",
    "parse_model_spec",
    "register_backend",
]
