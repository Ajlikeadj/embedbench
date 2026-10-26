"""Deprecated: embedding providers moved to :mod:`embedbench.models`.

This package exists so 0.1.x code (``from embedbench.backends import
register_backend``) and third-party plugins keep working.  New code should
import from :mod:`embedbench.models`.
"""

from __future__ import annotations

from embedbench.models import (
    CACHE_FORMAT_VERSION,
    BaseEmbeddingModel,
    EmbeddingBackend,
    TextKind,
    available_backends,
    available_providers,
    create_backend,
    create_model,
    ensure_backends_loaded,
    ensure_models_loaded,
    heuristic_token_count,
    load_model_plugins,
    load_plugins,
    normalize_rows,
    parse_model_spec,
    register_backend,
    register_model,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "BaseEmbeddingModel",
    "EmbeddingBackend",
    "TextKind",
    "available_backends",
    "available_providers",
    "create_backend",
    "create_model",
    "ensure_backends_loaded",
    "ensure_models_loaded",
    "heuristic_token_count",
    "load_model_plugins",
    "load_plugins",
    "normalize_rows",
    "parse_model_spec",
    "register_backend",
    "register_model",
]
