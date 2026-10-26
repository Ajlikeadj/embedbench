"""Deprecated: use :mod:`embedbench.models` instead."""

from __future__ import annotations

from embedbench.models import (
    available_backends,
    create_backend,
    ensure_backends_loaded,
    load_plugins,
    parse_model_spec,
    register_backend,
)

__all__ = [
    "available_backends",
    "create_backend",
    "ensure_backends_loaded",
    "load_plugins",
    "parse_model_spec",
    "register_backend",
]
