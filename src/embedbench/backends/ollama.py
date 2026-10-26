"""Deprecated: use :class:`embedbench.models.local.OllamaEmbeddingModel`."""

from __future__ import annotations

from embedbench.models.local import OllamaEmbeddingModel

OllamaBackend = OllamaEmbeddingModel

__all__ = ["OllamaBackend", "OllamaEmbeddingModel"]
