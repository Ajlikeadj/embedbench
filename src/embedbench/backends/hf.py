"""Deprecated: use :class:`embedbench.models.local.SentenceTransformerEmbeddingModel`."""

from __future__ import annotations

from embedbench.models.local import SentenceTransformerEmbeddingModel

HuggingFaceBackend = SentenceTransformerEmbeddingModel

__all__ = ["HuggingFaceBackend", "SentenceTransformerEmbeddingModel"]
