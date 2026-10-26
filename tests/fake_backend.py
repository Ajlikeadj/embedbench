"""Deterministic, dependency-free embedding backend for tests.

It is deliberately registered as ``fake`` and is not part of the public catalog.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from embedbench.backends import register_backend
from embedbench.backends.base import EmbeddingBackend

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_index(token: str, dimension: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dimension


@register_backend("fake")
class FakeEmbeddingBackend(EmbeddingBackend):
    """Bag-of-words feature hashing; good enough to exercise retrieval end to end."""

    backend_name = "fake"

    def load(self) -> None:
        if self.model_name == "missing":
            from embedbench.errors import BackendError

            raise BackendError("Fake model 'missing' is not installed")

    def _embed_prepared(self, texts: list[str]) -> np.ndarray:
        dimension = int(self.options.get("dimension", 128))
        matrix = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            for token in tokens:
                matrix[row, _hash_index(token, dimension)] += 1.0
            for left, right in zip(tokens, tokens[1:], strict=False):
                matrix[row, _hash_index(f"{left} {right}", dimension)] += 1.5
        return matrix
