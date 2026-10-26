"""Deterministic, dependency-free embedding model for tests.

Registered as ``fake``; not part of the public catalogue.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from embedbench.models import BaseEmbeddingModel, register_model

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_index(token: str, dimension: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dimension


@register_model("fake")
class FakeEmbeddingModel(BaseEmbeddingModel):
    """Bag-of-words feature hashing; good enough to exercise retrieval end to end."""

    provider_name = "fake"

    def load(self) -> None:
        if self.model_name == "missing":
            from embedbench.errors import BackendError

            raise BackendError("Fake model 'missing' is not installed")

    def _embed_prepared(self, texts: list[str], kind: str = "document") -> np.ndarray:
        dimension = int(self.options.get("dimension", 128))
        matrix = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            for token in tokens:
                matrix[row, _hash_index(token, dimension)] += 1.0
            for left, right in zip(tokens, tokens[1:], strict=False):
                matrix[row, _hash_index(f"{left} {right}", dimension)] += 1.5
        return matrix


# 0.1.x name kept for tests and probes written against the old API.
FakeEmbeddingBackend = FakeEmbeddingModel
