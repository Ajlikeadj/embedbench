"""Typed data structures passed between ingestion, benchmarking and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Document:
    """A parsed source document."""

    doc_id: str
    path: Path
    text: str
    source_type: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "path": str(self.path),
            "source_type": self.source_type,
            "title": self.title,
            "chars": len(self.text),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Chunk:
    """A contiguous span of a document that will be embedded."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "ordinal": self.ordinal,
            "chars": len(self.text),
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(slots=True)
class EvalExample:
    """One retrieval query and the document IDs considered relevant."""

    query_id: str
    query: str
    relevant_doc_ids: list[str]
    answer: str | None = None
    source_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query_id": self.query_id,
            "query": self.query,
            "relevant_doc_ids": list(self.relevant_doc_ids),
        }
        if self.answer is not None:
            payload["answer"] = self.answer
        if self.source_chunk_id is not None:
            payload["source_chunk_id"] = self.source_chunk_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A ``backend:model`` identifier."""

    raw: str
    backend: str
    model: str

    @classmethod
    def parse(cls, raw: str) -> ModelSpec:
        value = raw.strip()
        backend, separator, model = value.partition(":")
        backend = backend.strip().lower()
        model = model.strip()
        if not separator or not backend or not model:
            raise ValueError(
                f"Invalid model spec {raw!r}. Expected 'backend:model', "
                "for example 'ollama:nomic-embed-text' or "
                "'hf:BAAI/bge-small-en-v1.5'."
            )
        return cls(raw=value, backend=backend, model=model)


@dataclass(frozen=True, slots=True)
class TokenCount:
    """Token count for a set of texts and whether it is exact."""

    tokens: int
    estimated: bool
    method: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage and cost-equivalent estimate for one model run."""

    index_tokens: int
    query_tokens: int
    total_tokens: int
    estimated: bool
    method: str
    cost_usd: float
    price_per_1m_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_tokens": self.index_tokens,
            "query_tokens": self.query_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
            "method": self.method,
            "cost_usd": self.cost_usd,
            "price_per_1m_tokens": self.price_per_1m_tokens,
        }
