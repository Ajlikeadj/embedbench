"""Curated model catalog for ``embedbench list-models``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embedbench.ollama_client import OllamaClient, OllamaError


@dataclass(frozen=True, slots=True)
class CatalogModel:
    backend: str
    model: str
    dimension: int | None
    notes: str
    recommended: bool = False

    @property
    def spec(self) -> str:
        return f"{self.backend}:{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "backend": self.backend,
            "model": self.model,
            "dimension": self.dimension,
            "notes": self.notes,
            "recommended": self.recommended,
        }


CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel(
        "hf",
        "BAAI/bge-small-en-v1.5",
        384,
        "Strong small English model; good default for local RAG.",
        True,
    ),
    CatalogModel(
        "hf",
        "all-MiniLM-L6-v2",
        384,
        "Very fast and tiny; lower accuracy but excellent baseline.",
        False,
    ),
    CatalogModel(
        "hf",
        "intfloat/e5-base-v2",
        768,
        "Strong general model; query/passage prefixes applied automatically.",
        True,
    ),
    CatalogModel(
        "hf",
        "BAAI/bge-base-en-v1.5",
        768,
        "Larger BGE variant; better accuracy, more memory and latency.",
        False,
    ),
    CatalogModel(
        "hf",
        "thenlper/gte-small",
        384,
        "Compact general-purpose alternative to BGE-small.",
        False,
    ),
    CatalogModel(
        "hf",
        "mixedbread-ai/mxbai-embed-large-v1",
        1024,
        "High-quality large English embedding model.",
        False,
    ),
    CatalogModel(
        "ollama",
        "nomic-embed-text",
        768,
        "Popular Ollama embedding model with a large context window.",
        True,
    ),
    CatalogModel(
        "ollama",
        "mxbai-embed-large",
        1024,
        "Large Ollama model; strong retrieval quality if you have the RAM.",
        False,
    ),
    CatalogModel(
        "ollama",
        "bge-m3",
        1024,
        "Multilingual, multi-granularity model served by Ollama.",
        True,
    ),
    CatalogModel(
        "ollama",
        "all-minilm",
        384,
        "Small and quick Ollama embedding model.",
        False,
    ),
)


def _installed_matches(requested: str, installed: str) -> bool:
    if requested == installed:
        return True
    if ":" not in requested and installed == f"{requested}:latest":
        return True
    return ":" not in requested and installed.split(":", 1)[0] == requested


def ollama_inventory(host: str, timeout_s: float = 2.0) -> tuple[set[str], str | None]:
    """Return installed Ollama model names and an error message if unreachable."""

    client = OllamaClient(host, timeout_s=timeout_s, max_retries=0)
    try:
        entries = client.list_models()
    except OllamaError as exc:
        return set(), str(exc)
    finally:
        client.close()
    names: set[str] = set()
    for entry in entries:
        name = entry.get("name") or entry.get("model")
        if isinstance(name, str) and name:
            names.add(name)
    return names, None


def catalog_with_status(host: str, timeout_s: float = 2.0) -> dict[str, Any]:
    """Catalog plus Ollama availability, suitable for JSON output."""

    installed, error = ollama_inventory(host, timeout_s=timeout_s)
    entries = []
    for item in CATALOG:
        payload = item.to_dict()
        if item.backend == "ollama":
            if error is not None:
                payload["installed"] = None
            else:
                payload["installed"] = any(
                    _installed_matches(item.model, name) for name in installed
                )
            payload["pull_command"] = f"ollama pull {item.model}"
        else:
            payload["installed"] = None
            payload["pull_command"] = None
        entries.append(payload)
    return {
        "ollama_host": host,
        "ollama_available": error is None,
        "ollama_error": error,
        "models": entries,
    }
