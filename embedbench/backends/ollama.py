"""Ollama embedding backend."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from embedbench.backends.base import EmbeddingBackend, heuristic_token_count
from embedbench.backends.registry import register_backend
from embedbench.errors import BackendError
from embedbench.logging_utils import get_logger
from embedbench.models import TokenCount
from embedbench.ollama_client import OllamaClient, OllamaError

logger = get_logger(__name__)


@register_backend("ollama")
class OllamaBackend(EmbeddingBackend):
    """Embeddings served by a local or remote Ollama instance."""

    backend_name = "ollama"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self.host = str(self.options.get("host", "http://localhost:11434")).rstrip("/")
        self.timeout_s = float(self.options.get("timeout_s", 120.0))
        self.max_retries = int(self.options.get("max_retries", 3))
        self.retry_backoff_s = float(self.options.get("retry_backoff_s", 0.75))
        self.keep_alive = self.options.get("keep_alive")
        self._client: OllamaClient | None = None

    @property
    def cache_namespace(self) -> str:
        return f"{super().cache_namespace}:host={self.host}"

    def load(self) -> None:
        client = OllamaClient(
            self.host,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
        )
        try:
            client.require_model(self.model_name)
        except OllamaError as exc:
            client.close()
            raise BackendError(str(exc)) from exc
        self._client = client
        logger.debug("Ollama backend ready: %s at %s", self.model_name, self.host)

    def _embed_prepared(self, texts: list[str]) -> np.ndarray:
        if self._client is None:
            raise BackendError("Ollama backend used before load()")
        try:
            vectors = self._client.embed(
                self.model_name,
                texts,
                keep_alive=str(self.keep_alive) if self.keep_alive else None,
            )
        except OllamaError as exc:
            raise BackendError(str(exc)) from exc
        try:
            return np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise BackendError(f"Ollama returned malformed embeddings: {exc}") from exc

    def count_tokens(self, texts: Sequence[str]) -> TokenCount:
        return TokenCount(
            tokens=heuristic_token_count(texts),
            estimated=True,
            method="heuristic (Ollama REST does not expose the model tokenizer)",
        )

    def runtime_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": "ollama",
            "ollama_host": self.host,
        }
        if self._client is None:
            return metadata
        version = self._client.version()
        if version:
            metadata["ollama_version"] = version
        try:
            running = self._client.running_models()
        except OllamaError as exc:
            metadata["ollama_ps_error"] = str(exc)
            return metadata
        for entry in running:
            name = str(entry.get("name") or entry.get("model") or "")
            if OllamaClient._matches_model(self.model_name, name):
                metadata["model_resident_vram_bytes"] = entry.get("size_vram")
                metadata["model_resident_bytes"] = entry.get("size")
                break
        return metadata

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
