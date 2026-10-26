"""Minimal Ollama REST client used by embeddings and synthetic question generation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import requests

from embedbench.logging_utils import get_logger
from embedbench.utils import truncate

logger = get_logger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_BACKOFF_S = 20.0


class OllamaError(RuntimeError):
    """A user-actionable Ollama failure."""


class OllamaHTTPError(OllamaError):
    """An HTTP error with the status code preserved for endpoint fallback logic."""

    def __init__(self, message: str, status_code: int, body: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _missing_model_message(model: str, host: str) -> str:
    return (
        f"Model '{model}' is not installed in Ollama at {host}. "
        f"Install it with: ollama pull {model}"
    )


class OllamaClient:
    """Thin, synchronous client with bounded retries.

    Only transport errors, 429 and 5xx responses are retried.  4xx responses are
    surfaced immediately with actionable messages.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 0.75,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._session = requests.Session()
        self._legacy_embeddings = False

    @property
    def base_url(self) -> str:
        return self.host

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(self.retry_backoff_s * (2**attempt), _MAX_BACKOFF_S)
        if delay > 0:
            time.sleep(delay)

    def _connection_message(self) -> str:
        return (
            f"Could not reach Ollama at {self.host}. Start it with 'ollama serve' "
            "and check backends.ollama.host in your config."
        )

    def _http_error(self, response: requests.Response, model: str | None) -> OllamaHTTPError:
        body = ""
        try:
            body = response.text
        except Exception:  # pragma: no cover - response text is normally available
            body = ""
        lowered = body.lower()
        model_missing = "try pulling" in lowered or (
            "model" in lowered and ("not found" in lowered or "does not exist" in lowered)
        )
        if model and response.status_code == 404 and model_missing:
            message = _missing_model_message(model, self.host)
        else:
            message = (
                f"Ollama returned HTTP {response.status_code} for {response.url}: "
                f"{truncate(body, 400) or '<empty response>'}"
            )
        return OllamaHTTPError(message, response.status_code, body)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.host}{path}"
        last_transport_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(
                    method,
                    url,
                    json=payload,
                    timeout=self.timeout_s,
                )
            except requests.Timeout as exc:
                last_transport_error = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Ollama request timed out (%ss), retrying %d/%d",
                        self.timeout_s,
                        attempt + 1,
                        self.max_retries,
                    )
                    self._sleep_before_retry(attempt)
                    continue
                raise OllamaError(
                    f"Ollama request to {url} timed out after {self.timeout_s:.0f}s. "
                    "Check that Ollama is running and increase backends.ollama.timeout_s "
                    "if a model is loading."
                ) from exc
            except requests.ConnectionError as exc:
                last_transport_error = exc
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise OllamaError(self._connection_message()) from exc
            except requests.RequestException as exc:
                raise OllamaError(f"Ollama request failed: {exc}") from exc

            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                logger.warning(
                    "Ollama returned HTTP %d, retrying %d/%d",
                    response.status_code,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep_before_retry(attempt)
                continue

            if response.status_code >= 400:
                raise self._http_error(response, model)

            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaError(
                    f"Ollama returned non-JSON content from {url}: {truncate(response.text, 200)}"
                ) from exc
            if not isinstance(data, dict):
                raise OllamaError(f"Unexpected Ollama response from {url}: expected an object")
            return data

        raise OllamaError(self._connection_message()) from last_transport_error

    def list_models(self) -> list[dict[str, Any]]:
        """Return installed models from ``/api/tags``."""

        data = self._request_json("GET", "/api/tags")
        models = data.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama /api/tags returned an unexpected 'models' field")
        return [item for item in models if isinstance(item, dict)]

    def is_available(self) -> tuple[bool, str]:
        try:
            self.list_models()
            return True, "ok"
        except OllamaError as exc:
            return False, str(exc)

    @staticmethod
    def _matches_model(requested: str, installed: str) -> bool:
        if requested == installed:
            return True
        if ":" not in requested and installed == f"{requested}:latest":
            return True
        return installed.split(":", 1)[0] == requested.split(":", 1)[0] and ":" not in requested

    def require_model(self, model: str) -> dict[str, Any] | None:
        """Verify the model exists, with a ``ollama pull`` hint when it does not."""

        models = self.list_models()
        for entry in models:
            installed = str(entry.get("name") or entry.get("model") or "")
            if self._matches_model(model, installed):
                return entry
        raise OllamaError(_missing_model_message(model, self.host))

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        num_predict: int = 220,
    ) -> str:
        """Generate a completion with JSON formatting requested from Ollama."""

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
            },
        }
        data = self._request_json("POST", "/api/generate", payload, model=model)
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise OllamaError(
                f"Ollama model '{model}' returned an empty generation. "
                "Try a larger model or a shorter prompt."
            )
        return text

    def _embed_new_api(
        self,
        model: str,
        texts: Sequence[str],
        keep_alive: str | None,
    ) -> list[list[float]]:
        payload: dict[str, Any] = {"model": model, "input": list(texts)}
        if keep_alive:
            payload["keep_alive"] = keep_alive
        data = self._request_json("POST", "/api/embed", payload, model=model)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError(
                f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 'no'} "
                f"embeddings for {len(texts)} inputs"
            )
        return embeddings

    def _embed_legacy(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            data = self._request_json(
                "POST",
                "/api/embeddings",
                {"model": model, "prompt": text},
                model=model,
            )
            vector = data.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise OllamaError(f"Ollama returned an invalid legacy embedding for '{model}'")
            vectors.append(vector)
        return vectors

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        keep_alive: str | None = None,
    ) -> list[list[float]]:
        """Embed a batch, using ``/api/embed`` and falling back to the legacy route."""

        text_list = list(texts)
        if not text_list:
            return []
        if self._legacy_embeddings:
            return self._embed_legacy(model, text_list)
        try:
            return self._embed_new_api(model, text_list, keep_alive)
        except OllamaHTTPError as exc:
            lowered = exc.body.lower()
            model_missing = "try pulling" in lowered or (
                "model" in lowered and ("not found" in lowered or "does not exist" in lowered)
            )
            if exc.status_code != 404 or model_missing:
                raise
            logger.info("Ollama /api/embed unavailable; falling back to /api/embeddings")
            self._legacy_embeddings = True
            return self._embed_legacy(model, text_list)

    def running_models(self) -> list[dict[str, Any]]:
        data = self._request_json("GET", "/api/ps")
        models = data.get("models", [])
        return (
            [item for item in models if isinstance(item, dict)] if isinstance(models, list) else []
        )

    def version(self) -> str | None:
        try:
            data = self._request_json("GET", "/api/version")
        except OllamaError:
            return None
        value = data.get("version")
        return str(value) if value is not None else None

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
