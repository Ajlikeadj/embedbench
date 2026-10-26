"""Tests for the Ollama HTTP client (retries, fallbacks and model checks)."""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from embedbench.ollama_client import OllamaClient, OllamaError


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, body: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = body if body is not None else json.dumps(payload or {})
        self.url = "http://example.invalid/api"
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def install_sequence(
    client: OllamaClient, monkeypatch: pytest.MonkeyPatch, sequence: list[Any]
) -> list[str]:
    calls: list[str] = []

    def fake_request(method: str, url: str, json: Any = None, timeout: float = 0) -> FakeResponse:
        calls.append(url)
        item = sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client._session, "request", fake_request)
    return calls


def test_ollama_client_retries_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=2, retry_backoff_s=0)
    calls = install_sequence(
        client,
        monkeypatch,
        [requests.ConnectionError("boom"), FakeResponse(200, {"models": []})],
    )
    assert client.list_models() == []
    assert len(calls) == 2
    client.close()


def test_ollama_client_retries_503_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=1, retry_backoff_s=0)
    calls = install_sequence(
        client,
        monkeypatch,
        [FakeResponse(503, {"error": "loading"}), FakeResponse(200, {"models": []})],
    )
    assert client.list_models() == []
    assert len(calls) == 2
    client.close()


def test_ollama_connection_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=1, retry_backoff_s=0)
    install_sequence(
        client,
        monkeypatch,
        [requests.ConnectionError("no route"), requests.ConnectionError("no route")],
    )
    with pytest.raises(OllamaError, match="ollama serve"):
        client.list_models()
    client.close()


def test_ollama_require_model_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    install_sequence(
        client, monkeypatch, [FakeResponse(200, {"models": [{"name": "other:latest"}]})]
    )
    with pytest.raises(OllamaError, match="ollama pull nomic-embed-text"):
        client.require_model("nomic-embed-text")
    client.close()


def test_ollama_require_model_accepts_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    install_sequence(
        client, monkeypatch, [FakeResponse(200, {"models": [{"name": "nomic-embed-text:latest"}]})]
    )
    assert client.require_model("nomic-embed-text") is not None
    client.close()


def test_ollama_embed_falls_back_to_legacy_api(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    calls: list[str] = []

    def fake_request(method: str, url: str, json: Any = None, timeout: float = 0) -> FakeResponse:
        calls.append(url)
        if url.endswith("/api/embed"):
            return FakeResponse(404, body="404 page not found")
        prompt = json["prompt"]
        return FakeResponse(200, {"embedding": [float(len(prompt)), 2.0]})

    monkeypatch.setattr(client._session, "request", fake_request)
    vectors = client.embed("nomic-embed-text", ["ab", "abcd"])
    assert vectors == [[2.0, 2.0], [4.0, 2.0]]
    assert calls[0].endswith("/api/embed")
    assert calls[1].endswith("/api/embeddings")

    vectors_again = client.embed("nomic-embed-text", ["abc"])
    assert vectors_again == [[3.0, 2.0]]
    assert calls[-1].endswith("/api/embeddings")
    client.close()


def test_ollama_embed_missing_model_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient("http://example.invalid", max_retries=0, retry_backoff_s=0)
    calls = install_sequence(
        client,
        monkeypatch,
        [FakeResponse(404, {"error": "model 'nomic-embed-text' not found"})],
    )
    with pytest.raises(OllamaError, match="ollama pull"):
        client.embed("nomic-embed-text", ["text"])
    assert len(calls) == 1
    client.close()
