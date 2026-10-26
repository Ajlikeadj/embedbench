from __future__ import annotations

import os

import pytest

# Registers the "fake" backend once per test process.
from tests.fake_backend import FakeEmbeddingBackend

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture
def fake_backend_class() -> type:
    return FakeEmbeddingBackend
