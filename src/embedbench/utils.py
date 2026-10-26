"""Small, dependency-light helpers used across embedbench."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_hex(value: str | bytes) -> str:
    """Return the SHA-256 hex digest of a string or byte sequence."""

    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy.

    PyTorch is seeded only when it has already been imported.  Importing torch
    eagerly would make Ollama-only runs pay several seconds of startup cost.
    The HF backend is inference-only, so its output is deterministic in practice.
    """

    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = __import__("sys").modules.get("torch")
    if torch is not None:  # pragma: no branch - depends on installed/loaded extras
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except (RuntimeError, AttributeError):  # pragma: no cover - defensive
            pass


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` atomically so a crashed run never leaves a half file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="",
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)


def json_default(value: Any) -> Any:
    """JSON encoder fallback for NumPy scalars, paths, dataclasses and enums."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Serialize ``payload`` as UTF-8 JSON with a trailing newline."""

    text = json.dumps(payload, indent=2, ensure_ascii=False, default=json_default, allow_nan=False)
    atomic_write_text(path, text + "\n")


def percentile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return count, mean, min, max and p50/p95/p99 for latency values."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    array = np.asarray(values, dtype=np.float64)
    p50, p95, p99 = np.percentile(array, [50, 95, 99], method="linear")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


def human_bytes(size: float | int | None) -> str:
    """Format a byte count like ``12.4 MB``."""

    if size is None:
        return "n/a"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def truncate(text: str, max_chars: int = 160) -> str:
    """Shorten text for logs without splitting mid-word where avoidable."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def relative_posix(path: Path, root: Path) -> str:
    """Return a portable, POSIX-style relative path used as a stable doc ID."""

    return path.relative_to(root).as_posix()
