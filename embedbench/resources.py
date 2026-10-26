"""Process memory sampling and environment metadata for reproducibility."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import threading
import time
from typing import Any

import psutil

from embedbench.logging_utils import get_logger
from embedbench.utils import utc_now_iso

logger = get_logger(__name__)


class ResourceSampler:
    """Sample process RSS on a background thread and report the peak."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = max(0.01, interval_s)
        self._process: psutil.Process | None = None
        self._baseline_bytes = 0
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._process = psutil.Process(os.getpid())
            self._baseline_bytes = int(self._process.memory_info().rss)
            self._peak_bytes = self._baseline_bytes
        except psutil.Error as exc:
            self._error = str(exc)
            logger.warning("Could not initialize memory sampling: %s", exc)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, name="embedbench-memory-sampler", daemon=True
        )
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.sample()

    def sample(self) -> None:
        if self._process is None:
            return
        try:
            rss = int(self._process.memory_info().rss)
        except psutil.Error as exc:
            self._error = str(exc)
            logger.debug("Memory sample failed: %s", exc)
            return
        if rss > self._peak_bytes:
            self._peak_bytes = rss

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():  # pragma: no cover - defensive
                logger.warning("Memory sampler thread did not stop within 2 seconds")
            self._thread = None
        self.sample()

        if self._process is None:
            return {
                "ram_baseline_mb": None,
                "ram_peak_mb": None,
                "ram_delta_mb": None,
                "ram_sampling_interval_s": self.interval_s,
                "ram_sampling_error": self._error,
            }

        return {
            "ram_baseline_mb": round(self._baseline_bytes / (1024 * 1024), 2),
            "ram_peak_mb": round(self._peak_bytes / (1024 * 1024), 2),
            "ram_delta_mb": round(
                (self._peak_bytes - self._baseline_bytes) / (1024 * 1024), 2
            ),
            "ram_sampling_interval_s": self.interval_s,
            "ram_sampling_error": self._error,
        }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    """Collect non-sensitive interpreter, OS and dependency versions."""

    try:
        total_ram = psutil.virtual_memory().total
    except psutil.Error:  # pragma: no cover - defensive
        total_ram = None

    return {
        "captured_at": utc_now_iso(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "total_ram_gb": round(total_ram / (1024**3), 2) if total_ram else None,
        "cwd": os.getcwd(),
        "packages": {
            "embedbench": _distribution_version("embedbench"),
            "numpy": _distribution_version("numpy"),
            "faiss-cpu": _distribution_version("faiss-cpu"),
            "sentence-transformers": _distribution_version("sentence-transformers"),
            "torch": _distribution_version("torch"),
            "pydantic": _distribution_version("pydantic"),
        },
    }


def timed_ms() -> float:
    """Monotonic millisecond timestamp used by benchmark timing code."""

    return time.perf_counter() * 1000.0
