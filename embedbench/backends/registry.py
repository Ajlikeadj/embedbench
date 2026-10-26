"""Backend registry and factory.

Built-in backends register themselves on import.  Third-party backends can call
``register_backend("name")`` or expose an ``embedbench.backends`` entry point.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

from embedbench.backends.base import EmbeddingBackend
from embedbench.config import RunConfig
from embedbench.errors import BackendError, ConfigError
from embedbench.logging_utils import get_logger
from embedbench.models import ModelSpec
from embedbench.utils import sha256_hex

logger = get_logger(__name__)

_BACKENDS: dict[str, type[EmbeddingBackend]] = {}
_ENTRY_POINTS_LOADED = False
T = TypeVar("T", bound=type[EmbeddingBackend])


def register_backend(
    name: str,
) -> Callable[[T], T]:
    """Class decorator registering an embedding backend under ``name``."""

    key = name.strip().lower()
    if not key or ":" in key:
        raise ValueError("Backend name must be non-empty and cannot contain ':'")

    def decorator(cls: T) -> T:
        if not isinstance(cls, type) or not issubclass(cls, EmbeddingBackend):
            raise TypeError("register_backend expects an EmbeddingBackend subclass")
        existing = _BACKENDS.get(key)
        if existing is not None and existing is not cls:
            logger.warning("Replacing embedding backend %r (%s -> %s)", key, existing.__name__, cls.__name__)
        _BACKENDS[key] = cls
        logger.debug("Registered embedding backend %r -> %s", key, cls.__name__)
        return cls

    return decorator


def _load_builtin_backends() -> None:
    # Importing for their registration side effects.
    importlib.import_module("embedbench.backends.ollama")
    importlib.import_module("embedbench.backends.hf")


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        entry_points = importlib.metadata.entry_points(group="embedbench.backends")
    except TypeError:  # pragma: no cover - Python < 3.10 compatibility path
        entry_points = importlib.metadata.entry_points().get("embedbench.backends", [])
    for entry_point in entry_points:
        try:
            loaded = entry_point.load()
        except Exception as exc:  # a broken plugin should not break every backend
            logger.warning("Could not load backend entry point %r: %s", entry_point.name, exc)
            continue
        if not isinstance(loaded, type) or not issubclass(loaded, EmbeddingBackend):
            logger.warning("Entry point %r is not an EmbeddingBackend subclass", entry_point.name)
            continue
        register_backend(entry_point.name)(loaded)


def ensure_backends_loaded() -> None:
    _load_builtin_backends()
    _load_entry_points()


def parse_model_spec(raw: str) -> ModelSpec:
    """Parse ``backend:model`` and verify that the backend exists."""

    ensure_backends_loaded()
    try:
        spec = ModelSpec.parse(raw)
    except ValueError as exc:
        raise BackendError(str(exc)) from exc
    if spec.backend not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS)) or "none"
        raise BackendError(
            f"Unknown embedding backend {spec.backend!r} in {raw!r}. "
            f"Known backends: {known}. Use 'hf:<model>' or 'ollama:<model>'."
        )
    return spec


def available_backends() -> list[str]:
    ensure_backends_loaded()
    return sorted(_BACKENDS)


def _options_for(spec: ModelSpec, config: RunConfig) -> dict[str, Any]:
    if spec.backend == "ollama":
        return config.backends.ollama.model_dump(mode="python")
    if spec.backend == "hf":
        return config.backends.hf.model_dump(mode="python")
    return dict(config.backends.extra.get(spec.backend, {}))


def create_backend(spec: str, config: RunConfig) -> EmbeddingBackend:
    """Instantiate the backend named by ``backend:model``."""

    parsed = parse_model_spec(spec)
    cls = _BACKENDS[parsed.backend]
    options = _options_for(parsed, config)
    try:
        backend = cls(parsed.model, options)
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(
            f"Could not initialize backend {parsed.backend!r} for {parsed.model!r}: {exc}"
        ) from exc
    backend.backend_name = parsed.backend  # type: ignore[misc]
    return backend


def _load_plugin_target(target: str) -> None:
    """Import a dotted module or execute a standalone ``.py`` plugin file."""

    candidate = Path(target)
    if candidate.suffix == ".py" or candidate.is_file():
        if not candidate.is_file():
            raise ImportError(f"Plugin file not found: {candidate}")
        module_name = f"_embedbench_plugin_{sha256_hex(str(candidate.resolve()))[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create an import spec for {candidate}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return
    importlib.import_module(target)


def load_plugins(modules: list[str]) -> None:
    """Load dotted modules or ``.py`` files for backend registration side effects."""

    for module in modules:
        try:
            _load_plugin_target(module)
        except Exception as exc:
            raise ConfigError(f"Could not load backend plugin {module!r}: {exc}") from exc
        logger.info("Loaded backend plugin %s", module)
    ensure_backends_loaded()
