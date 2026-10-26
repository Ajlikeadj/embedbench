"""Embedding-model registry and factory.

Providers register themselves with :func:`register_model`; callers usually use
:func:`create_model` with a ``provider:model`` spec:

.. code-block:: python

    from embedbench.models import create_model

    model = create_model("openai:text-embedding-3-small")
    model = create_model("ollama:nomic-embed-text", {"host": "http://localhost:11434"})

Third-party providers can ship an entry point in the ``embedbench.models``
group (the 0.1.x ``embedbench.backends`` group is also honoured) or load a
dotted module with :func:`load_model_plugins`.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from embedbench.config import RunConfig
from embedbench.errors import BackendError, ConfigError
from embedbench.logging_utils import get_logger
from embedbench.models.base import (
    CACHE_FORMAT_VERSION,
    BaseEmbeddingModel,
    EmbeddingBackend,
    TextKind,
    heuristic_token_count,
    normalize_rows,
)
from embedbench.models.providers import (
    MODEL_PRICING,
    PRICING_NOTE,
    pricing_for,
)
from embedbench.schemas import ModelSpec
from embedbench.utils import sha256_hex

logger = get_logger(__name__)

T = TypeVar("T", bound=type[BaseEmbeddingModel])

_MODEL_CLASSES: dict[str, type[BaseEmbeddingModel]] = {}
_ENTRY_POINTS_LOADED = False
_BUILTINS_LOADED = False

ENTRY_POINT_GROUPS: tuple[str, ...] = ("embedbench.models", "embedbench.backends")

#: Provider name -> (SDK extra, API-key environment variable, description)
PROVIDER_METADATA: dict[str, dict[str, str | None]] = {
    "ollama": {
        "extra": None,
        "api_key_env": None,
        "description": "Local or remote Ollama server (no SDK needed)",
    },
    "hf": {
        "extra": "local",
        "api_key_env": None,
        "description": "sentence-transformers model running on CPU/CUDA/MPS",
    },
    "openai": {
        "extra": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "description": "OpenAI text-embedding-3 models and compatible endpoints",
    },
    "cohere": {
        "extra": "cohere",
        "api_key_env": "COHERE_API_KEY",
        "description": "Cohere embed v3/v4 models",
    },
    "voyage": {
        "extra": "voyage",
        "api_key_env": "VOYAGE_API_KEY",
        "description": "Voyage AI retrieval and code embedding models",
    },
}


def register_model(
    name: str,
    *aliases: str,
) -> Callable[[T], T]:
    """Class decorator registering an embedding provider under ``name``.

    Args:
        name: Registry name, used as the ``provider`` half of a model spec.
        *aliases: Additional names that resolve to the same provider.

    Returns:
        The decorator.

    Raises:
        ValueError: If a name is empty or contains ``:``.
    """

    keys = (name, *aliases)
    for key in keys:
        if not key or ":" in key:
            raise ValueError("Provider names must be non-empty and cannot contain ':'")

    def decorator(cls: T) -> T:
        if not isinstance(cls, type) or not issubclass(cls, BaseEmbeddingModel):
            raise TypeError("register_model expects a BaseEmbeddingModel subclass")
        for key in keys:
            existing = _MODEL_CLASSES.get(key)
            if existing is not None and existing is not cls:
                logger.warning(
                    "Replacing embedding provider %r (%s -> %s)",
                    key,
                    existing.__name__,
                    cls.__name__,
                )
            _MODEL_CLASSES[key] = cls
            logger.debug("Registered embedding provider %r -> %s", key, cls.__name__)
        return cls

    return decorator


def _load_builtin_models() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    # Imported for their registration side effects; SDKs stay lazy.
    from embedbench.models import local, providers

    register_model("ollama")(local.OllamaEmbeddingModel)
    register_model("hf", "sentence-transformers", "sentence_transformers")(
        local.SentenceTransformerEmbeddingModel
    )
    register_model("openai")(providers.OpenAIEmbeddingModel)
    register_model("cohere")(providers.CohereEmbeddingModel)
    register_model("voyage")(providers.VoyageEmbeddingModel)


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for group in ENTRY_POINT_GROUPS:
        try:
            entry_points = importlib.metadata.entry_points(group=group)
        except TypeError:  # pragma: no cover - Python < 3.10 compatibility path
            entry_points = importlib.metadata.entry_points().get(group, [])
        for entry_point in entry_points:
            if entry_point.name in _MODEL_CLASSES:
                continue
            try:
                loaded = entry_point.load()
            except Exception as exc:  # a broken plugin must not break every provider
                logger.warning("Could not load provider entry point %r: %s", entry_point.name, exc)
                continue
            if not isinstance(loaded, type) or not issubclass(loaded, BaseEmbeddingModel):
                logger.warning(
                    "Entry point %r is not a BaseEmbeddingModel subclass", entry_point.name
                )
                continue
            register_model(entry_point.name)(loaded)


def ensure_models_loaded() -> None:
    """Import built-in providers and third-party entry points once."""

    _load_builtin_models()
    _load_entry_points()


def parse_model_spec(raw: str) -> ModelSpec:
    """Parse ``provider:model`` and verify that the provider exists.

    Args:
        raw: Model spec string.

    Returns:
        The parsed spec.

    Raises:
        BackendError: If the spec is malformed or the provider is unknown.
    """

    ensure_models_loaded()
    try:
        spec = ModelSpec.parse(raw)
    except ValueError as exc:
        raise BackendError(str(exc)) from exc
    if spec.backend not in _MODEL_CLASSES:
        known = ", ".join(sorted(_MODEL_CLASSES)) or "none"
        raise BackendError(
            f"Unknown embedding provider {spec.backend!r} in {raw!r}. "
            f"Known providers: {known}. Examples: 'hf:BAAI/bge-small-en-v1.5', "
            "'ollama:nomic-embed-text', 'openai:text-embedding-3-small'."
        )
    return spec


def available_providers() -> list[str]:
    """Return the sorted registry names of every provider.

    Returns:
        Provider names such as ``["cohere", "hf", "ollama", "openai", "voyage"]``.
    """

    ensure_models_loaded()
    return sorted(_MODEL_CLASSES)


def provider_metadata() -> list[dict[str, Any]]:
    """Describe built-in providers for the CLI and documentation.

    Returns:
        One mapping per provider with extra package, API-key variable and
        description.  Imports nothing heavy.
    """

    ensure_models_loaded()
    seen: set[type[BaseEmbeddingModel]] = set()
    rows: list[dict[str, Any]] = []
    for name in available_providers():
        cls = _MODEL_CLASSES[name]
        if cls in seen:
            continue
        seen.add(cls)
        metadata = PROVIDER_METADATA.get(name, {})
        rows.append(
            {
                "provider": name,
                "class": cls.__name__,
                "extra": cls.requires_extra,
                "api_key_env": metadata.get("api_key_env"),
                "description": metadata.get("description", ""),
                "price_per_1m_tokens": getattr(cls, "default_price_usd_per_1m", None),
            }
        )
    return rows


def pricing_catalog() -> list[dict[str, Any]]:
    """Return the built-in hosted-model price list.

    Returns:
        One mapping per priced model with ``provider``, ``model`` and price.
    """

    rows = []
    for spec, price in sorted(MODEL_PRICING.items()):
        provider, _, model = spec.partition(":")
        rows.append({"provider": provider, "model": model, "usd_per_1m_tokens": price})
    return rows


def _options_for(spec: ModelSpec, config: RunConfig) -> dict[str, Any]:
    section = getattr(config.backends, spec.backend, None)
    options = section.model_dump(mode="python") if section is not None else {}
    options.update(config.backends.extra.get(spec.backend, {}))
    return options


def create_model(
    spec: str,
    options: Mapping[str, Any] | None = None,
) -> BaseEmbeddingModel:
    """Instantiate the model named by ``provider:model``.

    Args:
        spec: Model spec, for example ``"ollama:nomic-embed-text"``.
        options: Provider options; merged over any defaults.

    Returns:
        An unloaded model instance.  Call ``load()`` (or use it as a context
        manager) before embedding.

    Raises:
        BackendError: If the spec is invalid or the provider cannot be created.
    """

    parsed = parse_model_spec(spec)
    cls = _MODEL_CLASSES[parsed.backend]
    try:
        return cls(parsed.model, dict(options or {}))
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(
            f"Could not initialize provider {parsed.backend!r} for {parsed.model!r}: {exc}"
        ) from exc


def create_backend(spec: str, config: RunConfig) -> BaseEmbeddingModel:
    """Instantiate a model using options from a :class:`RunConfig`.

    Kept for the file-based pipeline and for 0.1.x compatibility.

    Args:
        spec: Model spec string.
        config: Run configuration whose ``backends`` section supplies options.

    Returns:
        An unloaded model instance.
    """

    parsed = parse_model_spec(spec)
    return create_model(spec, _options_for(parsed, config))


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


def load_model_plugins(modules: list[str]) -> None:
    """Load dotted modules or ``.py`` files for provider registration side effects.

    Args:
        modules: Module paths or file paths.

    Raises:
        ConfigError: If a plugin cannot be imported.
    """

    for module in modules:
        try:
            _load_plugin_target(module)
        except Exception as exc:
            raise ConfigError(f"Could not load model plugin {module!r}: {exc}") from exc
        logger.info("Loaded model plugin %s", module)
    ensure_models_loaded()


# -- 0.1.x compatibility aliases ------------------------------------------

register_backend = register_model
load_plugins = load_model_plugins
available_backends = available_providers
parse_backend_spec = parse_model_spec
ensure_backends_loaded = ensure_models_loaded


__all__ = [
    "CACHE_FORMAT_VERSION",
    "ENTRY_POINT_GROUPS",
    "MODEL_PRICING",
    "PRICING_NOTE",
    "PROVIDER_METADATA",
    "BaseEmbeddingModel",
    "EmbeddingBackend",
    "TextKind",
    "available_backends",
    "available_providers",
    "create_backend",
    "create_model",
    "ensure_backends_loaded",
    "ensure_models_loaded",
    "heuristic_token_count",
    "load_model_plugins",
    "load_plugins",
    "normalize_rows",
    "parse_model_spec",
    "pricing_catalog",
    "pricing_for",
    "provider_metadata",
    "register_backend",
    "register_model",
]
