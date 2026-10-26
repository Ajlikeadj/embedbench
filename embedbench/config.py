"""Typed configuration, YAML loading and dotted override handling.

Relative paths in a YAML file are interpreted relative to the process working
directory.  This keeps configs portable across machines and matches how CLI
arguments behave.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from embedbench.errors import ConfigError
from embedbench.utils import atomic_write_text

ChunkingStrategy = Literal["fixed", "sentence", "recursive"]
Priority = Literal["accuracy", "speed", "balanced"]

DEFAULT_EXTENSIONS = [".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"]
DEFAULT_SKIP_DIRS = [
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".embedbench_cache",
]


class _Section(BaseModel):
    """Base config section: typos are errors instead of silently ignored keys."""

    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(_Section):
    """How documents are split before embedding."""

    strategy: ChunkingStrategy = "recursive"
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=4000)
    min_chunk_chars: int = Field(default=80, ge=0, le=4000)

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_chars > self.chunk_size:
            raise ValueError("min_chunk_chars cannot exceed chunk_size")
        return self


class IngestionConfig(_Section):
    """Document discovery and filtering settings."""

    extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    recursive: bool = True
    max_file_bytes: int = Field(default=20_000_000, ge=1024)
    skip_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_SKIP_DIRS))

    @field_validator("extensions")
    @classmethod
    def _normalize_extensions(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = value.strip().lower()
            if not item:
                continue
            if not item.startswith("."):
                item = f".{item}"
            if item not in normalized:
                normalized.append(item)
        if not normalized:
            raise ValueError("at least one file extension is required")
        return normalized


class EvalsetConfig(_Section):
    """Where evaluation queries come from and how synthetic ones are filtered."""

    path: Path | None = None
    auto_generate: bool = False
    ollama_model: str = "llama3.2"
    questions_per_chunk: int = Field(default=1, ge=1, le=5)
    max_chunks: int | None = Field(default=None, ge=1)
    min_chunk_chars: int = Field(default=120, ge=20)
    min_question_words: int = Field(default=5, ge=3, le=20)
    max_question_words: int = Field(default=40, ge=8, le=100)
    dedupe_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    max_generation_attempts: int = Field(default=2, ge=1, le=5)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    num_predict: int = Field(default=220, ge=64, le=1024)
    request_timeout_s: float = Field(default=180.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_word_limits(self) -> "EvalsetConfig":
        if self.min_question_words >= self.max_question_words:
            raise ValueError("min_question_words must be smaller than max_question_words")
        return self


class EmbeddingSettings(_Section):
    """Knobs shared by local and HTTP embedding backends."""

    batch_size: int = Field(default=32, ge=1, le=2048)
    query_prefix: str | None = None
    document_prefix: str | None = None

    @field_validator("query_prefix", "document_prefix")
    @classmethod
    def _empty_prefix_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value if value else None


class HTTPSettings(EmbeddingSettings):
    """Retry and timeout settings for HTTP backends."""

    timeout_s: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_s: float = Field(default=0.75, ge=0.0, le=30.0)


class OllamaSettings(HTTPSettings):
    """Settings for the Ollama REST backend."""

    host: str = "http://localhost:11434"
    keep_alive: str | None = None

    @field_validator("host")
    @classmethod
    def _strip_host(cls, value: str) -> str:
        stripped = value.strip().rstrip("/")
        if not stripped:
            raise ValueError("Ollama host cannot be empty")
        return stripped


class HFSettings(EmbeddingSettings):
    """Settings for the sentence-transformers backend."""

    device: str = "auto"
    trust_remote_code: bool = False
    cache_folder: Path | None = None
    local_files_only: bool = False
    max_seq_length: int | None = Field(default=None, ge=16, le=8192)


class BackendsConfig(_Section):
    """Per-backend configuration; ``extra`` supports third-party backends."""

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    hf: HFSettings = Field(default_factory=HFSettings)
    extra: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CacheConfig(_Section):
    """Embedding cache settings."""

    enabled: bool = True
    dir: Path = Path(".embedbench_cache")


class ReportConfig(_Section):
    """Report rendering and cost assumptions."""

    title: str = "embedbench report"
    priority: Priority = "balanced"
    cost_per_1m_tokens: float = Field(default=0.02, ge=0.0)


class RunConfig(_Section):
    """Top-level configuration for a benchmark run."""

    docs: Path | None = None
    models: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=100)
    seed: int = Field(default=42, ge=0)
    output_dir: Path = Path("results")
    plugins: list[str] = Field(default_factory=list)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    evalset: EvalsetConfig = Field(default_factory=EvalsetConfig)
    backends: BackendsConfig = Field(default_factory=BackendsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @field_validator("models", mode="before")
    @classmethod
    def _split_models(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    @field_validator("plugins")
    @classmethod
    def _clean_plugins(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned


def default_config_path() -> Path | None:
    """Return ``embedbench.yaml`` / ``embedbench.yml`` in the CWD if present."""

    for name in ("embedbench.yaml", "embedbench.yml"):
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    return None


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise ConfigError(f"Invalid override key: {dotted_key!r}")
    cursor: dict[str, Any] = data
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise ConfigError(
                f"Cannot set {dotted_key!r}: {part!r} is not a mapping in the config"
            )
        cursor = existing
    cursor[parts[-1]] = value


def load_config(
    path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load YAML configuration and apply dotted CLI overrides.

    Raises:
        ConfigError: when the file is missing, not a mapping, or fails validation.
    """

    data: dict[str, Any] = {}
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"Could not read config file {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config file {path} must contain a YAML mapping")
        data = dict(loaded)

    for key, value in (overrides or {}).items():
        _set_dotted(data, key, value)

    try:
        return RunConfig.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError, imported lazily for typing
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            details = []
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"]) or "<root>"
                details.append(f"  - {location}: {error['msg']}")
            source = f" in {path}" if path else ""
            raise ConfigError(
                "Invalid configuration" + source + ":\n" + "\n".join(details)
            ) from exc
        raise


def config_snapshot_yaml(config: RunConfig) -> str:
    """Serialize a run config for the reproducibility snapshot."""

    payload = config.model_dump(mode="json")
    header = (
        "# embedbench configuration snapshot\n"
        "# Generated automatically with every run. Relative paths are cwd-relative.\n"
    )
    return header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def save_config_snapshot(config: RunConfig, path: Path) -> None:
    """Write the config snapshot atomically."""

    atomic_write_text(path, config_snapshot_yaml(config))
