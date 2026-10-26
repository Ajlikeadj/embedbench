"""Hugging Face sentence-transformers embedding backend."""

from __future__ import annotations

import gc
import inspect
from typing import Any, Sequence

import numpy as np

from embedbench.backends.base import EmbeddingBackend, heuristic_token_count
from embedbench.backends.registry import register_backend
from embedbench.errors import BackendError
from embedbench.logging_utils import get_logger
from embedbench.models import TokenCount

logger = get_logger(__name__)


@register_backend("hf")
class HuggingFaceBackend(EmbeddingBackend):
    """Local sentence-transformers model, optionally on CUDA or Apple MPS."""

    backend_name = "hf"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self.requested_device = str(self.options.get("device", "auto"))
        self.trust_remote_code = bool(self.options.get("trust_remote_code", False))
        self.cache_folder = self.options.get("cache_folder")
        self.local_files_only = bool(self.options.get("local_files_only", False))
        self.max_seq_length = self.options.get("max_seq_length")
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str = "cpu"
        # E5 models are trained with instruction prefixes.  Config values win,
        # including explicit null (which leaves the prefix disabled).
        if "e5" in model_name.lower():
            if not self.query_prefix:
                self.query_prefix = "query: "
            if not self.document_prefix:
                self.document_prefix = "passage: "

    @property
    def cache_namespace(self) -> str:
        return f"{super().cache_namespace}:maxseq={self.max_seq_length}"

    def _resolve_device(self, torch: Any) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _load_error(self, exc: Exception) -> BackendError:
        message = str(exc)
        lowered = message.lower()
        if "401" in lowered or "gated" in lowered or "unauthorized" in lowered:
            hint = (
                "This model may be gated. Accept its license on Hugging Face and run "
                "'huggingface-cli login', or choose an open model such as "
                "BAAI/bge-small-en-v1.5."
            )
        elif "not found" in lowered or "404" in lowered or "no such file" in lowered:
            hint = (
                "Check the model ID (for example BAAI/bge-small-en-v1.5), or set "
                "backends.hf.local_files_only only if it is already downloaded."
            )
        elif "connection" in lowered or "offline" in lowered or "timeout" in lowered:
            hint = "Hugging Face Hub could not be reached. Check your network or local cache."
        else:
            hint = "Check that the model is a sentence-transformers compatible embedding model."
        return BackendError(f"Could not load HF model '{self.model_name}'. {hint} Cause: {message}")

    def load(self) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BackendError(
                "The Hugging Face backend requires sentence-transformers and torch. "
                "Install them with: pip install 'sentence-transformers>=3.0'"
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(torch)
        kwargs: dict[str, Any] = {"device": self._device}
        if self.cache_folder:
            kwargs["cache_folder"] = str(self.cache_folder)
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.local_files_only:
            kwargs["local_files_only"] = True

        try:
            parameters = inspect.signature(SentenceTransformer.__init__).parameters
            has_var_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if not has_var_kwargs:
                kwargs = {key: value for key, value in kwargs.items() if key in parameters}
            model = SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            logger.debug("SentenceTransformer load failed", exc_info=True)
            raise self._load_error(exc) from exc

        if self.max_seq_length is not None:
            model.max_seq_length = int(self.max_seq_length)
        self._model = model
        dimension_getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        self._dimension = int(dimension_getter())
        logger.info(
            "Loaded HF model %s on %s (dimension=%d, max_seq_length=%s)",
            self.model_name,
            self._device,
            self._dimension,
            getattr(model, "max_seq_length", None),
        )

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            raise BackendError("Hugging Face backend used before load()")
        return self._model

    def _embed_prepared(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_loaded()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise BackendError(
                f"HF model '{self.model_name}' failed while embedding: {exc}"
            ) from exc
        return np.asarray(vectors, dtype=np.float32)

    def count_tokens(self, texts: Sequence[str]) -> TokenCount:
        """Count tokens with the model's own tokenizer (truncated as the model sees it)."""

        model = self._ensure_loaded()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return TokenCount(
                tokens=heuristic_token_count(texts),
                estimated=True,
                method="heuristic (model exposes no tokenizer)",
            )
        max_length = getattr(model, "max_seq_length", None)
        total = 0
        batch_size = 256
        try:
            for offset in range(0, len(texts), batch_size):
                batch = list(texts[offset : offset + batch_size])
                encoded = tokenizer(
                    batch,
                    add_special_tokens=True,
                    truncation=bool(max_length),
                    max_length=max_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids = encoded["input_ids"]
                total += sum(len(ids) for ids in input_ids)
        except Exception as exc:
            logger.warning("Tokenizer counting failed (%s); using heuristic", exc)
            return TokenCount(
                tokens=heuristic_token_count(texts),
                estimated=True,
                method="heuristic (tokenizer counting failed)",
            )

        tokenizer_name = type(tokenizer).__name__
        truncation_note = f", truncated to {max_length}" if max_length else ""
        return TokenCount(
            tokens=total,
            estimated=False,
            method=f"tokenizer:{tokenizer_name}{truncation_note}",
        )

    def reset_peak_memory(self) -> None:
        if self._torch is None:
            return
        if self._device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()

    def runtime_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": "sentence-transformers",
            "device": self._device,
            "requested_device": self.requested_device,
        }
        if self._torch is not None:
            metadata["torch_version"] = getattr(self._torch, "__version__", None)
        model = self._model
        if model is not None:
            metadata["sentencetransformer_max_seq_length"] = getattr(
                model, "max_seq_length", None
            )
        if (
            self._torch is not None
            and self._device.startswith("cuda")
            and self._torch.cuda.is_available()
        ):
            try:
                metadata["gpu_name"] = self._torch.cuda.get_device_name(0)
                properties = self._torch.cuda.get_device_properties(0)
                metadata["gpu_total_memory_mb"] = round(
                    properties.total_memory / (1024 * 1024), 2
                )
                metadata["gpu_peak_allocated_mb"] = round(
                    self._torch.cuda.max_memory_allocated() / (1024 * 1024), 2
                )
                metadata["gpu_peak_reserved_mb"] = round(
                    self._torch.cuda.max_memory_reserved() / (1024 * 1024), 2
                )
            except (RuntimeError, AttributeError) as exc:  # pragma: no cover - driver-specific
                metadata["gpu_metadata_error"] = str(exc)
        return metadata

    def close(self) -> None:
        self._model = None
        gc.collect()
        if self._torch is not None:
            cuda = getattr(self._torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                try:
                    cuda.empty_cache()
                except RuntimeError:  # pragma: no cover - driver-specific
                    pass
        self._torch = None
