"""Exception hierarchy for embedbench.

Every user-facing failure that the CLI should report without a traceback derives
from :class:`EmbedbenchError`.
"""

from __future__ import annotations


class EmbedbenchError(Exception):
    """Base class for all expected embedbench failures."""


class ConfigError(EmbedbenchError):
    """Raised when a configuration file or CLI override is invalid."""


class IngestionError(EmbedbenchError):
    """Raised when documents cannot be discovered or parsed."""


class EvalsetError(EmbedbenchError):
    """Raised for invalid evaluation sets or generation failures."""


class BackendError(EmbedbenchError):
    """Raised when an embedding backend cannot be loaded or used."""


class IndexBuildError(EmbedbenchError):
    """Raised when a vector index cannot be built or searched."""


class ReportError(EmbedbenchError):
    """Raised when results cannot be read or rendered."""
