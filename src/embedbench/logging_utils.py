"""Logging setup shared by the CLI and library callers."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_LOGGER_NAME = "embedbench"
_NOISY_LOGGERS = (
    "httpx",
    "urllib3",
    "transformers",
    "sentence_transformers",
    "torch",
    "faiss",
    "filelock",
    "huggingface_hub",
)


def setup_logging(
    *,
    verbose: bool = False,
    log_file: Path | None = None,
    force: bool = False,
) -> None:
    """Configure the ``embedbench`` logger once per process.

    Args:
        verbose: show debug messages and source locations.
        log_file: optional file that receives the same messages.
        force: replace existing handlers (used by tests and repeated CLI calls).
    """

    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers and not force:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_level=True,
        show_path=verbose,
        markup=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8", delay=True)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.WARNING)


def close_logging_handlers() -> None:
    """Close and remove all file handlers attached to the embedbench logger."""

    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the embedbench namespace."""

    if name == _LOGGER_NAME or name.startswith(_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def console() -> Console:
    """Return a fresh stdout console for CLI tables and progress bars."""

    return Console()
