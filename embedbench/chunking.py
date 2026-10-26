"""Deterministic document chunkers.

All strategies return character spans into the cleaned document text.  Overlap
is expressed in characters; sentence- and recursive-based chunkers snap chunk
boundaries to sentence/separator boundaries where possible, so the effective
overlap can be slightly smaller than requested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from embedbench.config import ChunkingConfig
from embedbench.logging_utils import get_logger
from embedbench.models import Chunk, Document

logger = get_logger(__name__)

# Split after sentence punctuation followed by whitespace, on blank lines, and
# before Markdown headings.  This is intentionally a heuristic: abbreviation
# handling is left to the LLM/question layer rather than a heavyweight NLP dep.
_BOUNDARY_RE = re.compile(
    r"(?:[.!?…][\"')\]]?\s+)|\n{2,}|\n(?=#{1,6}\s)",
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)

_RECURSIVE_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
)


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _trim_span(text: str, start: int, end: int) -> _Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return _Span(start, end)


def _hard_split(text: str, start: int, end: int, size: int) -> Iterator[_Span]:
    """Split a span into consecutive windows of at most ``size`` characters."""

    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        trimmed = _trim_span(text, cursor, stop)
        if trimmed is not None:
            yield trimmed
        if stop >= end:
            break
        cursor = stop


def _fixed_spans(text: str, size: int, overlap: int) -> list[_Span]:
    step = max(1, size - overlap)
    spans: list[_Span] = []
    cursor = 0
    total = len(text)
    while cursor < total:
        stop = min(cursor + size, total)
        trimmed = _trim_span(text, cursor, stop)
        if trimmed is not None:
            spans.append(trimmed)
        if stop >= total:
            break
        cursor += step
    return spans


def _sentence_spans(text: str, size: int) -> list[_Span]:
    boundaries = [0]
    boundaries.extend(match.end() for match in _BOUNDARY_RE.finditer(text))
    boundaries.append(len(text))
    boundaries = sorted({position for position in boundaries if 0 <= position <= len(text)})

    spans: list[_Span] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end <= start:
            continue
        if end - start <= size:
            trimmed = _trim_span(text, start, end)
            if trimmed is not None:
                spans.append(trimmed)
        else:
            spans.extend(_hard_split(text, start, end, size))
    return spans


def _split_on_separator(
    text: str, start: int, end: int, separator: str
) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        index = text.find(separator, cursor, end)
        if index < 0:
            pieces.append((cursor, end))
            break
        piece_end = index + len(separator)
        # Keep the separator attached to the preceding piece so no characters
        # are silently dropped from the source spans.
        pieces.append((cursor, piece_end))
        cursor = piece_end
    return [(piece_start, piece_stop) for piece_start, piece_stop in pieces if piece_stop > piece_start]


def _recursive_spans(
    text: str,
    start: int,
    end: int,
    size: int,
    separator_index: int = 0,
) -> Iterator[_Span]:
    if end - start <= size:
        trimmed = _trim_span(text, start, end)
        if trimmed is not None:
            yield trimmed
        return

    if separator_index >= len(_RECURSIVE_SEPARATORS):
        yield from _hard_split(text, start, end, size)
        return

    separator = _RECURSIVE_SEPARATORS[separator_index]
    if separator == "":
        yield from _hard_split(text, start, end, size)
        return

    pieces = _split_on_separator(text, start, end, separator)
    if len(pieces) <= 1:
        yield from _recursive_spans(text, start, end, size, separator_index + 1)
        return

    for piece_start, piece_end in pieces:
        yield from _recursive_spans(text, piece_start, piece_end, size, separator_index + 1)


def _pack_spans(
    text: str,
    spans: Sequence[_Span],
    size: int,
    overlap: int,
) -> list[_Span]:
    """Greedily pack atomic spans into chunks and apply boundary-aligned overlap."""

    packed: list[_Span] = []
    count = len(spans)
    index = 0
    while index < count:
        chunk_start = spans[index].start
        last_end = chunk_start
        cursor = index
        while cursor < count and spans[cursor].end - chunk_start <= size:
            last_end = spans[cursor].end
            cursor += 1

        if cursor == index:
            # Defensive fallback: an atomic span longer than the target size.
            for split_span in _hard_split(text, spans[index].start, spans[index].end, size):
                packed.append(split_span)
            index += 1
            continue

        trimmed = _trim_span(text, chunk_start, last_end)
        if trimmed is not None:
            packed.append(trimmed)

        if cursor >= count:
            break

        if overlap <= 0:
            index = cursor
            continue

        overlap_target = last_end - overlap
        next_index = cursor
        while next_index > index and spans[next_index - 1].start >= overlap_target:
            next_index -= 1
        # Always advance by at least one atomic span; otherwise a huge overlap
        # relative to the chunk size could loop forever.
        index = max(next_index, index + 1)

    return packed


def _materialize(
    text: str,
    spans: Iterable[_Span],
    doc_id: str,
    min_chunk_chars: int,
) -> list[Chunk]:
    spans = list(spans)
    valid: list[_Span] = []
    for span in spans:
        trimmed = _trim_span(text, span.start, span.end)
        if trimmed is None:
            continue
        if trimmed.length >= min_chunk_chars:
            valid.append(trimmed)

    if not valid:
        # Never drop a short document entirely: keep its longest non-empty span.
        candidates = [
            trimmed
            for span in spans
            if (trimmed := _trim_span(text, span.start, span.end)) is not None
        ]
        if candidates:
            valid = [max(candidates, key=lambda item: item.length)]

    chunks: list[Chunk] = []
    for ordinal, span in enumerate(valid):
        piece = text[span.start : span.end].strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{ordinal:04d}",
                doc_id=doc_id,
                ordinal=ordinal,
                text=piece,
                start_char=span.start,
                end_char=span.end,
            )
        )
    return chunks


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split one document according to ``config``."""

    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.chunk_overlap < 0 or config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    text = document.text
    if not text.strip():
        return []

    if config.strategy == "fixed":
        spans = _fixed_spans(text, config.chunk_size, config.chunk_overlap)
    elif config.strategy == "sentence":
        atomic = _sentence_spans(text, config.chunk_size)
        spans = _pack_spans(text, atomic, config.chunk_size, config.chunk_overlap)
    elif config.strategy == "recursive":
        atomic = list(_recursive_spans(text, 0, len(text), config.chunk_size))
        spans = _pack_spans(text, atomic, config.chunk_size, config.chunk_overlap)
    else:  # pragma: no cover - constrained by pydantic, defensive for library API
        raise ValueError(f"Unknown chunking strategy: {config.strategy!r}")

    return _materialize(text, spans, document.doc_id, config.min_chunk_chars)


def chunk_documents(documents: Sequence[Document], config: ChunkingConfig) -> list[Chunk]:
    """Chunk every document, preserving document order and chunk ordinals."""

    chunks: list[Chunk] = []
    for document in documents:
        document_chunks = chunk_document(document, config)
        if not document_chunks:
            logger.warning("No extractable text chunks for %s", document.doc_id)
        chunks.extend(document_chunks)
    logger.info(
        "Created %d chunks from %d documents (strategy=%s, size=%d, overlap=%d)",
        len(chunks),
        len(documents),
        config.strategy,
        config.chunk_size,
        config.chunk_overlap,
    )
    return chunks
