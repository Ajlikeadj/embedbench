from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from embedbench.chunking import chunk_document
from embedbench.config import ChunkingConfig
from embedbench.schemas import Document


def make_document(text: str, doc_id: str = "doc.md") -> Document:
    return Document(
        doc_id=doc_id,
        path=Path(doc_id),
        text=text,
        source_type="markdown",
    )


def make_config(
    strategy: str,
    *,
    size: int,
    overlap: int = 0,
    min_chars: int = 0,
) -> ChunkingConfig:
    # model_construct lets us exercise sizes below the user-facing minimum (100).
    return ChunkingConfig.model_construct(
        strategy=strategy,
        chunk_size=size,
        chunk_overlap=overlap,
        min_chunk_chars=min_chars,
    )


def assert_positional_consistency(document: Document, chunks: list) -> None:
    for chunk in chunks:
        assert document.text[chunk.start_char : chunk.end_char].strip() == chunk.text
        assert chunk.end_char > chunk.start_char


def assert_non_whitespace_coverage(document: Document, chunks: list) -> None:
    covered = [False] * len(document.text)
    for chunk in chunks:
        for index in range(chunk.start_char, chunk.end_char):
            covered[index] = True
    for index, character in enumerate(document.text):
        if not character.isspace():
            assert covered[index], f"uncovered non-whitespace at {index}: {character!r}"


def test_fixed_chunking_exact_windows_and_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    document = make_document(text)
    config = make_config("fixed", size=8, overlap=2)

    chunks = chunk_document(document, config)

    # Windows start at 0, 6, 12 and 18; the window starting at 18 already
    # reaches the end, so no extra tail window is emitted.
    assert [chunk.text for chunk in chunks] == [
        "abcdefgh",
        "ghijklmn",
        "mnopqrst",
        "stuvwxyz",
    ]
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc.md#0000",
        "doc.md#0001",
        "doc.md#0002",
        "doc.md#0003",
    ]
    assert_positional_consistency(document, chunks)
    assert_non_whitespace_coverage(document, chunks)


def test_fixed_chunking_min_chunk_chars_drops_tail() -> None:
    document = make_document("abcdefghijklmnopqrstuvwxyz")
    chunks = chunk_document(document, make_config("fixed", size=8, overlap=2, min_chars=5))
    assert [chunk.text for chunk in chunks] == ["abcdefgh", "ghijklmn", "mnopqrst", "stuvwxyz"]


def test_short_document_is_never_dropped() -> None:
    document = make_document("tiny but useful")
    chunks = chunk_document(document, make_config("fixed", size=100, overlap=0, min_chars=80))
    assert len(chunks) == 1
    assert chunks[0].text == "tiny but useful"


def test_sentence_chunking_keeps_sentence_boundaries() -> None:
    sentence = "Sentence number %d is short and clear. "
    text = "".join(sentence % number for number in range(1, 10))
    document = make_document(text)
    config = make_config("sentence", size=100, overlap=10)

    chunks = chunk_document(document, config)

    assert len(chunks) >= 3
    assert all(chunk.text.endswith(".") for chunk in chunks)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert_positional_consistency(document, chunks)
    assert_non_whitespace_coverage(document, chunks)


def test_recursive_chunking_prefers_paragraph_boundaries() -> None:
    p1 = "Alpha paragraph explains the first concept with enough detail."
    p2 = "Beta paragraph explains the second concept with different details."
    text = f"{p1}\n\n{p2}"
    document = make_document(text)
    config = make_config("recursive", size=100, overlap=0)

    chunks = chunk_document(document, config)

    assert len(chunks) == 2
    assert "Alpha" in chunks[0].text and "Beta" not in chunks[0].text
    assert "Beta" in chunks[1].text and "Alpha" not in chunks[1].text
    assert_non_whitespace_coverage(document, chunks)


def test_recursive_chunking_respects_size_and_coverage_for_long_paragraphs() -> None:
    text = ("word " * 60).strip() + "\n\n" + ("other " * 60).strip()
    document = make_document(text)
    config = make_config("recursive", size=100, overlap=20)

    chunks = chunk_document(document, config)

    assert len(chunks) > 2
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert_positional_consistency(document, chunks)
    assert_non_whitespace_coverage(document, chunks)


def test_chunking_is_deterministic() -> None:
    document = make_document("Deterministic chunks are important. " * 20)
    config = make_config("recursive", size=120, overlap=20, min_chars=30)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert [(chunk.chunk_id, chunk.text, chunk.start_char) for chunk in first] == [
        (chunk.chunk_id, chunk.text, chunk.start_char) for chunk in second
    ]


def test_empty_document_returns_no_chunks() -> None:
    document = make_document("   \n\n  ")
    assert chunk_document(document, make_config("recursive", size=100)) == []


def test_chunk_ids_are_stable_and_ordinal() -> None:
    document = make_document("one two three. " * 30, doc_id="folder/file.md")
    chunks = chunk_document(document, make_config("sentence", size=100, overlap=0))
    assert all(chunk.doc_id == "folder/file.md" for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(
        chunk.chunk_id == f"folder/file.md#{index:04d}" for index, chunk in enumerate(chunks)
    )


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=100)

    document = make_document("some text " * 30)
    invalid = make_config("fixed", size=50, overlap=50)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(document, invalid)


def test_whitespace_is_trimmed_from_chunk_edges() -> None:
    text = "   First sentence.    Second sentence.   "
    document = make_document(text)
    chunks = chunk_document(document, make_config("sentence", size=100, overlap=0))
    for chunk in chunks:
        assert chunk.text == chunk.text.strip()
