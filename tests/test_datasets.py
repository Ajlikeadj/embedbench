"""Tests for retrieval datasets and the synthetic stress generators."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from embedbench.datasets import (
    CHARS_PER_TOKEN,
    DatasetDocument,
    RetrievalDataset,
    RetrievalQuery,
    build_haystack,
    chunk_size_sensitivity,
    degrade_query,
    demo_dataset,
    load_corpus_dataset,
    load_documents_dataset,
    load_jsonl_dataset,
    load_queries,
    make_needle,
    needle_in_haystack,
    query_variants,
)
from embedbench.errors import DatasetError


def _dataset() -> RetrievalDataset:
    return RetrievalDataset(
        name="tiny",
        documents=[
            DatasetDocument(doc_id="d1", text="Alpha beta gamma delta."),
            DatasetDocument(doc_id="d2", text="Epsilon zeta eta theta."),
        ],
        queries=[
            RetrievalQuery(query_id="q1", query="alpha?", relevant_doc_ids=("d1",)),
            RetrievalQuery(query_id="q2", query="zeta?", relevant_doc_ids=("d2",)),
        ],
    )


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


def test_demo_dataset_is_valid_and_has_stats() -> None:
    dataset = demo_dataset()
    dataset.validate()
    stats = dataset.stats()
    assert stats["documents"] == 6
    assert stats["queries"] == 6
    assert stats["groups"] == 6
    assert stats["chars"] > 0
    assert stats["avg_query_words"] > 0


def test_validate_rejects_duplicate_ids_empty_documents_and_unknown_relevance() -> None:
    dataset = _dataset()
    dataset.documents.append(DatasetDocument(doc_id="d1", text="duplicate"))
    with pytest.raises(DatasetError, match="Duplicate doc_id"):
        dataset.validate()

    empty_text = _dataset()
    empty_text.documents[0] = DatasetDocument(doc_id="d1", text="   ")
    with pytest.raises(DatasetError, match="is empty"):
        empty_text.validate()

    unknown = _dataset()
    unknown.queries[0] = RetrievalQuery(
        query_id="q1", query="alpha?", relevant_doc_ids=("missing",)
    )
    with pytest.raises(DatasetError, match="unknown documents"):
        unknown.validate()

    no_queries = _dataset()
    no_queries.queries = []
    with pytest.raises(DatasetError, match="no queries"):
        no_queries.validate()

    no_docs = _dataset()
    no_docs.documents = []
    with pytest.raises(DatasetError, match="no documents"):
        no_docs.validate()

    duplicate_query = _dataset()
    duplicate_query.queries.append(duplicate_query.queries[0])
    with pytest.raises(DatasetError, match="Duplicate query_id"):
        duplicate_query.validate()


def test_group_ranking_collapses_chunks_in_first_seen_order() -> None:
    dataset = RetrievalDataset(
        name="chunked",
        documents=[
            DatasetDocument(doc_id="c1", text="a", group_id="doc-a"),
            DatasetDocument(doc_id="c2", text="b", group_id="doc-b"),
            DatasetDocument(doc_id="c3", text="c", group_id="doc-a"),
            DatasetDocument(doc_id="c4", text="d", group_id="doc-c"),
        ],
        queries=[RetrievalQuery(query_id="q", query="x", relevant_doc_ids=("doc-a",))],
    )
    assert dataset.num_groups == 3
    assert dataset.group_ranking(["c2", "c1", "c3", "c4"]) == ["doc-b", "doc-a", "doc-c"]
    assert dataset.group_ranking(["c2", "c1", "c3", "c4"], limit=2) == ["doc-b", "doc-a"]
    assert dataset.group_ranking(["c1", "c3"]) == ["doc-a"]


def test_dataset_dict_round_trip() -> None:
    dataset = demo_dataset()
    payload = {
        "name": dataset.name,
        "documents": [document.to_dict() for document in dataset.documents],
        "queries": [query.to_dict() for query in dataset.queries],
        "metadata": dataset.metadata,
    }
    restored = RetrievalDataset.from_dict(payload)
    assert restored.name == dataset.name
    assert restored.stats() == dataset.stats()
    with pytest.raises(DatasetError, match="missing key"):
        RetrievalDataset.from_dict({"name": "broken"})


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def test_jsonl_round_trip(tmp_path: Path) -> None:
    dataset = demo_dataset()
    path = dataset.save_jsonl(tmp_path / "dataset.jsonl")
    restored = load_jsonl_dataset(path)
    assert restored.name == "dataset"
    assert restored.stats()["documents"] == dataset.stats()["documents"]
    assert restored.stats()["queries"] == dataset.stats()["queries"]
    restored.validate()


def test_jsonl_accepts_legacy_query_only_files(tmp_path: Path) -> None:
    path = tmp_path / "evalset.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "hello", "relevant_doc_ids": ["d1"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="no documents"):
        load_jsonl_dataset(path)


def test_jsonl_rejects_bad_lines(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_jsonl_dataset(bad_json)

    unknown_type = tmp_path / "unknown.jsonl"
    unknown_type.write_text(json.dumps({"type": "other"}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="unknown record type"):
        load_jsonl_dataset(unknown_type)

    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"type": "document", "doc_id": "d1"}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing"):
        load_jsonl_dataset(missing)


# ---------------------------------------------------------------------------
# Corpus bridge
# ---------------------------------------------------------------------------


def test_corpus_dataset_groups_chunks_by_source_document(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("# Alpha\n\n" + "alpha beta gamma " * 90, encoding="utf-8")
    (docs / "two.md").write_text("# Beta\n\n" + "delta epsilon zeta " * 90, encoding="utf-8")
    evalset = tmp_path / "evalset.jsonl"
    evalset.write_text(
        json.dumps({"query_id": "q1", "query": "alpha beta", "relevant_doc_ids": ["one.md"]})
        + "\n",
        encoding="utf-8",
    )

    dataset = load_corpus_dataset(docs, evalset)
    dataset.validate()
    assert dataset.num_documents > 2
    assert dataset.num_groups == 2
    assert dataset.queries[0].relevant_doc_ids == ("one.md",)
    assert dataset.metadata["chunks"] == dataset.num_documents

    documents = load_documents_dataset(docs)
    assert [document.doc_id for document in documents] == ["one.md", "two.md"]
    queries = load_queries(evalset)
    assert queries[0].query == "alpha beta"


def test_corpus_dataset_reports_unknown_relevance(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("alpha beta gamma", encoding="utf-8")
    evalset = tmp_path / "evalset.jsonl"
    evalset.write_text(
        json.dumps({"query_id": "q1", "query": "alpha", "relevant_doc_ids": ["nope.md"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="nope.md|unknown"):
        load_corpus_dataset(docs, evalset)


# ---------------------------------------------------------------------------
# Needle in a haystack
# ---------------------------------------------------------------------------


def test_build_haystack_places_needles_at_depths() -> None:
    needles = [
        (0.0, make_needle(random.Random(1), 0)),
        (1.0, make_needle(random.Random(2), 1)),
    ]
    text, spans = build_haystack(512, needles, seed=7)
    assert len(spans) == 2
    for needle, start, end in spans:
        assert text[start:end] == needle.sentence
    first_start = spans[0][1]
    second_start = spans[1][1]
    assert first_start < len(text) * 0.3
    assert second_start > len(text) * 0.7

    with pytest.raises(DatasetError):
        build_haystack(16, [])
    with pytest.raises(DatasetError):
        build_haystack(512, [(1.5, needles[0][1])])


def test_needle_datasets_contain_the_needle_chunk() -> None:
    datasets = needle_in_haystack(
        context_sizes=(256, 512),
        depths=(0.0, 0.5, 1.0),
        chunk_size=200,
        chunk_overlap=40,
        seed=11,
    )
    assert set(datasets) == {256, 512}
    for context_size, dataset in datasets.items():
        dataset.validate()
        by_id = {document.doc_id: document.text for document in dataset.documents}
        assert dataset.num_queries == 3
        assert dataset.metadata["context_tokens"] == context_size
        for query in dataset.queries:
            assert query.metadata["depth"] in (0.0, 0.5, 1.0)
            code = query.metadata["code"]
            hits = [text for text in by_id.values() if code in text]
            assert hits, "the needle must be present in exactly the indexed chunks"
            assert any(code in by_id[doc_id] for doc_id in query.relevant_doc_ids)

    larger = datasets[512]
    smaller = datasets[256]
    assert larger.num_documents >= smaller.num_documents


def test_needle_queries_are_deterministic() -> None:
    first = needle_in_haystack(context_sizes=(256,), depths=(0.0,), seed=5)[256]
    second = needle_in_haystack(context_sizes=(256,), depths=(0.0,), seed=5)[256]
    assert first.queries[0].query == second.queries[0].query
    assert first.queries[0].metadata["code"] == second.queries[0].metadata["code"]


# ---------------------------------------------------------------------------
# Chunk-size sensitivity
# ---------------------------------------------------------------------------


def test_chunk_size_sensitivity_shrinks_chunk_counts() -> None:
    documents = [
        DatasetDocument(
            doc_id=f"doc-{index}",
            text=("Paragraph about retrieval quality and chunk boundaries. " * 40),
        )
        for index in range(2)
    ]
    queries = [
        RetrievalQuery(query_id="q1", query="chunk boundaries?", relevant_doc_ids=("doc-0",)),
    ]
    datasets = chunk_size_sensitivity(documents, queries, chunk_tokens=(128, 512))
    assert set(datasets) == {128, 512}
    small, large = datasets[128], datasets[512]
    assert small.num_documents > large.num_documents
    assert large.metadata["chunk_chars"] == int(512 * CHARS_PER_TOKEN)
    large.validate()
    for document in large.documents:
        assert document.group_id in {"doc-0", "doc-1"}
        assert document.doc_id.startswith(document.group_id)

    with pytest.raises(DatasetError):
        chunk_size_sensitivity([], queries)
    with pytest.raises(DatasetError):
        chunk_size_sensitivity(documents, queries, chunk_tokens=(8,))


# ---------------------------------------------------------------------------
# Query degradation
# ---------------------------------------------------------------------------


def test_degrade_query_is_deterministic_and_changes_text() -> None:
    query = "How many times are failed charges retried?"
    for kind in ("typo", "case", "truncate", "shuffle", "paraphrase", "noise"):
        first = degrade_query(query, kind, seed=3)
        second = degrade_query(query, kind, seed=3)
        assert first == second
        assert first != query
        assert first.strip()

    assert degrade_query(query, "clean") == query
    with pytest.raises(DatasetError, match="Unknown query noise kind"):
        degrade_query(query, "gibberish")


def test_query_variants_keep_documents_and_share_relevance() -> None:
    dataset = demo_dataset()
    variants = query_variants(dataset, ("typo", "truncate"), seed=9)
    assert list(variants) == ["clean", "typo", "truncate"]
    assert variants["clean"] is dataset
    for name, variant in variants.items():
        variant.validate()
        assert variant.documents == dataset.documents
        assert [query.relevant_doc_ids for query in variant.queries] == [
            query.relevant_doc_ids for query in dataset.queries
        ]
        if name != "clean":
            assert all(query.metadata["noise"] == name for query in variant.queries)
            assert [query.query for query in variant.queries] != [
                query.query for query in dataset.queries
            ]
