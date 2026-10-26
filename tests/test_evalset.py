from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from embedbench.config import EvalsetConfig
from embedbench.errors import EvalsetError
from embedbench.evalset import (
    SyntheticEvalsetGenerator,
    load_evalset,
    parse_generated_questions,
    save_evalset,
    validate_evalset_documents,
)
from embedbench.ollama_client import OllamaError
from embedbench.schemas import Chunk, EvalExample


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_load_evalset_valid_records_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n"
        + json.dumps(
            {
                "query": "  What is   the timeout? ",
                "relevant_doc_ids": ["./docs\\guide.md", "docs/guide.md"],
                "query_id": "q1",
                "answer": "Thirty seconds.",
                "category": "api",
            }
        )
        + "\n\n"
        + json.dumps({"query": "How many retries?", "relevant_doc_ids": ["docs/api.md"]})
        + "\n",
        encoding="utf-8",
    )

    examples = load_evalset(path)

    assert len(examples) == 2
    assert examples[0].query_id == "q1"
    assert examples[0].query == "What is the timeout?"
    assert examples[0].relevant_doc_ids == ["docs/guide.md"]
    assert examples[0].metadata["category"] == "api"
    assert examples[1].query_id == "q0001"
    assert examples[1].answer is None


def test_auto_query_ids_do_not_collide_with_explicit_ids(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path,
        [
            {"query": "one?", "relevant_doc_ids": ["a"], "query_id": "q0002"},
            {"query": "two?", "relevant_doc_ids": ["b"]},
            {"query": "three?", "relevant_doc_ids": ["c"]},
        ],
    )
    examples = load_evalset(path)
    assert [example.query_id for example in examples] == ["q0002", "q0003", "q0004"]


def test_duplicate_query_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path,
        [
            {"query": "one?", "relevant_doc_ids": ["a"], "query_id": "dup"},
            {"query": "two?", "relevant_doc_ids": ["b"], "query_id": "dup"},
        ],
    )
    with pytest.raises(EvalsetError, match="Duplicate query_id"):
        load_evalset(path)


@pytest.mark.parametrize(
    "record",
    [
        {"query": "", "relevant_doc_ids": ["a"]},
        {"query": "q?", "relevant_doc_ids": []},
        {"query": "q?", "relevant_doc_ids": "a"},
        {"query": "q?", "relevant_doc_ids": [""]},
        {"query": "q?", "relevant_doc_ids": ["a"], "query_id": 3},
    ],
)
def test_invalid_records_are_rejected(tmp_path: Path, record: dict) -> None:
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [record])
    with pytest.raises(EvalsetError):
        load_evalset(path)


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query": "ok?", "relevant_doc_ids": ["a"]}\nnot json\n', encoding="utf-8")
    with pytest.raises(EvalsetError, match="line 2"):
        load_evalset(path)


def test_empty_evalset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(EvalsetError, match="empty"):
        load_evalset(path)


def test_save_and_reload_roundtrip(tmp_path: Path) -> None:
    examples = [
        EvalExample(
            query_id="q1",
            query="What is the payload limit?",
            relevant_doc_ids=["api.md"],
            answer="5 MB",
            source_chunk_id="api.md#0000",
            metadata={"origin": "manual"},
        )
    ]
    path = tmp_path / "roundtrip.jsonl"
    save_evalset(examples, path)
    loaded = load_evalset(path)
    assert loaded[0].to_dict() == examples[0].to_dict()


def test_validate_evalset_documents_reports_missing_ids() -> None:
    examples = [
        EvalExample("q1", "first?", ["a.md"]),
        EvalExample("q2", "second?", ["missing.md"]),
    ]
    with pytest.raises(EvalsetError, match="missing.md"):
        validate_evalset_documents(examples, {"a.md"})
    validate_evalset_documents(examples[:1], {"a.md"})


def test_parse_generated_questions_variants() -> None:
    assert parse_generated_questions('{"questions":[{"question":"What?","answer":"This."}]}') == [
        {"question": "What?", "answer": "This."}
    ]
    assert parse_generated_questions('```json\n{"question":"Why?","answer":"Because."}\n```') == [
        {"question": "Why?", "answer": "Because."}
    ]
    assert parse_generated_questions('Here you go: ["How many?", {"question":"Which one?"}]') == [
        {"question": "How many?", "answer": ""},
        {"question": "Which one?", "answer": ""},
    ]
    with pytest.raises(ValueError):
        parse_generated_questions("not json at all")
    with pytest.raises(ValueError):
        parse_generated_questions('{"questions": []}')


def make_chunk(text: str, doc_id: str = "api.md", chunk_id: str = "api.md#0000") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=0,
        text=text,
        start_char=0,
        end_char=len(text),
    )


class FakeOllamaClient:
    def __init__(self, payload: object | None = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls = 0
        self.require_calls = 0

    def require_model(self, model: str) -> None:
        self.require_calls += 1

    def generate(self, model: str, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload)


def generator_config(**overrides) -> EvalsetConfig:
    values = {
        "min_chunk_chars": 20,
        "min_question_words": 3,
        "max_question_words": 12,
        "questions_per_chunk": 3,
        "max_generation_attempts": 2,
        "dedupe_threshold": 0.8,
        "temperature": 0.0,
    }
    values.update(overrides)
    return EvalsetConfig(**values)


def test_generator_filters_vague_short_and_duplicate_questions() -> None:
    chunk = make_chunk(
        "The maximum payload size is five megabytes. The Growth plan can monitor fifty robots."
    )
    payload = {
        "questions": [
            {"question": "What is the maximum telemetry payload size?", "answer": "5 MB"},
            {"question": "What is this passage about?", "answer": "Stuff"},
            {"question": "How many robots can the Growth plan monitor?"},
            {"question": "What is the maximum telemetry payload size?", "answer": "5 MB"},
        ]
    }
    client = FakeOllamaClient(payload)
    generator = SyntheticEvalsetGenerator(
        client,  # type: ignore[arg-type]
        generator_config(),
        seed=7,
        show_progress=False,
    )

    examples, stats = generator.generate([chunk])

    assert len(examples) == 2
    assert examples[0].query_id == "q0001"
    assert examples[0].relevant_doc_ids == ["api.md"]
    assert examples[0].source_chunk_id == "api.md#0000"
    assert examples[0].metadata["generator_model"] == "llama3.2"
    assert stats.accepted == 2
    assert stats.rejections["vague"] == 1
    assert stats.rejections["duplicate"] == 1
    assert client.require_calls == 1


def test_generator_rejects_non_json_after_retries() -> None:
    class InvalidClient(FakeOllamaClient):
        def generate(self, model: str, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            return "sorry, I cannot do that"

    generator = SyntheticEvalsetGenerator(
        InvalidClient(),  # type: ignore[arg-type]
        generator_config(),
        seed=1,
        show_progress=False,
    )
    with pytest.raises(EvalsetError, match="no usable questions"):
        generator.generate([make_chunk("A sufficiently long chunk about payload limits.")])


def test_generator_wraps_ollama_connection_error() -> None:
    client = FakeOllamaClient(error=OllamaError("Could not reach Ollama"))
    generator = SyntheticEvalsetGenerator(
        client,  # type: ignore[arg-type]
        generator_config(),
        show_progress=False,
    )
    with pytest.raises(EvalsetError, match="Could not reach Ollama"):
        generator.generate([make_chunk("A sufficiently long chunk about payload limits.")])


def test_generator_requires_long_enough_chunks() -> None:
    generator = SyntheticEvalsetGenerator(
        FakeOllamaClient({"questions": []}),  # type: ignore[arg-type]
        generator_config(min_chunk_chars=20),
        show_progress=False,
    )
    with pytest.raises(EvalsetError, match="long enough"):
        generator.generate([make_chunk("short")])


def test_generator_max_chunks_is_deterministic() -> None:
    class EchoClient(FakeOllamaClient):
        def generate(self, model: str, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            match = re.search(r"marker-([a-z0-9]+)", prompt)
            marker = match.group(1) if match else "none"
            return json.dumps(
                {
                    "questions": [
                        {"question": f"What is chunk marker-{marker} about?", "answer": "x"}
                    ]
                }
            )

    chunks = [
        make_chunk(
            f"Chunk content marker-a{index} with enough detail to generate a question. " * 2,
            doc_id="doc-a.md",
            chunk_id=f"doc-a.md#{index:04d}",
        )
        for index in range(3)
    ] + [
        make_chunk(
            f"Chunk content marker-b{index} with enough detail to generate a question. " * 2,
            doc_id="doc-b.md",
            chunk_id=f"doc-b.md#{index:04d}",
        )
        for index in range(3)
    ]

    def run() -> list[str]:
        generator = SyntheticEvalsetGenerator(
            EchoClient({"questions": []}),  # type: ignore[arg-type]
            generator_config(max_chunks=2, dedupe_threshold=1.0),
            seed=99,
            show_progress=False,
        )
        examples, _ = generator.generate(chunks)
        return [example.source_chunk_id or "" for example in examples]

    assert run() == run()
    assert len(run()) == 2
