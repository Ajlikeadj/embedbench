"""Evaluation-set loading, validation and synthetic question generation."""

from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from embedbench.config import EvalsetConfig
from embedbench.errors import EvalsetError
from embedbench.logging_utils import get_logger
from embedbench.models import Chunk, EvalExample
from embedbench.ollama_client import OllamaClient, OllamaError
from embedbench.utils import atomic_write_text

logger = get_logger(__name__)

_KNOWN_FIELDS = {"query_id", "query", "relevant_doc_ids", "answer", "source_chunk_id", "metadata"}
_PLACEHOLDER_RE = re.compile(
    r"\b(?:this|the|that)\s+(?:document|passage|text|excerpt|article|chunk|context|paragraph)\b",
    re.IGNORECASE,
)
_GENERIC_RE = re.compile(
    r"^(?:"
    r"what\s+(?:is|are)\s+(?:this|the)\s+(?:document|passage|text|article|content)\s+(?:about|discussing|describing)"
    r"|what\s+does\s+(?:this|the)\s+(?:document|passage|text)\s+(?:say|discuss|describe)"
    r"|can\s+you\s+summari[sz]e"
    r"|summari[sz]e\b"
    r"|describe\s+(?:this|the)\s+(?:document|passage|text)"
    r"|what\s+is\s+the\s+main\s+(?:topic|idea|point|subject)"
    r")",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "there", "these", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would", "you", "your",
}


def _normalize_doc_id(value: str) -> str:
    cleaned = value.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _normalize_question(text: str) -> str:
    lowered = text.casefold().strip()
    lowered = re.sub(r"[\"'“”‘’]", "", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def load_evalset(path: Path) -> list[EvalExample]:
    """Load JSONL records with ``query`` and ``relevant_doc_ids``.

    Blank lines are ignored.  Optional fields are ``query_id``, ``answer``,
    ``source_chunk_id`` and ``metadata``; unknown top-level fields are preserved
    in ``metadata``.  Errors include the offending line number.
    """

    path = Path(path)
    if not path.exists():
        raise EvalsetError(f"Evaluation set not found: {path}")
    if not path.is_file():
        raise EvalsetError(f"Evaluation set is not a file: {path}")

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalsetError(f"Could not read evaluation set {path}: {exc}") from exc

    examples: list[EvalExample] = []
    used_ids: set[str] = set()
    seen_queries: dict[str, int] = {}
    next_auto_id = 1
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EvalsetError(
                f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(raw, dict):
            raise EvalsetError(
                f"Invalid record in {path} at line {line_number}: expected a JSON object"
            )

        query_value = raw.get("query")
        if not isinstance(query_value, str) or not query_value.strip():
            raise EvalsetError(
                f"Missing non-empty 'query' string in {path} at line {line_number}"
            )
        query = " ".join(query_value.split())

        relevant_value = raw.get("relevant_doc_ids")
        if not isinstance(relevant_value, list) or not relevant_value:
            raise EvalsetError(
                f"'relevant_doc_ids' must be a non-empty list in {path} at line {line_number}"
            )
        relevant: list[str] = []
        for item in relevant_value:
            if not isinstance(item, str) or not item.strip():
                raise EvalsetError(
                    f"Invalid document ID in 'relevant_doc_ids' at {path}:{line_number}"
                )
            normalized = _normalize_doc_id(item)
            if normalized not in relevant:
                relevant.append(normalized)

        query_id_value = raw.get("query_id")
        if query_id_value is None:
            while f"q{next_auto_id:04d}" in used_ids:
                next_auto_id += 1
            query_id = f"q{next_auto_id:04d}"
            next_auto_id += 1
        elif isinstance(query_id_value, str) and query_id_value.strip():
            query_id = query_id_value.strip()
        else:
            raise EvalsetError(
                f"'query_id' must be a non-empty string in {path} at line {line_number}"
            )
        if query_id in used_ids:
            raise EvalsetError(
                f"Duplicate query_id {query_id!r} in {path} at line {line_number}"
            )
        used_ids.add(query_id)
        auto_match = re.fullmatch(r"q(\d{4,})", query_id)
        if auto_match is not None:
            next_auto_id = max(next_auto_id, int(auto_match.group(1)) + 1)

        answer = raw.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise EvalsetError(f"'answer' must be a string at {path}:{line_number}")
        source_chunk_id = raw.get("source_chunk_id")
        if source_chunk_id is not None and not isinstance(source_chunk_id, str):
            raise EvalsetError(f"'source_chunk_id' must be a string at {path}:{line_number}")

        metadata_value = raw.get("metadata", {})
        if metadata_value is None:
            metadata_value = {}
        if not isinstance(metadata_value, dict):
            raise EvalsetError(f"'metadata' must be an object at {path}:{line_number}")
        metadata = dict(metadata_value)
        for key, value in raw.items():
            if key not in _KNOWN_FIELDS:
                metadata.setdefault(key, value)

        normalized_query = _normalize_question(query)
        if normalized_query in seen_queries:
            logger.warning(
                "Duplicate query text at %s:%d (also line %d); it will be evaluated twice",
                path,
                line_number,
                seen_queries[normalized_query],
            )
        else:
            seen_queries[normalized_query] = line_number

        examples.append(
            EvalExample(
                query_id=query_id,
                query=query,
                relevant_doc_ids=relevant,
                answer=answer,
                source_chunk_id=source_chunk_id,
                metadata=metadata,
            )
        )

    if not examples:
        raise EvalsetError(f"Evaluation set is empty: {path}")
    logger.info("Loaded %d evaluation queries from %s", len(examples), path)
    return examples


def save_evalset(examples: Sequence[EvalExample], path: Path) -> None:
    """Write examples as UTF-8 JSONL, atomically."""

    if not examples:
        raise EvalsetError("Refusing to write an empty evaluation set")
    lines = [json.dumps(example.to_dict(), ensure_ascii=False) for example in examples]
    path = Path(path)
    atomic_write_text(path, "\n".join(lines) + "\n")
    logger.info("Wrote %d evaluation queries to %s", len(examples), path)


def validate_evalset_documents(
    examples: Sequence[EvalExample],
    doc_ids: Iterable[str],
) -> None:
    """Ensure every referenced document exists in the corpus."""

    available = set(doc_ids)
    missing: dict[str, list[str]] = {}
    for example in examples:
        absent = [doc_id for doc_id in example.relevant_doc_ids if doc_id not in available]
        if absent:
            missing[example.query_id] = absent
    if missing:
        lines = []
        for query_id, absent in list(missing.items())[:5]:
            lines.append(f"  - {query_id}: {', '.join(absent[:5])}")
        more = "" if len(missing) <= 5 else f"\n  ... and {len(missing) - 5} more queries"
        raise EvalsetError(
            "Evaluation set references document IDs that are not in the corpus. "
            "Document IDs are paths relative to the docs folder (for example "
            "'guides/install.md'). Check the docs folder or regenerate the eval set.\n"
            + "\n".join(lines)
            + more
        )


class _QuestionDeduplicator:
    """Cheap token-overlap prefilter plus SequenceMatcher for near duplicates."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._normalized: list[str] = []
        self._tokens: list[set[str]] = []

    def is_duplicate(self, question: str) -> bool:
        normalized = _normalize_question(question)
        if not normalized:
            return True
        tokens = set(normalized.split())
        for existing, existing_tokens in zip(self._normalized, self._tokens, strict=False):
            union = tokens | existing_tokens
            jaccard = len(tokens & existing_tokens) / len(union) if union else 0.0
            if jaccard < 0.35:
                continue
            if difflib.SequenceMatcher(None, normalized, existing).ratio() >= self.threshold:
                return True
        self._normalized.append(normalized)
        self._tokens.append(tokens)
        return False


@dataclass(slots=True)
class GenerationStats:
    """Diagnostics from synthetic eval-set generation."""

    chunks_available: int = 0
    chunks_used: int = 0
    chunks_failed: int = 0
    candidates_returned: int = 0
    accepted: int = 0
    invalid_responses: int = 0
    rejections: dict[str, int] = field(
        default_factory=lambda: {
            "not_a_question": 0,
            "too_short": 0,
            "too_long": 0,
            "vague": 0,
            "duplicate": 0,
        }
    )

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_available": self.chunks_available,
            "chunks_used": self.chunks_used,
            "chunks_failed": self.chunks_failed,
            "candidates_returned": self.candidates_returned,
            "accepted": self.accepted,
            "invalid_responses": self.invalid_responses,
            "rejections": dict(self.rejections),
        }


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("no JSON object or array found")


def parse_generated_questions(raw: str) -> list[dict[str, str]]:
    """Extract question/answer objects from a model response."""

    value = _extract_json(raw)
    if isinstance(value, dict):
        if isinstance(value.get("questions"), list):
            items = value["questions"]
        elif isinstance(value.get("question"), str):
            items = [value]
        else:
            raise ValueError("JSON object has no 'questions' list or 'question' field")
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError("JSON response is not an object or array")

    questions: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            question = item.strip()
            answer = ""
        elif isinstance(item, dict) and isinstance(item.get("question"), str):
            question = item["question"].strip()
            answer_value = item.get("answer", "")
            answer = answer_value.strip() if isinstance(answer_value, str) else ""
        else:
            continue
        if question:
            questions.append({"question": question, "answer": answer})
    if not questions:
        raise ValueError("JSON response contained no usable questions")
    return questions


class SyntheticEvalsetGenerator:
    """Generate one or more filtered questions per chunk with a local LLM."""

    def __init__(
        self,
        client: OllamaClient,
        config: EvalsetConfig,
        *,
        seed: int = 42,
        show_progress: bool = True,
        console: Console | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.seed = seed
        self.show_progress = show_progress
        self.console = console or Console(stderr=True)

    def _prompt(self, chunk: Chunk) -> str:
        count = self.config.questions_per_chunk
        plural = "questions" if count > 1 else "question"
        return (
            "You are building a retrieval evaluation set for a RAG system. "
            f"Read the passage and write {count} specific, self-contained {plural} "
            "that can be answered using only the passage.\n"
            "Rules:\n"
            "- Never mention \"this passage\", \"the document\", \"the text\", or \"the context\".\n"
            "- Ask about concrete facts, names, numbers, steps, definitions, or policies.\n"
            "- Each question must make sense on its own without seeing the passage.\n"
            "- Each question must end with a question mark and be distinct.\n"
            "- Provide a short answer drawn from the passage.\n"
            "Return JSON only, exactly in this shape:\n"
            '{"questions":[{"question":"...","answer":"..."}]}\n\n'
            f"Document: {chunk.doc_id}\n"
            'Passage:\n"""\n'
            f"{chunk.text}\n"
            '"""'
        )

    def _request_questions(self, chunk: Chunk, attempt: int) -> str:
        return self.client.generate(
            self.config.ollama_model,
            self._prompt(chunk),
            temperature=self.config.temperature,
            seed=self.seed + attempt,
            num_predict=self.config.num_predict,
        )

    def _filter_question(self, question: str, stats: GenerationStats) -> str | None:
        cleaned = " ".join(question.split()).strip().strip("\"'")
        if not cleaned.endswith("?"):
            stats.reject("not_a_question")
            return None
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", cleaned)
        if len(words) < self.config.min_question_words:
            stats.reject("too_short")
            return None
        if len(words) > self.config.max_question_words:
            stats.reject("too_long")
            return None
        if _GENERIC_RE.search(cleaned) or _PLACEHOLDER_RE.search(cleaned):
            stats.reject("vague")
            return None
        content_words = [
            word for word in words if len(word) >= 3 and word.casefold() not in _STOPWORDS
        ]
        if not content_words:
            stats.reject("vague")
            return None
        return cleaned

    def generate(self, chunks: Sequence[Chunk]) -> tuple[list[EvalExample], GenerationStats]:
        eligible = [
            chunk
            for chunk in chunks
            if len(chunk.text.strip()) >= self.config.min_chunk_chars
        ]
        stats = GenerationStats(chunks_available=len(eligible))
        if not eligible:
            raise EvalsetError(
                "No chunks are long enough for question generation. "
                f"Lower evalset.min_chunk_chars (currently {self.config.min_chunk_chars}) "
                "or use larger chunks."
            )

        ordered = sorted(eligible, key=lambda chunk: chunk.chunk_id)
        if self.config.max_chunks is not None and len(ordered) > self.config.max_chunks:
            sampled = random.Random(self.seed).sample(ordered, self.config.max_chunks)
            ordered = sorted(sampled, key=lambda chunk: chunk.chunk_id)
        stats.chunks_used = len(ordered)

        try:
            self.client.require_model(self.config.ollama_model)
        except OllamaError as exc:
            raise EvalsetError(str(exc)) from exc

        deduplicator = _QuestionDeduplicator(self.config.dedupe_threshold)
        examples: list[EvalExample] = []
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        use_progress = self.show_progress and self.console.is_terminal
        task_id = None
        if use_progress:
            progress.start()
            task_id = progress.add_task("Generating questions", total=len(ordered))

        try:
            for chunk in ordered:
                raw_candidates: list[dict[str, str]] | None = None
                for attempt in range(self.config.max_generation_attempts):
                    try:
                        raw = self._request_questions(chunk, attempt)
                    except OllamaError as exc:
                        raise EvalsetError(f"Question generation failed: {exc}") from exc
                    try:
                        raw_candidates = parse_generated_questions(raw)
                        break
                    except ValueError as exc:
                        stats.invalid_responses += 1
                        logger.debug(
                            "Invalid JSON for %s (attempt %d): %s", chunk.chunk_id, attempt + 1, exc
                        )

                if raw_candidates is None:
                    stats.chunks_failed += 1
                    logger.warning("No valid questions generated for chunk %s", chunk.chunk_id)
                    if task_id is not None:
                        progress.advance(task_id)
                    continue

                stats.candidates_returned += len(raw_candidates)
                accepted_for_chunk = 0
                for candidate in raw_candidates:
                    if accepted_for_chunk >= self.config.questions_per_chunk:
                        break
                    question = self._filter_question(candidate["question"], stats)
                    if question is None:
                        continue
                    if deduplicator.is_duplicate(question):
                        stats.reject("duplicate")
                        continue
                    examples.append(
                        EvalExample(
                            query_id=f"q{len(examples) + 1:04d}",
                            query=question,
                            relevant_doc_ids=[chunk.doc_id],
                            answer=candidate.get("answer") or None,
                            source_chunk_id=chunk.chunk_id,
                            metadata={
                                "generator": "ollama",
                                "generator_model": self.config.ollama_model,
                            },
                        )
                    )
                    accepted_for_chunk += 1
                    stats.accepted += 1
                if task_id is not None:
                    progress.advance(task_id)
        finally:
            if use_progress:
                progress.stop()

        if not examples:
            raise EvalsetError(
                "Synthetic generation produced no usable questions. "
                "Try a stronger model (for example qwen2.5:7b or llama3.1:8b), "
                "larger chunks, or relax the question filters."
            )
        rejections = ", ".join(
            f"{reason}={count}" for reason, count in stats.rejections.items() if count
        )
        logger.info(
            "Generated %d questions from %d chunks (%s)",
            len(examples),
            stats.chunks_used,
            rejections or "no rejections",
        )
        return examples, stats
