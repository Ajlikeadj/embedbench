"""Retrieval datasets and synthetic stress-workload generators.

A :class:`RetrievalDataset` is the in-memory unit every evaluator consumes: a
searchable set of documents plus queries that name the documents a good system
should return.

Two shapes of relevance are supported:

* Flat: every document is its own retrieval unit (``group_id`` is unset).
* Grouped: several documents (chunks) belong to one logical document.  Metrics
  are computed on the de-duplicated group ranking, so a query can list a
  document ID even though the index stores its chunks.  This is how the
  file-based pipeline keeps document-level relevance while indexing chunks.

The generators in this module build the stress workloads requested by
enterprise retrieval reviews: needle-in-a-haystack at increasing context
lengths, chunk-size sensitivity sweeps, and noisy or paraphrased queries.
Every generator is deterministic for a given seed.

JSONL format (one object per line, ``type`` selects the kind)::

    {"type": "document", "doc_id": "d1", "text": "...", "group_id": null}
    {"type": "query", "query_id": "q1", "query": "...", "relevant_doc_ids": ["d1"]}

Legacy eval-set files (query objects without a ``type``) load as queries.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embedbench.chunking import chunk_documents
from embedbench.config import ChunkingConfig, IngestionConfig
from embedbench.errors import DatasetError
from embedbench.evalset import load_evalset, validate_evalset_documents
from embedbench.ingest import load_documents
from embedbench.logging_utils import get_logger
from embedbench.models.base import heuristic_token_count
from embedbench.schemas import Chunk, Document
from embedbench.utils import atomic_write_text

logger = get_logger(__name__)

DEFAULT_CONTEXT_SIZES: tuple[int, ...] = (2_048, 8_192, 32_768)
"""Haystack sizes in estimated tokens, as requested by the upgrade brief."""

DEFAULT_DEPTHS: tuple[float, ...] = (0.0, 0.5, 1.0)
"""Insertion depths as a fraction of the haystack (0 = start, 1 = end)."""

DEFAULT_CHUNK_TOKENS: tuple[int, ...] = (128, 256, 512, 1_024)
"""Chunk sizes in estimated tokens for sensitivity sweeps."""

DEFAULT_NOISE_KINDS: tuple[str, ...] = ("typo", "case", "truncate", "shuffle", "paraphrase")
"""Deterministic query degradations applied by :func:`query_variants`."""

CHARS_PER_TOKEN = 4.0
"""Approximate characters per token used to translate token budgets."""


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetDocument:
    """One searchable unit.

    Attributes:
        doc_id: Unique identifier inside the dataset.
        text: Content that will be embedded and searched.
        group_id: Logical document this unit belongs to.  ``None`` means the
            unit is its own document.  Useful when indexing chunks.
        metadata: Free-form annotations copied into reports.
    """

    doc_id: str
    text: str
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def group(self) -> str:
        """Group identifier with the ``doc_id`` fallback applied."""

        return self.group_id or self.doc_id

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping in the JSONL schema."""

        payload: dict[str, Any] = {
            "type": "document",
            "doc_id": self.doc_id,
            "text": self.text,
        }
        if self.group_id is not None:
            payload["group_id"] = self.group_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """A query plus the document IDs that should be retrieved for it.

    Attributes:
        query_id: Unique identifier inside the dataset.
        query: The user-visible search text.
        relevant_doc_ids: Document (group) IDs considered relevant.
        metadata: Free-form annotations, for example the haystack depth.
    """

    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping in the JSONL schema."""

        payload: dict[str, Any] = {
            "type": "query",
            "query_id": self.query_id,
            "query": self.query,
            "relevant_doc_ids": list(self.relevant_doc_ids),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class RetrievalDataset:
    """Documents, queries and relevance judgements for one evaluation.

    Attributes:
        name: Human-readable dataset name used in reports.
        documents: Searchable units.
        queries: Queries with their relevance judgements.
        metadata: Dataset-level annotations (generator settings, source paths).
    """

    name: str
    documents: list[DatasetDocument]
    queries: list[RetrievalQuery]
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- derived views ----------------------------------------------------

    @property
    def num_documents(self) -> int:
        """Number of searchable units."""

        return len(self.documents)

    @property
    def num_queries(self) -> int:
        """Number of queries."""

        return len(self.queries)

    @property
    def num_groups(self) -> int:
        """Number of distinct logical documents."""

        return len({document.group for document in self.documents})

    def group_lookup(self) -> dict[str, str]:
        """Map every document ID to its logical document (group) ID.

        Returns:
            A mapping usable to convert a chunk ranking into a document ranking.
        """

        return {document.doc_id: document.group for document in self.documents}

    def group_ranking(
        self,
        ranked_doc_ids: Sequence[str],
        limit: int | None = None,
    ) -> list[str]:
        """Collapse a document ranking into a de-duplicated group ranking.

        Args:
            ranked_doc_ids: Retrieved document IDs, best first.
            limit: Keep at most this many groups.

        Returns:
            Group IDs in first-occurrence order.
        """

        lookup = self.group_lookup()
        seen: set[str] = set()
        ranking: list[str] = []
        for doc_id in ranked_doc_ids:
            group = lookup.get(doc_id, doc_id)
            if not group or group in seen:
                continue
            seen.add(group)
            ranking.append(group)
            if limit is not None and len(ranking) >= limit:
                break
        return ranking

    def stats(self) -> dict[str, Any]:
        """Return size and shape statistics for reports.

        Returns:
            Counts, character totals and query-length averages.
        """

        lengths = [len(document.text) for document in self.documents]
        query_words = [len(query.query.split()) for query in self.queries]
        relevant = [len(query.relevant_doc_ids) for query in self.queries]
        return {
            "documents": self.num_documents,
            "groups": self.num_groups,
            "queries": self.num_queries,
            "chars": sum(lengths),
            "estimated_tokens": int(sum(lengths) / CHARS_PER_TOKEN),
            "avg_doc_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "avg_query_words": round(sum(query_words) / len(query_words), 2)
            if query_words
            else 0.0,
            "avg_relevant_per_query": round(sum(relevant) / len(relevant), 2) if relevant else 0.0,
        }

    # -- validation -------------------------------------------------------

    def validate(self) -> None:
        """Check internal consistency.

        Raises:
            DatasetError: If documents or queries are missing, identifiers are
                duplicated, a document is empty, or a query names a document
                that is not in the dataset.
        """

        if not self.documents:
            raise DatasetError(f"Dataset {self.name!r} has no documents")
        if not self.queries:
            raise DatasetError(f"Dataset {self.name!r} has no queries")

        known: set[str] = set()
        for position, document in enumerate(self.documents):
            if not document.doc_id:
                raise DatasetError(f"Document {position} in {self.name!r} has an empty doc_id")
            if document.doc_id in known:
                raise DatasetError(f"Duplicate doc_id {document.doc_id!r} in {self.name!r}")
            if not document.text.strip():
                raise DatasetError(f"Document {document.doc_id!r} in {self.name!r} is empty")
            known.add(document.doc_id)

        groups = {document.group for document in self.documents}
        query_ids: set[str] = set()
        for query in self.queries:
            if not query.query_id:
                raise DatasetError(f"Dataset {self.name!r} has a query with an empty query_id")
            if query.query_id in query_ids:
                raise DatasetError(f"Duplicate query_id {query.query_id!r} in {self.name!r}")
            if not query.query.strip():
                raise DatasetError(f"Query {query.query_id!r} in {self.name!r} is empty")
            if not query.relevant_doc_ids:
                raise DatasetError(
                    f"Query {query.query_id!r} in {self.name!r} has no relevant documents"
                )
            missing = [doc_id for doc_id in query.relevant_doc_ids if doc_id not in groups]
            if missing:
                raise DatasetError(
                    f"Query {query.query_id!r} references unknown documents: "
                    + ", ".join(missing[:5])
                )
            query_ids.add(query.query_id)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation (documents are omitted)."""

        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "stats": self.stats(),
            "queries": [query.to_dict() for query in self.queries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetrievalDataset:
        """Rebuild a dataset from :meth:`to_dict` output plus documents.

        Args:
            payload: Mapping with ``name``, ``documents``, ``queries`` and
                optional ``metadata``.

        Returns:
            The dataset.

        Raises:
            DatasetError: If required keys are missing.
        """

        try:
            documents = [
                DatasetDocument(
                    doc_id=str(item["doc_id"]),
                    text=str(item["text"]),
                    group_id=item.get("group_id"),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in payload["documents"]
            ]
            queries = [
                RetrievalQuery(
                    query_id=str(item["query_id"]),
                    query=str(item["query"]),
                    relevant_doc_ids=tuple(str(value) for value in item["relevant_doc_ids"]),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in payload["queries"]
            ]
        except KeyError as exc:
            raise DatasetError(f"Dataset payload is missing key {exc.args[0]!r}") from exc
        return cls(
            name=str(payload.get("name") or "dataset"),
            documents=documents,
            queries=queries,
            metadata=dict(payload.get("metadata") or {}),
        )

    def save_jsonl(self, path: Path) -> Path:
        """Write the dataset as JSONL, documents first.

        Args:
            path: Destination file.

        Returns:
            The path that was written.
        """

        lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in self.documents]
        lines.extend(json.dumps(query.to_dict(), ensure_ascii=False) for query in self.queries)
        atomic_write_text(Path(path), "\n".join(lines) + "\n")
        return Path(path)


# ---------------------------------------------------------------------------
# Loading and conversion
# ---------------------------------------------------------------------------


def load_jsonl_dataset(path: Path, *, name: str | None = None) -> RetrievalDataset:
    """Load a dataset from the JSONL schema described in the module docstring.

    Args:
        path: Source file.  Eval-set files with plain query objects are also
            accepted; their documents must come from elsewhere.
        name: Override the dataset name (defaults to the file stem).

    Returns:
        The dataset, validated.

    Raises:
        DatasetError: If the file is unreadable, malformed or inconsistent.
    """

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read dataset {source}: {exc}") from exc

    documents: list[DatasetDocument] = []
    queries: list[RetrievalQuery] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{source}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise DatasetError(f"{source}:{line_number}: expected a JSON object")
        kind = record.get("type")
        if kind == "document":
            try:
                documents.append(
                    DatasetDocument(
                        doc_id=str(record["doc_id"]),
                        text=str(record["text"]),
                        group_id=record.get("group_id"),
                        metadata=dict(record.get("metadata") or {}),
                    )
                )
            except KeyError as exc:
                raise DatasetError(
                    f"{source}:{line_number}: document is missing {exc.args[0]!r}"
                ) from exc
        elif kind == "query" or ("query" in record and "relevant_doc_ids" in record):
            try:
                queries.append(
                    RetrievalQuery(
                        query_id=str(record["query_id"]),
                        query=str(record["query"]),
                        relevant_doc_ids=tuple(str(value) for value in record["relevant_doc_ids"]),
                        metadata=dict(record.get("metadata") or {}),
                    )
                )
            except KeyError as exc:
                raise DatasetError(
                    f"{source}:{line_number}: query is missing {exc.args[0]!r}"
                ) from exc
        else:
            raise DatasetError(
                f"{source}:{line_number}: unknown record type {kind!r} "
                "(expected 'document' or 'query')"
            )

    dataset = RetrievalDataset(
        name=name or source.stem,
        documents=documents,
        queries=queries,
        metadata={"source": str(source)},
    )
    dataset.validate()
    return dataset


def load_documents_dataset(
    docs_dir: Path,
    *,
    ingestion: IngestionConfig | None = None,
) -> list[DatasetDocument]:
    """Load a document folder as unchunked dataset documents.

    Args:
        docs_dir: Folder of supported documents.
        ingestion: Discovery settings.

    Returns:
        One dataset document per source file, in discovery order.

    Raises:
        DatasetError: If the folder yields no documents.
    """

    result = load_documents(Path(docs_dir), ingestion or IngestionConfig())
    documents = [
        DatasetDocument(
            doc_id=document.doc_id, text=document.text, metadata=dict(document.metadata)
        )
        for document in result.documents
    ]
    if not documents:
        raise DatasetError(f"No documents could be loaded from {docs_dir}")
    return documents


def load_queries(evalset_path: Path) -> list[RetrievalQuery]:
    """Load an eval-set file as retrieval queries.

    Args:
        evalset_path: JSONL eval set with ``query`` and ``relevant_doc_ids``.

    Returns:
        The queries in file order.

    Raises:
        DatasetError: If the file cannot be parsed.
    """

    try:
        examples = load_evalset(Path(evalset_path))
    except Exception as exc:
        raise DatasetError(f"Could not load eval set {evalset_path}: {exc}") from exc
    queries = [
        RetrievalQuery(
            query_id=example.query_id,
            query=example.query,
            relevant_doc_ids=tuple(example.relevant_doc_ids),
            metadata=dict(example.metadata),
        )
        for example in examples
    ]
    if not queries:
        raise DatasetError(f"Eval set {evalset_path} contains no queries")
    return queries


def load_corpus_dataset(
    docs_dir: Path,
    evalset_path: Path,
    *,
    ingestion: IngestionConfig | None = None,
    chunking: ChunkingConfig | None = None,
    name: str | None = None,
) -> RetrievalDataset:
    """Build a chunk-indexed dataset from a document folder and an eval set.

    Each chunk becomes a searchable document whose ``group_id`` is the source
    document, so relevance stays document-level exactly like the file-based
    pipeline.

    Args:
        docs_dir: Folder of ``.txt``/``.md``/``.pdf``/``.docx``/``.html`` files.
        evalset_path: JSONL eval set with ``query`` and ``relevant_doc_ids``.
        ingestion: Document discovery settings.
        chunking: Chunking settings.
        name: Dataset name used in reports.

    Returns:
        A validated dataset.

    Raises:
        DatasetError: If the corpus produces no chunks or the eval set
            references documents that produced no chunks.
    """

    ingestion_config = ingestion or IngestionConfig()
    chunking_config = chunking or ChunkingConfig()
    result = load_documents(Path(docs_dir), ingestion_config)
    chunks = chunk_documents(result.documents, chunking_config)
    if not chunks:
        raise DatasetError("Documents were loaded but produced no chunks")
    examples = load_evalset(Path(evalset_path))
    validate_evalset_documents(examples, {document.doc_id for document in result.documents})
    documents = [
        DatasetDocument(
            doc_id=chunk.chunk_id,
            text=chunk.text,
            group_id=chunk.doc_id,
            metadata={"start_char": chunk.start_char, "end_char": chunk.end_char},
        )
        for chunk in chunks
    ]
    queries = [
        RetrievalQuery(
            query_id=example.query_id,
            query=example.query,
            relevant_doc_ids=tuple(example.relevant_doc_ids),
            metadata=dict(example.metadata),
        )
        for example in examples
    ]
    dataset = RetrievalDataset(
        name=name or f"{Path(docs_dir).name}::{Path(evalset_path).stem}",
        documents=documents,
        queries=queries,
        metadata={
            "source": "corpus",
            "docs_dir": str(docs_dir),
            "evalset": str(evalset_path),
            "documents_ingested": len(result.documents),
            "chunks": len(chunks),
            **chunking_config.model_dump(mode="json"),
        },
    )
    dataset.validate()
    return dataset


# ---------------------------------------------------------------------------
# Filler corpus and synthetic needles
# ---------------------------------------------------------------------------

FILLER_PARAGRAPHS: tuple[str, ...] = (
    "The procurement team reviews supplier agreements every quarter. Each review "
    "covers delivery windows, penalties for late shipments, and the escalation "
    "contacts listed in the contract appendix. Regional managers forward their "
    "updates to the central desk before the tenth of the month.",
    "Campus facilities keep a maintenance calendar for lecture halls and labs. "
    "Requests arrive through the ticketing portal and are triaged by severity. "
    "A blocked doorway or a failed fire alarm is handled the same day, while "
    "cosmetic repairs wait for the next scheduled window.",
    "Analysts publish a weekly dashboard covering visitor traffic, conversion "
    "funnels, and the share of returning customers. The numbers are reconciled "
    "against the billing export before the Monday briefing so that finance and "
    "marketing quote the same figures.",
    "The data platform ingests event streams from three regions and stores "
    "them in a partitioned warehouse. Late arriving records are accepted for "
    "twenty four hours, after which they are written to a quarantine table for "
    "manual inspection by the on call engineer.",
    "During onboarding, new employees complete a short module on records "
    "retention. The policy distinguishes drafts from final versions and names "
    "the custodian responsible for each archive. Questions go to the compliance "
    "mailbox, which answers within two business days.",
    "Field engineers carry a checklist for every site visit. The checklist "
    "covers safety briefings, equipment serial numbers, and the calibration "
    "sticker on each sensor. Completed checklists are scanned into the asset "
    "management system before the engineer leaves the site.",
    "The library subscribes to a handful of journals and rotates the display "
    "shelf each month. Requests for interlibrary loans are handled by the "
    "reference desk, which keeps a waiting list and notifies readers when the "
    "requested volume arrives.",
    "A small research group publishes notes about soil moisture sensors. The "
    "notes describe calibration drift, battery replacement intervals, and the "
    "way shaded plots report lower temperatures than exposed ones.",
)

_NEEDLE_QUESTIONS: tuple[str, ...] = (
    "What is the maintenance code for pump {index}?",
    "Which access phrase unlocks archive cabinet {index}?",
    "What is the calibration token for sensor bank {index}?",
    "Which retrieval code belongs to locker {index}?",
)


@dataclass(frozen=True, slots=True)
class Needle:
    """A synthetic fact hidden in a haystack.

    Attributes:
        needle_id: Identifier stored in query metadata.
        sentence: The fact, including a unique code.
        question: Question whose answer is the fact.
        code: The unique code the model must surface.
    """

    needle_id: str
    sentence: str
    question: str
    code: str


def make_needle(rng: random.Random, index: int) -> Needle:
    """Build a unique, single-sentence fact and its question.

    Args:
        rng: Seeded random source.
        index: Needle ordinal, used in the question text.

    Returns:
        The needle.
    """

    code = f"{rng.choice(['ZX', 'QR', 'VK', 'MB'])}-{rng.randint(1000, 9999)}"
    template = _NEEDLE_QUESTIONS[index % len(_NEEDLE_QUESTIONS)]
    question = template.format(index=index + 1)
    sentence = (
        f"Note for operators: the record referenced by question {index + 1} uses "
        f"code {code}, and this value is not repeated anywhere else in the file."
    )
    return Needle(
        needle_id=f"needle-{index + 1}",
        sentence=sentence,
        question=question,
        code=code,
    )


def _filler_iter(rng: random.Random) -> Iterator[str]:
    paragraphs = list(FILLER_PARAGRAPHS)
    while True:
        rng.shuffle(paragraphs)
        yield from paragraphs


def build_haystack(
    target_tokens: int,
    needles: Sequence[tuple[float, Needle]],
    *,
    seed: int = 42,
) -> tuple[str, list[tuple[Needle, int, int]]]:
    """Assemble a long context with needles inserted at fractional depths.

    Args:
        target_tokens: Approximate haystack size in estimated tokens.
        needles: ``(depth, needle)`` pairs; depth 0 is the start, 1 the end.
        seed: Seed for filler ordering.

    Returns:
        The haystack text and, for each needle, its character span.

    Raises:
        DatasetError: If ``target_tokens`` is too small or a depth is outside
            ``[0, 1]``.
    """

    if target_tokens < 64:
        raise DatasetError("target_tokens must be >= 64")
    placements = sorted(needles, key=lambda item: item[0])
    for depth, _needle in placements:
        if not 0.0 <= depth <= 1.0:
            raise DatasetError("needle depth must be between 0 and 1")

    rng = random.Random(seed)
    filler = _filler_iter(rng)

    # Build paragraphs until the token budget is filled, tracking how many
    # tokens precede each paragraph so needles land at the requested depth.
    paragraphs: list[str] = []
    tokens_before: list[int] = []
    tokens = 0
    while tokens < target_tokens:
        paragraph = next(filler)
        tokens_before.append(tokens)
        paragraphs.append(paragraph)
        tokens += heuristic_token_count([paragraph])

    # Convert depth fractions into paragraph slots.
    slots: dict[int, list[Needle]] = {}
    total = max(tokens, 1)
    for depth, needle in placements:
        target = depth * total
        slot = 0
        while slot + 1 < len(tokens_before) and tokens_before[slot + 1] <= target:
            slot += 1
        slots.setdefault(slot, []).append(needle)

    parts: list[str] = []
    spans: list[tuple[Needle, int, int]] = []
    cursor = 0
    for position, paragraph in enumerate(paragraphs):
        for needle in slots.get(position, []):
            parts.append(needle.sentence)
            spans.append((needle, cursor, cursor + len(needle.sentence)))
            # Every part is followed by the "\n\n" separator.
            cursor += len(needle.sentence) + 2
        parts.append(paragraph)
        cursor += len(paragraph) + 2

    return "\n\n".join(parts), spans


def needle_in_haystack(
    *,
    context_sizes: Sequence[int] = DEFAULT_CONTEXT_SIZES,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    seed: int = 42,
) -> dict[int, RetrievalDataset]:
    """Generate one needle-in-haystack dataset per context size.

    Each dataset holds the chunked haystack and one query per inserted needle.
    The relevant document is the single chunk that contains the needle, so
    ``hit_rate@k`` answers "did the retriever surface the needle at all?".

    Args:
        context_sizes: Haystack sizes in estimated tokens.
        depths: Insertion depths as fractions of the haystack.
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        seed: Seed for filler and needle codes.

    Returns:
        ``{context_tokens: dataset}``.  Query metadata carries ``depth`` and
        ``needle_id`` so results can be grouped.

    Raises:
        DatasetError: If any context size is too small to hold the needles.
    """

    datasets: dict[int, RetrievalDataset] = {}
    for size_index, context_size in enumerate(context_sizes):
        rng = random.Random(seed + size_index)
        needled: list[tuple[float, Needle]] = []
        for depth_index, depth in enumerate(depths):
            needled.append((depth, make_needle(rng, depth_index)))
        text, spans = build_haystack(context_size, needled, seed=seed + size_index)

        source = Document(
            doc_id=f"haystack-{context_size}",
            path=Path(f"haystack-{context_size}.txt"),
            text=text,
            source_type="synthetic",
        )
        chunks = chunk_documents(
            [source], ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        documents = [
            DatasetDocument(
                doc_id=chunk.chunk_id,
                text=chunk.text,
                metadata={"start_char": chunk.start_char, "end_char": chunk.end_char},
            )
            for chunk in chunks
        ]

        queries: list[RetrievalQuery] = []
        for needle, start, end in spans:
            containing = _chunk_for_span(chunks, start, end)
            if containing is None:  # pragma: no cover - defensive
                raise DatasetError("Could not locate the needle in the generated chunks")
            depth = _depth_for_needle(needled, needle)
            queries.append(
                RetrievalQuery(
                    query_id=f"{needle.needle_id}-{context_size}",
                    query=needle.question,
                    relevant_doc_ids=(containing.chunk_id,),
                    metadata={
                        "depth": depth,
                        "needle_id": needle.needle_id,
                        "context_tokens": context_size,
                        "code": needle.code,
                    },
                )
            )

        dataset = RetrievalDataset(
            name=f"needle-in-a-haystack-{context_size}",
            documents=documents,
            queries=queries,
            metadata={
                "generator": "needle_in_haystack",
                "context_tokens": context_size,
                "requested_tokens": context_size,
                "actual_tokens": heuristic_token_count([text]),
                "chars": len(text),
                "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                "depths": list(depths),
                "seed": seed,
            },
        )
        dataset.validate()
        datasets[context_size] = dataset
    return datasets


def _chunk_for_span(chunks: Sequence[Chunk], start: int, end: int) -> Chunk | None:
    """Return the chunk whose span contains ``[start, end)``."""

    best: Chunk | None = None
    for chunk in chunks:
        if (
            chunk.start_char <= start
            and chunk.end_char >= end
            and (
                best is None
                or (chunk.end_char - chunk.start_char) < (best.end_char - best.start_char)
            )
        ):
            best = chunk
    if best is not None:
        return best
    for chunk in chunks:
        if chunk.start_char < end and chunk.end_char > start:
            return chunk
    return None


def _depth_for_needle(needled: Sequence[tuple[float, Needle]], needle: Needle) -> float:
    for depth, candidate in needled:
        if candidate.needle_id == needle.needle_id:
            return depth
    return 0.0


# ---------------------------------------------------------------------------
# Chunk-size sensitivity
# ---------------------------------------------------------------------------


def chunk_size_sensitivity(
    documents: Sequence[DatasetDocument],
    queries: Sequence[RetrievalQuery],
    *,
    chunk_tokens: Sequence[int] = DEFAULT_CHUNK_TOKENS,
    chars_per_token: float = CHARS_PER_TOKEN,
    overlap_ratio: float = 0.1,
    name: str = "chunk-size-sensitivity",
) -> dict[int, RetrievalDataset]:
    """Re-chunk a corpus at several sizes, keeping relevance document-level.

    Args:
        documents: Source documents (each becomes one or more chunks).
        queries: Queries whose ``relevant_doc_ids`` name source documents.
        chunk_tokens: Chunk sizes in estimated tokens.
        chars_per_token: Conversion used to turn tokens into characters.
        overlap_ratio: Chunk overlap as a fraction of the chunk size.
        name: Dataset name prefix.

    Returns:
        ``{chunk_tokens: dataset}`` with chunk IDs of the form
        ``"<doc_id>#chunk-3"`` and ``group_id`` set to the source document.

    Raises:
        DatasetError: If no documents are supplied or a chunk size is invalid.
    """

    if not documents:
        raise DatasetError("chunk_size_sensitivity needs at least one document")
    datasets: dict[int, RetrievalDataset] = {}
    for tokens in chunk_tokens:
        if tokens < 16:
            raise DatasetError("chunk size must be >= 16 tokens")
        chars = max(100, round(tokens * chars_per_token))
        overlap = max(0, round(chars * overlap_ratio))
        if overlap >= chars:
            overlap = chars // 4
        config = ChunkingConfig(chunk_size=chars, chunk_overlap=overlap)

        sources = [
            Document(
                doc_id=document.doc_id,
                path=Path(f"{document.doc_id}.txt"),
                text=document.text,
                source_type="dataset",
            )
            for document in documents
        ]
        chunks = chunk_documents(sources, config)
        if not chunks:
            raise DatasetError(f"Chunking at {tokens} tokens produced no chunks")
        chunked = [
            DatasetDocument(
                doc_id=f"{chunk.doc_id}#chunk-{chunk.ordinal}",
                text=chunk.text,
                group_id=chunk.doc_id,
                metadata={"start_char": chunk.start_char, "end_char": chunk.end_char},
            )
            for chunk in chunks
        ]
        dataset = RetrievalDataset(
            name=f"{name}-{tokens}",
            documents=chunked,
            queries=list(queries),
            metadata={
                "generator": "chunk_size_sensitivity",
                "chunk_tokens": tokens,
                "chunk_chars": chars,
                "chunk_overlap": overlap,
                "chunks": len(chunked),
                "source_documents": len(documents),
            },
        )
        dataset.validate()
        datasets[tokens] = dataset
    return datasets


# ---------------------------------------------------------------------------
# Noisy, paraphrased and degraded queries
# ---------------------------------------------------------------------------


_SYNONYMS: dict[str, str] = {
    "what": "which",
    "how": "in what way",
    "show": "list",
    "find": "locate",
    "tell": "explain",
    "need": "require",
    "use": "apply",
    "get": "obtain",
    "code": "identifier",
    "phrase": "passphrase",
    "document": "file",
    "procedure": "process",
    "policy": "rule",
    "employee": "staff member",
    "company": "organisation",
}

_PARAPHRASE_LEADS: tuple[str, ...] = (
    "Quick question: ",
    "I was wondering, ",
    "Could you tell me ",
    "For our records, ",
    "Just to confirm, ",
)


def _typo(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    index = rng.randrange(len(words))
    word = words[index]
    if len(word) >= 4:
        position = rng.randrange(1, len(word) - 1)
        chars = list(word)
        chars[position], chars[position - 1] = chars[position - 1], chars[position]
        words[index] = "".join(chars)
    elif word:
        words[index] = word[:-1]
    return " ".join(words)


def _shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    middle = words[1:-1]
    rng.shuffle(middle)
    return " ".join([words[0], *middle, words[-1]])


def _truncate_query(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    keep = max(2, int(len(words) * rng.uniform(0.4, 0.7)))
    return " ".join(words[:keep])


def _random_case(text: str, rng: random.Random) -> str:
    return "".join(char.upper() if rng.random() < 0.3 else char.lower() for char in text)


def _paraphrase(text: str, rng: random.Random) -> str:
    words = text.split()
    swapped = [_SYNONYMS.get(word.lower().rstrip("?,.!"), word) for word in words]
    body = " ".join(swapped).strip()
    body = body[0].lower() + body[1:] if body else body
    lead = rng.choice(_PARAPHRASE_LEADS)
    return f"{lead}{body.rstrip('?')}?"


_NOISE_TRANSFORMS = {
    "typo": _typo,
    "shuffle": _shuffle_words,
    "truncate": _truncate_query,
    "case": _random_case,
    "paraphrase": _paraphrase,
}


def _stable_rng(*parts: object) -> random.Random:
    """Return a process-independent seeded RNG.

    ``hash()`` is salted per process for strings, which would make generated
    queries differ between runs; a SHA-256 digest keeps them stable.
    """

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def degrade_query(text: str, kind: str, seed: int = 42) -> str:
    """Apply one deterministic degradation to a query.

    Args:
        text: Original query.
        kind: One of ``typo``, ``shuffle``, ``truncate``, ``case``,
            ``paraphrase``, ``noise`` (typo + case) or ``clean``.
        seed: Seed for the deterministic transform.

    Returns:
        The degraded query.

    Raises:
        DatasetError: If ``kind`` is unknown.
    """

    if kind == "clean":
        return text
    rng = _stable_rng(text, kind, seed)
    if kind == "noise":
        return _random_case(_typo(text, rng), rng)
    transform = _NOISE_TRANSFORMS.get(kind)
    if transform is None:
        known = ", ".join(["clean", "noise", *_NOISE_TRANSFORMS])
        raise DatasetError(f"Unknown query noise kind {kind!r}. Choose one of: {known}")
    return transform(text, rng)


def query_variants(
    dataset: RetrievalDataset,
    kinds: Sequence[str] = DEFAULT_NOISE_KINDS,
    *,
    seed: int = 42,
    name: str | None = None,
) -> dict[str, RetrievalDataset]:
    """Build one dataset per query degradation, plus a ``clean`` baseline.

    Documents are shared between variants; only the query text changes, which
    isolates the effect of noise from indexing differences.

    Args:
        dataset: Source dataset.
        kinds: Degradations to generate.
        seed: Seed for deterministic transforms.
        name: Dataset name prefix.

    Returns:
        ``{"clean": dataset, "typo": dataset, ...}`` in the requested order.

    Raises:
        DatasetError: If a requested kind is unknown.
    """

    variants: dict[str, RetrievalDataset] = {"clean": dataset}
    prefix = name or dataset.name
    for kind in kinds:
        degraded = [
            RetrievalQuery(
                query_id=f"{query.query_id}::{kind}",
                query=degrade_query(query.query, kind, seed=seed),
                relevant_doc_ids=query.relevant_doc_ids,
                metadata={**query.metadata, "noise": kind},
            )
            for query in dataset.queries
        ]
        variant = RetrievalDataset(
            name=f"{prefix}::{kind}",
            documents=list(dataset.documents),
            queries=degraded,
            metadata={**dataset.metadata, "generator": "query_variants", "noise": kind},
        )
        variant.validate()
        variants[kind] = variant
    return variants


# ---------------------------------------------------------------------------
# Small built-in dataset for demos and tests
# ---------------------------------------------------------------------------


def demo_dataset() -> RetrievalDataset:
    """Return a small, self-contained dataset for quickstarts and tests.

    Returns:
        Six short documents about a fictional analytics platform with five
        queries and unambiguous relevance judgements.
    """

    docs = [
        (
            "billing-retries",
            "Billing retries: failed subscription charges are retried three times, "
            "on days one, three and seven. After the third failure the subscription "
            "is marked past due and the account owner is notified by email.",
        ),
        (
            "api-keys",
            "API keys: every key is scoped to a single project and expires after "
            "ninety days. Rotating a key keeps the old key valid for one hour so "
            "clients can roll over without downtime.",
        ),
        (
            "export-limits",
            "Export limits: a workspace can run two exports at once, and each "
            "export may contain at most five million rows. Larger requests must "
            "be split or scheduled as recurring jobs.",
        ),
        (
            "sso-setup",
            "SSO setup: administrators upload a SAML metadata file, map three "
            "attributes (email, display name, role) and enable enforcement. "
            "Enforcement forces every member to sign in through the identity "
            "provider on their next session.",
        ),
        (
            "retention-policy",
            "Retention policy: event data is kept for thirteen months, aggregate "
            "reports for three years, and audit logs for seven years. Deleting a "
            "workspace removes event data immediately but keeps audit logs.",
        ),
        (
            "webhook-signatures",
            "Webhook signatures: each delivery includes an HMAC-SHA256 header "
            "computed over the raw body with the workspace signing secret. "
            "Receivers should compare signatures in constant time and reject "
            "requests older than five minutes.",
        ),
    ]
    queries = [
        ("q1", "How many times are failed subscription charges retried?", ["billing-retries"]),
        ("q2", "When does an API key expire?", ["api-keys"]),
        ("q3", "What is the maximum number of rows in one export?", ["export-limits"]),
        ("q4", "How is SSO enforced for existing members?", ["sso-setup"]),
        ("q5", "How long are audit logs retained?", ["retention-policy"]),
        ("q6", "How should webhook receivers validate a signature?", ["webhook-signatures"]),
    ]
    return RetrievalDataset(
        name="demo",
        documents=[DatasetDocument(doc_id=doc_id, text=text) for doc_id, text in docs],
        queries=[
            RetrievalQuery(query_id=query_id, query=query, relevant_doc_ids=tuple(doc_ids))
            for query_id, query, doc_ids in queries
        ],
        metadata={"generator": "demo_dataset"},
    )


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_CONTEXT_SIZES",
    "DEFAULT_DEPTHS",
    "DEFAULT_NOISE_KINDS",
    "FILLER_PARAGRAPHS",
    "DatasetDocument",
    "Needle",
    "RetrievalDataset",
    "RetrievalQuery",
    "build_haystack",
    "chunk_size_sensitivity",
    "degrade_query",
    "demo_dataset",
    "load_corpus_dataset",
    "load_jsonl_dataset",
    "make_needle",
    "needle_in_haystack",
    "query_variants",
]
