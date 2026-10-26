# embedbench

**Benchmark text embedding models on your own data — retrieval quality, latency, memory and cost.**

`embedbench` embeds and indexes your corpus with several models, runs the same queries
through every one, and compares retrieval accuracy, latency percentiles, throughput,
memory footprint and dollar cost. It works with local models (Ollama,
sentence-transformers) and hosted APIs (OpenAI, Cohere, Voyage AI) behind one interface,
with no benchmarking service in between.

- **Bring your own corpus:** `.txt`, `.md`, `.pdf`, `.docx`, `.html`.
- **Bring your own queries or generate them:** JSONL datasets, or synthetic questions
  from a local Ollama LLM with quality filtering.
- **Compare fairly on one machine:** per-model indexes, exact cosine retrieval,
  cold/warm timings, cache-aware throughput, token counts, RSS and VRAM sampling.
- **Stress the failure modes:** needle-in-a-haystack at 2k/8k/32k tokens, chunk-size
  sweeps at 128–1024 tokens, and typo/paraphrase query robustness — all reproducible.
- **Get an answer:** Rich console tables, JSON/CSV/Markdown artifacts, PNG comparison
  plots, and a self-contained HTML report with weighted recommendations.
- **Use it as a library:** `from embedbench import Evaluator` runs the same engine
  without touching the filesystem.

Status: `0.2.0`, beta. Python 3.10+. MIT licensed.

---

## Table of contents

1. [Why this exists](#why-this-exists)
2. [Architecture](#architecture)
3. [Install](#install)
4. [Quickstart](#quickstart)
5. [Python API](#python-api)
6. [Stress workloads](#stress-workloads)
7. [CLI reference](#cli-reference)
8. [Configuration reference](#configuration-reference)
9. [Evaluation sets](#evaluation-sets)
10. [Metrics and report interpretation](#metrics-and-report-interpretation)
11. [Sample output](#sample-output)
12. [Adding a custom provider](#adding-a-custom-provider)
13. [Reproducibility and caching](#reproducibility-and-caching)
14. [Project layout](#project-layout)
15. [Tests](#tests)
16. [Known limitations](#known-limitations)
17. [Troubleshooting](#troubleshooting)
18. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
19. [License](#license)

---

## Why this exists

Generic embedding leaderboards (MTEB and friends) measure models on public benchmarks
that may look nothing like your documents. The model that wins on a leaderboard can lose
on your product manuals, support tickets or internal wiki. Retrieval quality also depends
on chunking, query style, top-k, hardware and cold-start behavior.

`embedbench` makes the local loop small and explicit:

1. Point it at a folder.
2. Use a hand-written eval set or generate one from your own chunks.
3. Run every model through the same chunks, the same queries and the same metrics.
4. Keep the config, raw results and environment snapshot for later comparison.

It is intentionally not a hosted platform, a distributed benchmark cluster, or an
LLM-agent framework. It is one reproducible command on a laptop or workstation.

---

## Architecture

```text
                    ┌───────────────────────────────┐
                    │ Typer CLI                     │
                    │ run · bench · stress · report │
                    │ generate-evalset · providers  │
                    └───────────────┬───────────────┘
                                    │
             ┌──────────────────────┼───────────────────────┐
             │                      │                       │
   ┌─────────▼─────────┐  ┌─────────▼──────────┐  ┌─────────▼─────────┐
   │ Python API         │  │ Pydantic RunConfig │  │ Dataset loaders    │
   │ Evaluator,         │  │ + snapshot         │  │ docs+evalset,      │
   │ StressRunner       │  │                    │  │ JSONL              │
   └─────────┬─────────┘  └─────────────────────┘  └─────────┬─────────┘
             │                                               │
             └──────────────────────┬────────────────────────┘
                                    │
                       ┌────────────▼─────────────┐
                       │ RetrievalDataset          │
                       │ documents + queries +     │
                       │ relevance (group aware)   │
                       └────────────┬─────────────┘
                                    │
           ┌────────────────────────┼─────────────────────────┐
           │                        │                         │
 ┌─────────▼─────────┐   ┌──────────▼─────────┐    ┌──────────▼─────────┐
 │ Models             │   │ Evaluator           │    │ Stress generators  │
 │ BaseEmbeddingModel │   │ embed → index →     │    │ needle-in-haystack │
 │ ollama · hf ·      │   │ query → measure     │    │ chunk-size sweep   │
 │ openai · cohere ·  │   │ latency · memory ·  │    │ query noise        │
 │ voyage (+plugins)  │   │ tokens · cost       │    │                    │
 └─────────┬─────────┘   └──────────┬──────────┘    └──────────┬─────────┘
           │                        │                          │
           │              ┌─────────▼──────────┐               │
           └──────────────► metrics.py          ◄──────────────┘
                          │ scalar + vectorised │
                          │ P/R/MRR/nDCG/Hit    │
                          │ latency/QPS/mem/cost│
                          └─────────┬──────────┘
                                    │
     ┌──────────────────────────────┼───────────────────────────────┐
     │                              │                               │
┌────▼─────────────┐   ┌────────────▼───────────┐   ┌───────────────▼────┐
│ Rich console      │   │ results.json           │   │ results.md          │
│ comparison table  │   │ summary.csv            │   │ latency-vs-quality  │
│ recommendation    │   │ queries.csv            │   │ quality-bars PNG    │
└───────────────────┘   └────────────────────────┘   └─────────────────────┘
```

Mermaid version:

```mermaid
flowchart LR
  CLI[Typer CLI] --> API[Evaluator / StressRunner]
  CLI --> CFG[RunConfig + snapshot]
  CFG --> ING[Ingest txt/md/pdf/docx/html]
  ING --> DS[RetrievalDataset]
  GEN[Needle / chunk / noise generators] --> DS
  DS --> EV[Evaluator]
  EV --> MODELS[BaseEmbeddingModel<br/>ollama · hf · openai · cohere · voyage]
  MODELS --> IDX[Cosine index: FAISS or NumPy]
  IDX --> Q[Answer queries, top-k]
  Q --> MET[metrics.py: P/R/MRR/nDCG/Hit<br/>latency · QPS · memory · cost]
  MET --> REPORTS[Rich / JSON / CSV / MD / PNG]
```

A provider is a small class: set `provider_name`, implement `load()` and
`_embed_prepared()`, register it with `register_model`. Batching, instruction prefixes,
validation, L2 normalisation, async fan-out and cost metadata live in
:class:`embedbench.models.BaseEmbeddingModel`, so a new provider is typically under 30
lines.

---

## Install

Requirements:

- Python **3.10+**
- Enough RAM for your chosen local model
- Optional: [Ollama](https://ollama.com/) for Ollama models and synthetic questions
- For PDF/DOCX parsing: `pip install "embedbench[docs]"` (not needed for txt/md/html)

```bash
pip install embedbench                 # Ollama + core metrics
pip install "embedbench[local]"        # + sentence-transformers / Hugging Face
pip install "embedbench[openai]"       # + OpenAI SDK and tiktoken
pip install "embedbench[cohere]"       # + Cohere SDK
pip install "embedbench[voyage]"       # + Voyage AI SDK
pip install "embedbench[faiss]"        # + FAISS (NumPy fallback otherwise)
pip install "embedbench[plots]"        # + matplotlib comparison plots
pip install "embedbench[docs]"         # + pypdf and python-docx
pip install "embedbench[all]"          # everything above
```

The core install is deliberately small: NumPy, Typer, Rich, Pydantic, PyYAML, Requests,
psutil and BeautifulSoup. Cloud SDKs, PyTorch, FAISS and matplotlib are opt-in extras, so
`pip install embedbench` never downloads a deep-learning stack you do not use.

From a checkout of this repository:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Check the CLI:

```bash
embedbench --help
embedbench providers
embedbench list-models
```

`providers` lists every registered provider, the extra it needs, its API-key variable and
the hosted-model price table. `list-models` shows a curated catalogue and which Ollama
models are installed.

### Ollama

Install Ollama from [ollama.com](https://ollama.com/), then:

```bash
ollama serve
ollama pull nomic-embed-text
# optional, for synthetic evaluation questions:
ollama pull llama3.2
```

`embedbench` talks to `http://localhost:11434` by default. Model names are the same
names you use with `ollama run`, for example `nomic-embed-text`,
`mxbai-embed-large`, or `bge-m3`.

### Hugging Face

```bash
pip install "embedbench[local]"

embedbench run \
  --config examples/config.yaml \
  --models hf:BAAI/bge-small-en-v1.5,hf:all-MiniLM-L6-v2
```

The first run downloads model weights to the Hugging Face cache. Gated models require
accepting the model license and `huggingface-cli login`. No key is bundled with this tool.

### Hosted providers

Set the provider's environment variable, then use the `provider:model` spec:

```bash
export OPENAI_API_KEY=...
export COHERE_API_KEY=...
export VOYAGE_API_KEY=...

embedbench bench --demo --models openai:text-embedding-3-small
embedbench bench --demo --models cohere:embed-english-v3.0,voyage:voyage-3-large
```

Keys can also be supplied with `backends.<provider>.api_key` in YAML or a `--option`
flag; environment variables are preferred so secrets stay out of config files. Set
`backends.openai.base_url` (or the `base_url` model option) to target any
OpenAI-compatible endpoint such as vLLM, LM Studio or an Azure gateway.

---

## Quickstart

From the repository root:

```bash
# 1. Compare two local Hugging Face models on the sample corpus.
embedbench run --config examples/config.yaml

# 2. Open the generated report.
#    macOS: open examples/results/report.html
#    Windows: start examples/results/report.html
#    Linux: xdg-open examples/results/report.html
```

The example config uses:

- `examples/docs/` — six sample documents.
- `examples/evalset.jsonl` — 12 hand-written queries with document-level relevance.
- `hf:BAAI/bge-small-en-v1.5` and `hf:all-MiniLM-L6-v2`.
- `top_k: 5`, recursive 600-character chunks with 100 characters of overlap.
- Outputs in `examples/results/`.

Run the same corpus with an Ollama model:

```bash
ollama serve
ollama pull nomic-embed-text

embedbench run \
  --config examples/config.yaml \
  --models hf:BAAI/bge-small-en-v1.5,ollama:nomic-embed-text \
  --top-k 5 \
  --no-progress
```

`--no-progress` is optional; it only hides embedbench's progress bars, not the
download progress printed by Hugging Face itself.

### Generate an eval set from your documents

```bash
ollama serve
ollama pull llama3.2

embedbench generate-evalset \
  --docs examples/docs \
  --output examples/generated-evalset.jsonl \
  --model llama3.2 \
  --max-chunks 12 \
  --questions-per-chunk 1

embedbench run \
  --docs examples/docs \
  --evalset examples/generated-evalset.jsonl \
  --models ollama:nomic-embed-text \
  --top-k 3
```

To generate and benchmark in one command, use a config without `evalset.path` and set
`evalset.auto_generate: true`, or pass `--auto-evalset` with no eval-set path:

```bash
embedbench run \
  --docs examples/docs \
  --models ollama:nomic-embed-text \
  --auto-evalset \
  --evalset-model llama3.2 \
  --top-k 3 \
  --output-dir results/auto
```

An explicit `evalset.path` wins over auto-generation.

### Re-render with another priority

```bash
embedbench report \
  --results examples/results/results.json \
  --priority accuracy \
  --output examples/results/report-accuracy.html
```

`--results` also accepts the output directory. The original JSON and CSV are untouched;
only a new HTML report is written.

### End-to-end command sequence from a fresh checkout

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
embedbench providers
embedbench list-models
embedbench run --config examples/config.yaml --no-progress
embedbench bench --docs examples/docs --evalset examples/evalset.jsonl \
  --models ollama:nomic-embed-text --k 1,5,10 --no-plots
embedbench report --results examples/results --priority speed
pytest
```

---

## Python API

Everything the CLI does is available as a library. The smallest useful program:

```python
from embedbench import Evaluator, demo_dataset, write_reports

dataset = demo_dataset()  # or load_corpus_dataset("./docs", "evalset.jsonl")

result = Evaluator(
    ["ollama:nomic-embed-text", "openai:text-embedding-3-small"],
    k_values=(1, 5, 10),
).evaluate(dataset)

print(result.best().spec)                 # highest nDCG@10
print(result.models[0].metrics_at(10))    # hit_rate/mrr/ndcg/precision/recall
print(result.models[0].cost)              # tokens, list price, estimated USD

write_reports(result, "results/", plots=True)
```

`Evaluator.evaluate()` returns a `BenchmarkResult` with one `ModelResult` per model:

| Field | Contents |
| --- | --- |
| `metrics_by_k` | mean hit rate, MRR, nDCG, precision and recall per cut-off |
| `latency`, `latency_warm` | count, mean, min, max, p50/p90/p95/p99, QPS |
| `throughput` | documents per second and queries per second |
| `tokens`, `cost` | document/query tokens, list price, estimated USD |
| `memory` | vector bytes, index size, peak RSS and VRAM |
| `timings` | load, token counting, embedding, index build, queries |
| `per_query` | ranked documents, scores, latency and metrics per query |

Useful helpers:

```python
from embedbench import (
    DatasetDocument, RetrievalDataset, RetrievalQuery,   # build a dataset by hand
    load_jsonl_dataset, load_corpus_dataset,             # load one from disk
    needle_in_haystack, chunk_size_sensitivity,          # synthetic workloads
    query_variants, StressRunner,                        # robustness sweeps
    render_console, render_markdown,                      # reporting
    create_model, register_model, BaseEmbeddingModel,     # providers
    ndcg_at_k, evaluate_rankings, latency_stats, estimate_cost,
)
```

Build a dataset in code:

```python
dataset = RetrievalDataset(
    name="support",
    documents=[
        DatasetDocument("billing", "Failed charges are retried on days 1, 3 and 7."),
        DatasetDocument("keys", "API keys expire after ninety days."),
    ],
    queries=[
        RetrievalQuery("q1", "How often are failed charges retried?", ("billing",)),
        RetrievalQuery("q2", "When do API keys expire?", ("keys",)),
    ],
)
dataset.validate()

result = Evaluator(["ollama:nomic-embed-text"], k_values=(1, 2)).evaluate(dataset)
```

Evaluation is one model at a time for comparable latency, and every model sees the same
documents, queries and cut-offs. A failing provider never aborts the run: it is recorded
with `status="error"` and the other models still produce a report.

## Stress workloads

Three workloads answer the questions a generic leaderboard cannot:

| Workload | Question | Generator |
| --- | --- | --- |
| Needle in a haystack | Can one fact survive a 2k, 8k or 32k token context, at the start, middle or end? | `needle_in_haystack()` |
| Chunk-size sensitivity | How much quality does 128 vs 256 vs 512 vs 1024 token chunking cost? | `chunk_size_sensitivity()` |
| Query robustness | How far does nDCG fall under typos, truncation, shuffling or paraphrasing? | `query_variants()` |

They run from the CLI:

```bash
# Synthetic haystacks: 2k/8k/32k tokens, needles at 0%, 50% and 100% depth.
embedbench stress needle \
  --models ollama:nomic-embed-text \
  --context-sizes 2048,8192,32768 \
  --depths 0,0.5,1

# Re-chunk the same corpus and compare quality.
embedbench stress chunk-size \
  --docs ./docs --evalset evalset.jsonl \
  --models ollama:nomic-embed-text \
  --chunk-sizes 128,256,512,1024

# Degrade the queries, keep the index fixed.
embedbench stress noisy \
  --docs ./docs --evalset evalset.jsonl \
  --models ollama:nomic-embed-text \
  --noise typo,case,truncate,shuffle,paraphrase
```

Each command writes `<workload>.json` and `<workload>.md` (default `results/stress/`),
including a depth breakdown for needles and a delta-versus-clean table for noise.
From Python, use `StressRunner`:

```python
from embedbench import StressRunner, demo_dataset

runner = StressRunner(["ollama:nomic-embed-text"], k_values=(1, 5))
needle = runner.needle_in_haystack(context_sizes=(2048, 8192), depths=(0.0, 0.5, 1.0))
noise = runner.query_robustness(demo_dataset(), kinds=("typo", "paraphrase"))
print(needle.markdown())
```

All generators are deterministic for a given `--seed`, so a regression is a real
regression rather than a lucky shuffle.

---

## CLI reference

### `embedbench run`

```text
embedbench run [OPTIONS]

  --docs, -d PATH           Documents folder (.txt/.md/.pdf/.docx/.html)
  --models, -m TEXT         Comma-separated backend:model specs
  --top-k INT               Chunks retrieved per query (default: config, 5)
  --config, -c PATH         YAML config (default: ./embedbench.yaml if present)
  --evalset PATH            JSONL eval set; wins over synthetic generation
  --auto-evalset/--no-auto-evalset
  --evalset-model TEXT      Ollama LLM used for synthetic questions
  --output-dir, -o PATH     Directory for results.json/results.csv/HTML/log
  --seed INT                Global seed
  --priority TEXT           accuracy | speed | balanced
  --chunking TEXT           fixed | sentence | recursive
  --chunk-size INT          Target chunk size in characters
  --chunk-overlap INT       Overlap in characters
  --batch-size INT          Batch size for built-in backends
  --cost-per-1m FLOAT       Hosted price per 1M tokens for estimate
  --cache/--no-cache        Enable/disable the embedding cache
  --cache-dir PATH          Cache directory
  --ollama-host TEXT        Ollama base URL
  --progress/--no-progress  Show embedbench progress bars
  --verbose, -v             Debug logging
```

CLI flags override YAML values. Relative paths resolve from the current working
directory, whether they come from YAML or CLI. The output directory is reused between
runs; files with the same names are overwritten, while `run.log` is recreated.

Exit codes:

- `0` — at least one model completed successfully.
- `1` — setup error, no successful model, or a failed report command.
- `130` — interrupted.

If one model fails (for example, a missing Ollama model), the other models still run and
artifacts are still written. The failed model appears in the report with its error.

### `embedbench generate-evalset`

```text
embedbench generate-evalset [OPTIONS]

  --docs, -d PATH           Documents folder
  --output, -o PATH         Destination JSONL (default: evalset.jsonl)
  --config, -c PATH         YAML config
  --model TEXT              Ollama LLM (default: llama3.2)
  --questions-per-chunk INT 1..5
  --max-chunks INT          Deterministic sample of eligible chunks
  --min-question-words INT
  --dedupe-threshold FLOAT  0.5..1.0 sequence similarity
  --chunking / --chunk-size / --chunk-overlap
  --seed INT
  --ollama-host TEXT
  --progress/--no-progress
  --verbose, -v
```

This command only needs Ollama plus the document parsers; it does not load embedding
models. It writes JSONL with `query_id`, `query`, `relevant_doc_ids`, optional `answer`,
`source_chunk_id` and generator metadata. All questions reference the document ID of
their source chunk.

### `embedbench report`

```text
embedbench report [OPTIONS]

  --results, -r PATH        results.json or the run output directory
  --output, -o PATH         HTML path (or a directory)
  --priority TEXT           accuracy | speed | balanced
  --title TEXT              Override report title
  --verbose, -v
```

The console summary is printed, and a fresh `report.html` is rendered from the saved
JSON. This command does not re-run embeddings or queries.

### `embedbench list-models`

```text
embedbench list-models [OPTIONS]

  --ollama-host TEXT        Ollama base URL
  --json                    Emit JSON
```

Lists curated Hugging Face and Ollama models, dimensions, notes, whether Ollama is
reachable, and which Ollama models are installed. It does not download or ping Hugging
Face. `--json` is suitable for scripting.

### `embedbench bench`

```text
embedbench bench [OPTIONS]

  --models, -m TEXT         Comma-separated provider:model specs (required)
  --docs, -d PATH           Documents folder to chunk and index
  --evalset PATH            JSONL eval set with queries and relevant_doc_ids
  --dataset PATH            JSONL dataset (documents + queries) to evaluate directly
  --demo                    Use the built-in demo dataset
  --k TEXT                  Comma-separated cut-offs (default: 1,5,10)
  --output-dir, -o PATH     results.json/.md/.csv and plots (default: results)
  --priority TEXT           accuracy | speed | balanced | cost
  --index TEXT              auto | faiss | numpy
  --seed INT                Global seed
  --batch-size INT          Override the embedding batch size
  --save-dataset PATH       Write the resolved dataset as JSONL
  --plugins TEXT            Modules or .py files registering custom providers
  --plots/--no-plots        Render PNG comparisons when matplotlib is installed
  --config, -c PATH         YAML config for ingestion/chunking defaults
  --progress/--no-progress
  --verbose, -v
```

Dataset-based evaluation. Choose exactly one source: `--demo`, `--dataset FILE`, or
`--docs` plus `--evalset`. Artifacts in `--output-dir`:

| File | Contents |
| --- | --- |
| `results.json` | full schema-2.0 report with per-query records and recommendation |
| `results.md` | Markdown summary: quality, efficiency, recommendation, worst queries |
| `summary.csv` | one row per model with quality, latency, memory and cost |
| `queries.csv` | one row per query and model |
| `latency-vs-quality.png` | p95 latency against nDCG (needs `[plots]`) |
| `quality-bars.png` | hit rate / MRR / nDCG bars (needs `[plots]`) |
| `cost-vs-quality.png` | estimated USD against nDCG (needs `[plots]`, ≥2 priced models) |

Exit code `1` when every model failed; partial failures still write artifacts and exit
`0` so CI can compare what succeeded.

### `embedbench stress needle|chunk-size|noisy`

```text
embedbench stress needle [OPTIONS]
  --models, -m TEXT         Comma-separated provider:model specs (required)
  --context-sizes TEXT      Haystack sizes in tokens (default: 2048,8192,32768)
  --depths TEXT             Insertion depths 0..1 (default: 0,0.5,1)
  --chunk-size INT          Chunk size in characters (default: 800)
  --k TEXT                  Cut-offs (default: 1,5,10)
  --output-dir, -o PATH     Writes needle.json and needle.md
  --seed INT                Seed for filler and needle codes
  --index TEXT              auto | faiss | numpy
  --plugins TEXT            Custom provider modules
  --verbose, -v
```

`stress chunk-size` takes `--docs`/`--evalset` plus `--chunk-sizes 128,256,512,1024`;
`stress noisy` takes `--docs`/`--evalset` plus `--noise typo,case,truncate,shuffle,paraphrase`.
Both write `<workload>.json` and `<workload>.md` under `results/stress/`.

### `embedbench providers`

```text
embedbench providers [--json]
```

Lists registered providers with the install extra, API-key environment variable and
price, followed by the hosted-model list-price table. Prices are indicative; override
them per model with the `price_per_1m_tokens` option or `backends.extra`. `--json` emits
the same data for scripts.

---

## Configuration reference

Use `--config config.yaml`. If no path is supplied, `./embedbench.yaml` or
`./embedbench.yml` is loaded when present. Unknown keys are rejected so typos are loud.

A complete example is in [`examples/config.yaml`](examples/config.yaml). The defaults
below match the dataclasses/pydantic models in `embedbench/config.py`.

```yaml
docs: ./docs
models:
  - hf:BAAI/bge-small-en-v1.5
  - ollama:nomic-embed-text
top_k: 5
seed: 42
output_dir: ./results
plugins: []

chunking:
  strategy: recursive       # fixed | sentence | recursive
  chunk_size: 800           # characters, >= 100
  chunk_overlap: 120        # characters, < chunk_size
  min_chunk_chars: 80

ingestion:
  extensions: [".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"]
  recursive: true
  max_file_bytes: 20000000
  skip_dirs: [".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__", ".embedbench_cache"]

evalset:
  path: null                # JSONL; wins over auto_generate
  auto_generate: false
  ollama_model: llama3.2
  questions_per_chunk: 1    # 1..5
  max_chunks: null          # deterministic sample when set
  min_chunk_chars: 120
  min_question_words: 5
  max_question_words: 40
  dedupe_threshold: 0.85
  max_generation_attempts: 2
  temperature: 0.0
  num_predict: 220
  request_timeout_s: 180

backends:
  ollama:
    host: http://localhost:11434
    batch_size: 32
    timeout_s: 120
    max_retries: 3
    retry_backoff_s: 0.75
    query_prefix: null
    document_prefix: null
    keep_alive: null
  hf:
    device: auto            # auto | cpu | cuda | cuda:0 | mps
    batch_size: 32
    trust_remote_code: false
    cache_folder: null
    local_files_only: false
    max_seq_length: null    # null keeps the model default
    query_prefix: null
    document_prefix: null
  extra: {}                 # options for custom backends

cache:
  enabled: true
  dir: ./.embedbench_cache

report:
  title: "embedbench report"
  priority: balanced        # accuracy | speed | balanced
  cost_per_1m_tokens: 0.02
```

### Top-level

| Key | Default | Meaning |
| --- | --- | --- |
| `docs` | `null` | Folder to ingest. Required for `run` and `generate-evalset` unless passed with `--docs`. |
| `models` | `[]` | List of `backend:model` specs. CLI `--models` accepts a comma-separated string. |
| `top_k` | `5` | Number of chunks retrieved per query. Metrics are reported at this k. |
| `seed` | `42` | Seeds Python/NumPy, synthetic sampling and LLM generation. |
| `output_dir` | `results` | Artifact directory. Reused across runs. |
| `plugins` | `[]` | Dotted module names or `.py` paths that register extra backends. |

### Chunking

| Key | Default | Notes |
| --- | --- | --- |
| `strategy` | `recursive` | `fixed` = character windows; `sentence` = pack sentences; `recursive` = split on paragraphs/newlines/sentences/words then pack. |
| `chunk_size` | `800` | Target maximum characters. Minimum 100. |
| `chunk_overlap` | `120` | Character overlap target. Must be `< chunk_size`. Sentence/recursive overlap snaps to boundaries and may be smaller than requested. |
| `min_chunk_chars` | `80` | Drop shorter chunks, except that a short document always retains one chunk. |

### Ingestion

| Key | Default | Notes |
| --- | --- | --- |
| `extensions` | txt/md/markdown/pdf/docx/html/htm | Case-insensitive; leading dots optional. |
| `recursive` | `true` | Walk subfolders. Hidden files/directories are ignored. |
| `max_file_bytes` | `20000000` | Larger files are skipped with a reason. |
| `skip_dirs` | see example | Directory names skipped anywhere in the subtree. |

HTML parsing removes `script`, `style`, `nav`, `footer`, `noscript`, `svg` and
`template` elements. PDFs use text extraction only; scanned/image PDFs are skipped.
DOCX paragraphs and tables are flattened in document order.

### Eval set and synthetic generation

| Key | Default | Notes |
| --- | --- | --- |
| `path` | `null` | User JSONL. If set, it is used even if `auto_generate` is true. |
| `auto_generate` | `false` | Generate questions with Ollama when no path is set. |
| `ollama_model` | `llama3.2` | Chat/instruct model used to write questions. |
| `questions_per_chunk` | `1` | Ask for up to this many questions per chunk (1–5). Filtering may accept fewer. |
| `max_chunks` | `null` | If set, sample this many eligible chunks deterministically using `seed`. |
| `min_chunk_chars` | `120` | Chunks shorter than this are not used for generation. |
| `min_question_words` | `5` | Lower filter bound. |
| `max_question_words` | `40` | Upper filter bound. |
| `dedupe_threshold` | `0.85` | SequenceMatcher similarity at/above this drops a near duplicate. |
| `max_generation_attempts` | `2` | Retries for invalid JSON per chunk. |
| `temperature` | `0.0` | Passed to Ollama. |
| `num_predict` | `220` | Ollama max generated tokens. |
| `request_timeout_s` | `180` | HTTP timeout for question generation. |

The generator filters questions that do not end with `?`, are too short/long, mention
“this passage/the document”, are generic (“summarize…”, “what is this about”), have no
content words, or are near-duplicates of an accepted question. These are heuristics, not
an LLM judge; review generated sets before treating them as ground truth.

### Backends

`backends.ollama`:

| Key | Default | Notes |
| --- | --- | --- |
| `host` | `http://localhost:11434` | Trailing slash removed. |
| `batch_size` | `32` | Number of chunks sent per `/api/embed` request in benchmarks. |
| `timeout_s` | `120` | Per-request timeout. |
| `max_retries` | `3` | Retries for connection errors, 429 and 5xx. |
| `retry_backoff_s` | `0.75` | Exponential backoff base; capped at 20s per wait. |
| `query_prefix` / `document_prefix` | `null` | Optional text prepended before embedding. |
| `keep_alive` | `null` | Optional Ollama `keep_alive` value, e.g. `10m`. |

`backends.hf`:

| Key | Default | Notes |
| --- | --- | --- |
| `batch_size` | `32` | Passed to sentence-transformers `encode`. |
| `device` | `auto` | CUDA, then Apple MPS, then CPU. |
| `trust_remote_code` | `false` | Only enable for models you trust. |
| `cache_folder` | `null` | Hugging Face cache override. |
| `local_files_only` | `false` | Fail instead of downloading. |
| `max_seq_length` | `null` | Tokenizer/model truncation length. |
| `query_prefix` / `document_prefix` | `null` | Overrides; E5 model names automatically get `query: ` / `passage: ` unless set. |

`backends.openai`, `backends.cohere` and `backends.voyage` share one schema:

| Key | Default | Notes |
| --- | --- | --- |
| `api_key` | `null` | Prefer `OPENAI_API_KEY` / `COHERE_API_KEY` / `VOYAGE_API_KEY`. |
| `base_url` | `null` | Point at vLLM, LM Studio, Azure or a proxy (OpenAI-compatible). |
| `batch_size` | `32` | Inputs per SDK call (further split by provider limits). |
| `dimensions` | `null` | Output-dimension reduction where the provider supports it. |
| `timeout_s` | `120` | Per-request timeout. |
| `max_retries` | `3` | Retries on 408/409/425/429/5xx, with exponential backoff. |
| `retry_backoff_s` | `0.75` | Base backoff delay. |
| `max_concurrency` | `4` | Concurrent batches in the async path. |
| `query_prefix` / `document_prefix` | `null` | Instruction prefixes for your model. |

`backends.extra` holds free-form options for third-party providers. The base class reads
`batch_size`, `query_prefix`, `document_prefix`, `normalize`, `price_per_1m_tokens` and
`max_concurrency`; the provider can read anything else.

### Cache and report

| Key | Default | Notes |
| --- | --- | --- |
| `cache.enabled` | `true` | Persistent embedding cache. |
| `cache.dir` | `.embedbench_cache` | SQLite database directory. |
| `report.title` | `embedbench report` | HTML title/heading. |
| `report.priority` | `balanced` | Which recommendation is selected by default. |
| `report.cost_per_1m_tokens` | `0.02` | Hosted API price for the cost-equivalent estimate. |

---

## Evaluation sets

JSONL format, one object per line:

```jsonl
{"query_id": "q001", "query": "What is the maximum payload size?", "relevant_doc_ids": ["api.md"], "answer": "5 MB"}
{"query_id": "q002", "query": "How long are records retained?", "relevant_doc_ids": ["security.html"], "category": "retention"}
```

Rules:

- `query` is required and must be non-empty.
- `relevant_doc_ids` is a non-empty list of document IDs.
- Document IDs are paths relative to the `docs` folder, using `/` separators, for
  example `guides/install.md`. Leading `./` and backslashes are normalized.
- `query_id` is optional; generated IDs are `q0001`, `q0002`, … and never collide.
- `answer` is optional. It is stored in the report but not used by retrieval metrics.
- `source_chunk_id` is optional.
- Unknown top-level fields are preserved under `metadata`.
- Every referenced document must exist **and must produce at least one chunk**. This is
  checked before expensive model loading.

Blank lines are ignored. A duplicate query ID is an error; duplicate query text is a
warning because it intentionally double-weights that query.

Auto-generation timing: every eligible chunk is one Ollama generation call. Use
`evalset.max_chunks` (and `--max-chunks` in `generate-evalset`) for a representative
sample. Sampling is seeded and deterministic.

---

## Metrics and report interpretation

All metrics are computed per query, then averaged. Relevance is **binary and
document-level**. Retrieval returns chunks; each chunk maps to its `doc_id`. Duplicate
chunks from the same document are collapsed at their first rank before computing
metrics. `k` is `top_k` from the run.

### Formulas

Let `Rel` be the set of relevant document IDs, `Docs_k` the unique documents in the
top-k ranked list, and `rank_i` the rank of the first relevant document.

```text
Recall@k    = |Rel ∩ Docs_k| / |Rel|
Precision@k = |Rel ∩ Docs_k| / k
MRR@k       = 1 / rank_first_relevant          (0 if none in top-k)
HitRate@k   = 1 if |Rel ∩ Docs_k| > 0 else 0
DCG@k       = Σ_{i=1..k} rel_i / log2(i + 1)
IDCG@k      = Σ_{i=1..min(|Rel|, k)} 1 / log2(i + 1)
nDCG@k      = DCG@k / IDCG@k                   (0 if IDCG is 0)
```

Precision uses `k` as the denominator even when fewer than `k` unique documents are
returned. This is the standard IR definition and makes low-precision runs visible. With
one relevant document and `top_k=5`, the best possible precision is `0.2`.

A relevant document with many chunks can be retrieved several times but counts once; the
extra chunks consume top-k slots, which lowers precision. Chunk-level evaluation is not
implemented by design; document-level relevance matches the usual user question “did we
find the right source?”.

### Latency

- **Query p50/p95/p99** cover all queries and include query embedding plus vector search.
- **Cold query** is the first query. **Warm** percentiles exclude it.
- **Model load ms** is the time spent in `backend.load()`; for HF this is model loading,
  for Ollama it verifies the model exists (actual model warm-up appears in the first
  embedding).
- **Cold embed** is the first uncached embedding batch. **Warm batch** percentiles
  exclude it. With a full cache hit, there is no cold embed measurement.
- **Indexing seconds** includes cache lookups, vector assembly and index construction.
  **Embedding seconds** is only actual embedding calls. `chunks/sec` is corpus size
  divided by indexing seconds, so a warm cache inflates it; check `cache.hit_rate`.
- **Queries/sec** is `n_queries / sum(query_latency)`. It is a reciprocal mean latency,
  not a load test with concurrent users.

### Tokens, cost and resources

- **HF token counts** use the model's own tokenizer, including any query/document prefix,
  truncated to the model's `max_seq_length`.
- **Hosted token counts** use `tiktoken` when the OpenAI extra is installed (exact for
  OpenAI models, approximate for others), otherwise the same heuristic as Ollama.
- **Ollama token counts** are estimates (words + punctuation) because the Ollama REST API
  does not expose the model tokenizer. Reports mark these with `*` and
  `tokens.estimated: true`.
- **Cost** is `total_tokens / 1,000,000 × price_per_1m_tokens`. For `run` the price comes
  from `report.cost_per_1m_tokens`; for `bench` it is the provider's list price (or your
  `price_per_1m_tokens` override). It is a comparison aid, not a bill.
- **RAM peak** is process RSS sampled every 50 ms (PSUtil). It includes the model for HF.
  For Ollama the model lives in the server process, so `model_resident_mb` and VRAM come
  from `/api/ps` and are resident sizes, not sampled peaks.
- **VRAM** for HF is PyTorch CUDA peak allocated memory. It is `n/a` on CPU and Apple MPS.
- **Index size** is the serialized size of the in-memory index at report time. Indexes are
  not persisted between runs.

### Recommendation

Three answers are computed, regardless of the selected priority:

- **Accuracy-first:** highest nDCG@k; ties break on recall, then MRR, then p95.
- **Speed-first:** lowest p95 query latency; ties break on nDCG.
- **Balanced:** weighted score after min–max normalization within the run:
  `0.5 × accuracy + 0.3 × speed + 0.2 × cost`, where speed and cost scores are inverted
  (fastest/cheapest = 1). Ties break on nDCG, then name.

The selected priority comes from `report.priority`, `--priority`, or the report command.
Scores are relative to the models and corpus in that run; they are not absolute quality
scores and cannot be compared across runs without care.

### `bench` versus `run`

Both commands share the same metric definitions and the same exact-cosine retrieval, but
they answer different questions:

| | `embedbench run` | `embedbench bench` |
| --- | --- | --- |
| Input | docs folder + eval set, chunked by config | dataset: docs+eval set, JSONL, or `--demo` |
| Cut-offs | one `top_k` | several (`--k 1,5,10`) |
| Cache | SQLite embedding cache | none (measures the provider) |
| Cost | one `cost_per_1m_tokens` | per-model list price from `providers` |
| Artifacts | JSON/CSV/HTML + config snapshot | JSON/Markdown/CSV + PNG plots |
| Priorities | accuracy / speed / balanced | accuracy / speed / balanced / cost |

The evaluator's recommendation uses min-max normalisation across the successful models
with these weights: accuracy `0.80/0.15/0.05`, speed `0.20/0.75/0.05`, balanced
`0.55/0.30/0.15`, cost `0.30/0.10/0.60`. Unknown prices score neutrally (`0.5`); free
local models score `1.0` on cost. Every component is reported next to the score so you
can see why a model won.

### Report files

| File | Contents |
| --- | --- |
| `results.json` | Schema version, run metadata, config, corpus, eval set, per-model metrics/timings/resources/tokens/per-query results, recommendation, artifact paths. |
| `results.csv` | One flat summary row per model, including cold/warm timing, accuracy, cost, cache, RAM and VRAM. |
| `queries.csv` | One row per model/query with metrics, latency, retrieved documents and relevant documents. |
| `report.html` | Self-contained HTML/CSS/JS with six stat cards, recommendation cards, four SVG charts, timing/resource tables, per-query details, environment and config. No network requests. |
| `config.snapshot.yaml` | Exact validated config used for the run, written before benchmarking. |
| `run.log` | Human-readable run log. |
| `evalset.generated.jsonl` | Synthetic eval set, only when auto-generation was used and no path was given. |

The HTML report includes an accuracy-vs-latency scatter plot, grouped retrieval-metric
bars, latency-percentile bars, and a cost-equivalent bar chart. Charts are generated from
the embedded JSON with vanilla JS and inline SVG; opening the file offline works.

---

## Sample output

The following is from the sample corpus (`6` documents, `19` chunks, `12` queries) on a
16-core Windows CPU, CPU-only PyTorch, `--no-cache`, and two Hugging Face models. Timings
are illustrative and depend heavily on hardware, thermal state, model cache and whether
the embedding cache is warm.

### Accuracy

| Model | nDCG@5 | Recall@5 | Precision@5 | MRR@5 | Hit@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hf:BAAI/bge-small-en-v1.5` | 0.969 | 1.000 | 0.200 | 0.958 | 1.000 |
| `hf:all-MiniLM-L6-v2` | 0.969 | 1.000 | 0.200 | 0.958 | 1.000 |

Both models found the correct document for every query. Precision is `0.2` because there
is one relevant document and `k=5`; that is expected, not a bug. A harder corpus and
more queries are needed to separate these models on accuracy.

### Latency, indexing and footprint

| Model | p50 ms | p95 ms | p99 ms | Index s | Chunks/s | Queries/s | Dim | Index MB | Tokens | Cost est. | RAM MB | VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hf:BAAI/bge-small-en-v1.5` | 102.4 | 114.9 | 118.3 | 4.72 | 4.0 | 9.7 | 384 | 0.028 | 2,463 | $0.000049 | 684.1 | n/a |
| `hf:all-MiniLM-L6-v2` | 53.6 | 58.1 | 58.1 | 2.24 | 8.5 | 18.5 | 384 | 0.028 | 2,463 | $0.000049 | 653.8 | n/a |

### Recommendation

```text
Selected: hf:all-MiniLM-L6-v2 (Balanced)
Best weighted score (1.000) using 50% accuracy, 30% speed and 20% cost-equivalent:
nDCG@5=0.969, p95=58.05 ms, estimated cost $0.000049.
Alternatives: Accuracy-first: hf:all-MiniLM-L6-v2 | Speed-first: hf:all-MiniLM-L6-v2

Model                       Acc.  Speed   Cost  Balanced
hf:BAAI/bge-small-en-v1.5  1.000  0.000  1.000     0.700
hf:all-MiniLM-L6-v2        1.000  1.000  1.000     1.000
```

With tied accuracy and cost, the faster model wins the balanced and speed priorities.
On a real corpus you can expect different accuracy trade-offs.

---

## Adding a custom provider

A provider only needs `load()` and `_embed_prepared()`. The base class handles instruction
prefixes, text validation, row/dimension checks, finiteness checks, L2 normalisation,
batching and the async fan-out.

Create `my_provider.py`:

```python
from __future__ import annotations

import numpy as np

from embedbench.models import BaseEmbeddingModel, register_model


@register_model("constant")
class ConstantModel(BaseEmbeddingModel):
    """A deliberately simple provider; replace the vector with your API call."""

    provider_name = "constant"
    default_price_usd_per_1m = None   # set 0.0 for free local models

    def load(self) -> None:
        self.embedding_dimension = 8

    def _embed_prepared(self, texts, kind="document"):
        return np.ones((len(texts), self.embedding_dimension), dtype=np.float32)
```

Use it with the file-based pipeline:

```yaml
# constant.yaml
docs: ./examples/docs
models: [constant:demo]
plugins: [./my_provider.py]     # or an importable dotted module name

backends:
  extra:
    constant:
      batch_size: 8
```

```bash
embedbench run --config constant.yaml --evalset examples/evalset.jsonl --top-k 3
```

Or with the evaluator, loading the file directly:

```bash
embedbench bench --demo --models constant:demo --plugins ./my_provider.py
```

Notes:

- `plugins` entries are either dotted modules or paths to standalone `.py` files
  (resolved from the current working directory). For multi-file plugins, install the
  package or put it on `PYTHONPATH` and use a dotted module name.
- Providers can override `count_tokens()`, `runtime_metadata()`, `close()`,
  `reset_peak_memory()` and `cache_namespace`. The default token counter is a documented
  heuristic; cloud providers use `tiktoken` when they can.
- If your provider needs prefixes, pass `query_prefix`/`document_prefix` in
  `backends.extra`, or override `prepare_queries`/`prepare_documents`.
- `kind` is `"document"` or `"query"`; Cohere and Voyage use it to pick their input type.
- For distribution, expose an entry point:
  ```toml
  [project.entry-points."embedbench.models"]
  constant = "my_package.provider:ConstantModel"
  ```
  The entry point name is the registry name used in `provider:model` specs. The legacy
  `embedbench.backends` group is still loaded for compatibility.

The interface is intentionally small: load, embed, optional token counting and metadata.
There is no service framework to learn.

---

## Reproducibility and caching

Every run writes `config.snapshot.yaml`, environment versions, platform, CPU/RAM, seed,
corpus stats and the full JSONL examples used. Dataset-based runs embed the dataset name,
its statistics and generator settings in `results.json`. Seeds are applied to Python and
NumPy before each model, and every generator (`needle_in_haystack`,
`chunk_size_sensitivity`, `query_variants`) is deterministic for a given seed. LLM calls
use `temperature` and per-attempt seeds.

Embedding results are cached in SQLite under
`.embedbench_cache/embeddings.sqlite3` by default:

```text
cache key = (model namespace, SHA-256(raw chunk text))
namespace = provider + model + prefixes + normalisation + provider-specific identity
```

The cache stores float32 vectors and text hashes, not the raw chunk text. It never caches
query embeddings, and it is keyed by model, not by corpus path, so moving a corpus keeps
the cache useful. Changing prefixes, HF `max_seq_length`, or the Ollama host produces a
new namespace. `embedbench bench` and the stress commands do not use the SQLite cache,
so their latency numbers measure the provider rather than local disk.

To force a cold run:

```bash
embedbench run --config examples/config.yaml --no-cache
```

To clear the cache, delete the cache directory. A model revision fetched from Hugging
Face is not part of the cache key; if a model is updated in place, clear the cache or
pin your own local model directory. Timings are also affected by page cache, CPU
frequency, thermal throttling, other processes and whether the model was just loaded.

---

## Project layout

```text
embedbench/
├── src/embedbench/
│   ├── __init__.py             # public API: Evaluator, datasets, metrics, reporters
│   ├── __main__.py
│   ├── cli.py                  # Typer commands, bench/stress/providers
│   ├── evaluator.py            # Evaluator, BenchmarkResult, recommendations
│   ├── datasets.py             # RetrievalDataset + needle/chunk/noise generators
│   ├── stress.py               # StressRunner and StressReport
│   ├── metrics.py              # scalar + vectorised retrieval, latency, cost, memory
│   ├── reporter.py             # Rich, Markdown, CSV, JSON, plots
│   ├── models/
│   │   ├── __init__.py         # registry, factory, plugins, pricing catalogue
│   │   ├── base.py             # BaseEmbeddingModel contract
│   │   ├── local.py            # Ollama and sentence-transformers providers
│   │   └── providers.py        # OpenAI, Cohere, Voyage (+ list pricing)
│   ├── backends/               # 0.1.x compatibility shim (imports from models)
│   ├── schemas.py              # Document, Chunk, EvalExample, ModelSpec, tokens
│   ├── config.py               # Pydantic config, YAML loading, snapshots
│   ├── ingest.py               # discovery + txt/md/pdf/docx/html loaders
│   ├── chunking.py             # fixed / sentence / recursive chunkers
│   ├── evalset.py              # JSONL validation + Ollama question generation
│   ├── ollama_client.py        # retries, timeouts, legacy embed fallback
│   ├── cache.py                # SQLite embedding cache (file-based pipeline)
│   ├── index.py                # FAISS flat cosine + NumPy fallback
│   ├── resources.py            # RAM sampling and environment metadata
│   ├── catalog.py              # curated local model list + Ollama inventory
│   ├── benchmark.py            # file-based orchestrator behind `run`
│   ├── report.py               # HTML report and console summary for `run`
│   ├── py.typed
│   ├── errors.py
│   ├── logging_utils.py
│   └── utils.py                # hashing, atomic writes, seeds, percentiles
├── .github/workflows/ci.yml    # lint + tests (3.10-3.13) + wheel build
├── examples/
│   ├── config.yaml
│   ├── evalset.jsonl
│   └── docs/                   # sample overview/install/API/billing/security/errors
├── tests/
│   ├── conftest.py
│   ├── fake_backend.py         # deterministic hashing provider for tests
│   ├── test_datasets.py
│   ├── test_evaluator.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_reporter.py
│   ├── test_stress.py
│   ├── test_ollama_client.py
│   ├── test_chunking.py
│   ├── test_evalset.py
│   ├── test_ingest.py
│   ├── test_report.py
│   └── test_smoke.py
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=embedbench --cov-report=term-missing
ruff check src tests && ruff format --check src tests
```

The suite runs offline:

- Metrics: hand-computed Recall/Precision/MRR/nDCG/Hit, duplicate collapse, empty and
  single-result edge cases, k validation, aggregation; vectorised batch results checked
  against the scalar functions for every metric and cut-off; latency percentiles, QPS,
  memory footprint and cost arithmetic.
- Datasets: validation errors, chunk grouping, JSONL round-trips, corpus loading,
  needle placement and containment, chunk-size sweeps, deterministic query degradation.
- Evaluator: quality/latency/memory/cost capture, per-query consistency, JSON schema,
  failure isolation between models, recommendation weighting per priority.
- Models: base-contract prefixes/batching/normalisation/validation, registry, plugin
  loading, Ollama retries and missing models, mocked sentence-transformers and tokenizer
  counts, and OpenAI/Cohere/Voyage batching, input types, pricing, retries and missing
  SDK/key errors using fake SDK modules.
- Stress: needle depth breakdowns, chunk-size cells, query-noise deltas, failure cells
  and artifact writing.
- Reporter: Markdown/CSV/JSON output, console tables, and graceful plot degradation
  without matplotlib.
- Legacy pipeline: chunkers, ingestion (Markdown/HTML/TXT, real DOCX, mocked PDF),
  eval sets, HTML report escaping, CLI end-to-end with cache, and a failing model
  alongside a successful one.

No Ollama server, cloud key or Hugging Face download is required for the tests; the fake
hashing provider exercises the real cache, FAISS, metrics and report pipeline.

Release and PyPI publishing steps (trusted publishing, TestPyPI dry runs, manual `twine`
uploads) live in [PUBLISHING.md](PUBLISHING.md).

---

## Known limitations

- **Document-level relevance only.** `relevant_doc_ids` cannot point at a specific chunk.
  A document with many chunks gets one credit but can consume multiple top-k slots.
- **Synthetic haystacks use estimated tokens.** Needle-in-a-haystack sizes are measured
  with embedbench's word+punctuation heuristic (the same unit used for cost estimates),
  not the model's tokenizer, so a "32k" haystack may be a few percent off from the
  provider's own count. Relevance still points at the exact chunk containing the needle.
- **Query noise is rule-based.** Typos, truncation, word shuffling and synonym swaps are
  deterministic string transforms, not LLM paraphrases. They measure sensitivity to a
  class of degradation, not human paraphrase quality.
- **Synthetic questions are not verified.** Filters catch obvious vagueness and
  duplicates, but a local LLM can still write unanswerable or trivial questions. Review a
  sample and prefer your own JSONL for decisions.
- **Ollama tokens are external estimates.** The REST API returns embeddings, not a
  tokenizer. Cost-equivalent for Ollama is approximate and flagged in results.
- **Ollama VRAM is resident size, not peak** (from `/api/ps`). Ollama server RAM is not
  sampled by the benchmark process. HF CUDA peak uses PyTorch allocator stats; CPU/MPS
  have no VRAM number.
- **Exact search, not ANN.** FAISS `IndexFlatIP` is exact and simple, but memory and
  search cost grow linearly with corpus size. This is aimed at thousands to low hundreds
  of thousands of chunks. Sharded/IVF/HNSW indexes are future work.
- **Indexes are not persisted.** They are built and measured in memory for each model;
  only their serialized size is reported.
- **Latency is single-process and single-request.** Query throughput is not a concurrency
  benchmark. GPU batch throughput and server queueing under load are not modeled.
- **Chunk overlap is approximate for semantic strategies.** Sentence/recursive chunkers
  snap to boundaries, so overlap can be slightly less than `chunk_overlap`. A single
  oversized sentence is hard-split mid-word as a fallback.
- **PDF OCR is not included.** Scanned PDFs yield no text and are skipped with a warning.
  DOCX tables are flattened to ` | `-separated text; complex layouts lose structure.
- **HTML boilerplate removal is heuristic.** `nav`, `footer`, `script`, `style` and
  similar elements are removed, which could remove intentionally relevant text.
- **Cost is list price, not your contract.** Hosted prices come from the built-in table
  (override per model with `price_per_1m_tokens`); the file-based pipeline still applies
  one `cost_per_1m_tokens` to every model. Query/embedding price differences, volume
  discounts and cached-input pricing are out of scope.
- **Cache freshness is manual.** The cache key includes model name and format-affecting
  options, but not a Hugging Face revision or a local weight file hash.
- **Seeds do not guarantee bit-identical results** across hardware, PyTorch/FAISS
  versions, or Ollama model updates. They do make sampling and question filtering
  deterministic.
- **Single-machine, single-user tool.** No authentication, multi-tenancy, remote workers,
  scheduler or database backend. Reports contain document paths and queries; treat them
  as local data.
- **Plugins execute arbitrary Python.** Only load plugins you trust.
- **The synthetic filter is English-centric** (English stopwords and question patterns).
  Retrieval itself is model-dependent and can be multilingual; question generation and
  filtering may need adjustments for other languages.

---

## Troubleshooting

### `Model 'x' is not installed in Ollama`

```bash
ollama pull x
```

`embedbench list-models` shows installed Ollama models and the exact pull command.

### `Could not reach Ollama at ...`

Start the server:

```bash
ollama serve
```

Check `backends.ollama.host` / `--ollama-host`. Remote Ollama works if the port is
reachable; TLS/reverse-proxy auth is not implemented.

### HF model download or authentication errors

- Check the exact model ID on Hugging Face.
- Gated model: accept the license and run `huggingface-cli login`.
- Offline machine: pre-download with `huggingface-cli download <model>` and set
  `backends.hf.local_files_only: true`.
- `trust_remote_code` is disabled by default; enable only for trusted model repositories.

### `The Hugging Face backend requires sentence-transformers and torch`

```bash
pip install "sentence-transformers>=3.0"
```

### `FAISS unavailable; using slower NumPy exact search`

The NumPy fallback is correct and produces the same metrics. Install a compatible
`faiss-cpu` wheel for the faster index. On unusual platforms without a wheel, use conda
or the NumPy fallback.

### No documents / no text

- Check the folder and extensions.
- PDFs may be scans without a text layer; OCR them first.
- Binary `.txt` files with NUL bytes are rejected.
- Check `skipped_files` in `results.json` and the log for parse errors.

### Windows Hugging Face symlink warning

This is a cache-storage warning, not an embedbench failure. It is harmless; enabling
Windows Developer Mode or setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences it.

### Cache is making timings look too fast

That is expected. `results.json` contains `cache.hit_rate`, and the console prints a
cache reuse line. Use `--no-cache` for cold indexing numbers, and delete the cache
directory for a clean start.

### Generated questions are rejected

Try a stronger instruct model (`qwen2.5:7b`, `llama3.1:8b`), larger chunks, a lower
`min_question_words`, or a higher `max_chunks`. Run with `--verbose` to see per-chunk
generation diagnostics in the log.

---

## Design decisions and trade-offs

- **Local-first, two strong backends.** Ollama is a simple HTTP integration; HF uses
  sentence-transformers with the model's own tokenizer. Keeping both behind one interface
  avoids provider-specific benchmark code.
- **FAISS flat index with NumPy fallback.** Exact cosine is deterministic, requires no
  training, and is appropriate for local corpora. It does not scale to millions of chunks.
  The fallback keeps the tool usable where FAISS wheels are unavailable.
- **Document-level relevance, not chunk-level.** This matches the practical question
  “was the right source retrieved?” and avoids requiring users to label individual chunks.
  It makes precision sensitive to chunk count, which the report explains.
- **Synthetic set generation plus hard filters.** LLM-generated questions are useful, but
  unvalidated ones can flatter a model. The filters are transparent and deterministic;
  human-written JSONL remains the gold standard.
- **SQLite cache rather than one file per vector.** One file per embedding becomes
  unwieldy at thousands of chunks; SQLite gives atomic writes, compact storage, and no new
  dependency.
- **Measure what is visible.** Cold load, first-batch/first-query timings, cache hits,
  tokenizer counts, peak process RSS, GPU allocator peak, resident Ollama sizes and
  serialized index size are all reported; no single “score” hides the trade-offs.
- **Report as one portable file.** Inline CSS/JS and SVG charts mean no CDN, no server,
  no PII leaking to a chart library, and a report you can email or archive.
- **Strict config and graceful degradation.** Pydantic rejects typos, but a broken file or
  one failed model does not discard the run. Errors are actionable (`ollama pull`,
  `ollama serve`, install extras).
- **Heavy dependencies are declared.** The base install includes PyTorch-backed
  sentence-transformers to make the HF path work out of the box. Ollama-only users still
  pay the install cost, but no model is loaded at runtime. A future split could move HF
  into an extra.
- **No server, auth, or scheduler.** Those are a different product. The CLI, cache and
  artifacts are easy to script in CI or a local workflow.

---

## License

MIT. See [LICENSE](LICENSE).
