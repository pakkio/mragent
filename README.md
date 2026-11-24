# mragent

A lightweight, multi-role LLM agent with a persistent graph-based memory system and hybrid retrieval (FTS5 BM25 + embeddings + rerank). Designed for CLI-first workflows where you want an AI that can remember, retrieve, and reason over accumulated knowledge — everything lives in a single SQLite file plus an optional FAISS index, no heavyweight framework.

---

## Why mragent?

Most LLM tools treat every session as stateless. You repeat context, re-explain background, or paste in documents every time. mragent fixes this by giving the agent a persistent, structured memory it can write to and read from across sessions.

The backbone is a graph of **cues** (specific keywords), **tags** (broad topics), and **content** (the actual text) — traversed either locally or by the LLM itself. On top of that, recall is hybrid: FTS5 BM25 keyword search and embedding similarity run in parallel, are fused with reciprocal rank fusion, and optionally re-ranked with Cohere Rerank. Each layer degrades gracefully — no FAISS installed means brute-force cosine, no embeddings means BM25 only, no FTS5 means an in-memory BM25 with Snowball stemming and diacritic folding.

---

## Architecture

```
main.py          — CLI entry point, command dispatch
llmwrapper.py    — thin HTTP wrapper over OpenRouter (no SDK dependency):
                   chat, tools, embeddings, Cohere rerank, OCR
agent/
  agent.py       — MemoryAgent: store, recall (fast + deep), PDF ingest, RRF fusion
memory/
  graph.py       — Cue-Tag-Content graph on SQLite: FTS5 BM25, FAISS vector
                   search (optional), in-memory BM25 fallback, query expansion
  controller.py  — LLM tool schemas (search_memory, get_by_tag, …)
memory.db        — SQLite database: nodes, edges, embeddings, FTS5 index
                   (auto-created on first write)
memory.faiss     — persisted FAISS index (auto-rebuilt from memory.db)
.ocr_cache/      — per-batch OCR text checkpoints, keyed by docname
                   (gitignored; lets an interrupted --batch-pages ingest
                   resume without re-paying for already-OCR'd pages)
```

### The Cue-Tag-Content Graph

Every piece of stored knowledge is represented as three node types connected by edges:

```
cue:"python"  ──►  tag:"programming"  ──►  content:"Python is a …"
cue:"GIL"     ──►  tag:"concurrency"  ──►  content:"The Global …"
```

- **Cue nodes** are specific keywords or named entities (3–6 per item). They point to tag nodes.
- **Tag nodes** are broad topic categories (2–3 per item). They point to content nodes.
- **Content nodes** hold the raw text of a stored memory chunk.

This two-level indirection lets the system answer both narrow queries ("find everything about GIL") and broad queries ("what do I know about concurrency?") without embedding arithmetic.

Nodes are deduplicated: storing two items with the same tag reuses the existing tag node and fans out its edges. The graph lives in a single SQLite database (`memory.db`, WAL mode) committed per-write, protected by a threading lock for concurrent ingestion. A legacy `memory.json` is migrated automatically on first run.

---

## Commands

### One-shot / stateless

```bash
# Ask a single question (no memory)
python main.py ask What is the capital of France?

# Interactive multi-turn chat
python main.py chat --system "You are a concise tutor."

# Summarize a file or stdin (3–5 bullet points)
python main.py summarize report.txt
cat report.txt | python main.py summarize

# Multi-step research pipeline (brainstorm → distill → briefing)
python main.py research "Byzantine fault tolerance"
```

### Memory-aware

```bash
# Store text with LLM-extracted cues and tags
python main.py remember "The GIL prevents true thread parallelism in CPython."

# Store with manually specified metadata (skips LLM extraction call)
python main.py remember --cues "GIL,CPython,threads" --tags "python,concurrency" \
    "The GIL prevents true thread parallelism in CPython."

# Store from a file
python main.py remember --file notes.txt

# Recall — fast mode (default): local term ranking + single LLM call
python main.py recall "How does Python handle concurrency?"

# Recall — deep mode: LLM iteratively traverses the graph (slower, more thorough)
python main.py recall --deep "How does Python handle concurrency?"

# Ingest a PDF (OCR → chunk → auto-store each chunk)
python main.py ingest paper.pdf
python main.py ingest paper.pdf --engine mistral-ocr-4 --chunk-size 600 --workers 8

# Inspect what is stored
python main.py memory
```

---

## Recall Modes

Both modes return the same output shape: the answer framed between full terminal-width
`=` borders, followed by a **Riferimenti** footer and a **Costi** breakdown — both built
in Python from data already in hand, so they cost zero extra LLM output tokens.

```
====================================================================
<answer, citing sources inline as [3], [7], ...>
====================================================================

Riferimenti:
[3] (docname) preview of the chunk text...
[7] (docname) preview of the chunk text...

Costi:
  embedding : $0.00000
  rerank    : $0.00100
  sintesi   : $0.00138
  totale    : $0.00238
```

Every chunk sent to the LLM is numbered `[N]` in the prompt, and the system prompt
instructs the model to **cite the number inline instead of quoting the chunk
verbatim** — this cuts duplicated source text out of the answer and shrinks output
tokens. The Riferimenti footer only lists numbers the model actually cited; a
retrieved-but-ignored chunk gets no entry.

### Fast recall (default) — hybrid pipeline

1. **Query expansion** — related cues found in the graph are appended to the query.
2. **Dual retrieval** — FTS5 BM25 keyword search and FAISS embedding similarity run on the same budget (2× top-N candidates each).
3. **Reciprocal rank fusion** — the two ranked lists are merged (`k=60`); chunks found by both retrievers float to the top.
4. **Rerank** (optional, on by default) — Cohere Rerank trims the fused candidates to top-N (default 32) by relevance.
5. **Synthesis** — one LLM call, citing chunks as `[N]`.

**Cost breakdown:** `embedding` + `rerank` + `sintesi` (synthesis), each tracked individually. Falls back to BM25-only if embeddings fail. Best for focused queries.

### Deep recall (`--deep`)

1. Inject the full cue and tag index into the system prompt so the LLM knows what's available.
2. The LLM calls tools (`search_memory`, `get_by_tag`) in parallel — multiple graph lookups in one round-trip via `ThreadPoolExecutor`. Each distinct chunk a tool returns is assigned a stable `[N]` the first time it's seen (shared registry across all steps).
3. Tool results are appended to the message history and the LLM synthesizes a final answer, citing `[N]`.
4. Capped at `max_steps=3` to bound cost; a forced-answer turn fires if the cap is reached.

**Cost breakdown:** one entry per step — `ricerca in memoria (step N)` while the LLM is still searching, `risposta finale` (or `risposta finale (max passi raggiunti)` if the step cap forced it) for the synthesis call. Better for open-ended or multi-faceted queries.

---

## PDF Ingestion Pipeline

```
PDF file
  │
  ▼
OCR (OpenRouter file-parser plugin or Mistral OCR 4 direct)
  │  --batch-pages N splits the PDF into N-page batches, OCR'd in parallel;
  │  each batch's extracted text is checkpointed to .ocr_cache/<docname>/ as
  │  soon as it completes, so a crashed/interrupted run resumes for free —
  │  only the batches that never finished get (re)sent to the OCR engine.
  ▼
Split on blank lines → merge paragraphs up to chunk_size (default 1500 chars)
  │
  ▼
ThreadPoolExecutor (default 16 workers)
  │  ├─ group of extract-batch-size chunks (default 8) → ONE LLM call
  │  │  extracts cues+tags for all of them at once (falls back to
  │  │  per-chunk extraction if the batched JSON parse fails)
  │  └─ ... stored in the graph with embed=False (deferred)
  ▼
Single batched, parallelized embedding pass over every newly-stored chunk
(embed_batch, 50 texts/request, workers concurrent requests)
  │
  ▼
memory.db + FTS5 + FAISS updated
(interrupted before this point? --resume continues from the last stored chunk;
 OCR itself resumes from .ocr_cache/ regardless)
```

Cue/tag extraction and embeddings used to cost one HTTP round-trip *per chunk*;
batching both cuts the number of LLM/embedding requests by roughly
`extract_batch_size` and `50×` respectively, which is most of what makes ingest
fast — see [Optimisations](#optimisations) below.

OpenRouter occasionally returns HTTP 200 with a malformed body (moderation,
upstream hiccup) instead of raising a clean error — `LLMWrapper._request`
detects a missing `choices` key and retries it like a 5xx, instead of crashing
the whole ingest on a bare `KeyError`.

OCR engines available:

| Engine | Route | Notes |
|---|---|---|
| `mistral-ocr` | OpenRouter file-parser | Default, $2/1000 pages |
| `mistral-ocr-4` | Mistral API direct | Requires `MISTRAL_API_KEY`, best quality |
| `cloudflare-ai` | OpenRouter file-parser | Alternative |
| `native` | OpenRouter file-parser | Model-native PDF reading |

---

## Optimisations

### No SDK, no local embedding model

`llmwrapper.py` uses only `urllib.request` from the standard library — zero third-party HTTP dependencies. Embeddings and reranking go through the OpenRouter API, so no local model weights are downloaded. Core runtime dependencies are just `python-dotenv` and `snowballstemmer`; `faiss-cpu`/`numpy` are an optional extra (brute-force cosine fallback without them). This keeps cold-start time low and the install footprint minimal.

### Local ranking before LLM

Fast recall ranks content nodes locally — FTS5 BM25 (persisted index, no per-query tokenization) fused with FAISS vector search. The LLM only sees the top-ranked chunks, not the entire memory. For a 1000-chunk graph, this reduces context sent to the LLM by 99% while keeping recall quality high for specific queries.

### Parallel tool execution in deep recall

When the LLM emits multiple tool calls in one step, mragent runs all of them in parallel via `ThreadPoolExecutor(max_workers=len(tool_calls))`. A 4-tool batch that would take 4 × 300 ms serially completes in ~300 ms.

### Parallel PDF chunking

PDF ingestion dispatches all chunk-processing tasks (OCR text → cue/tag extraction → graph write) to a thread pool of `--workers` (default 16) threads.

### Batched cue/tag extraction and embeddings

The two most expensive per-chunk steps used to each cost one HTTP round-trip
per chunk. Both are now batched:

- **Cue/tag extraction** bundles `--extract-batch-size` (default 8) chunks into
  a single LLM call asking for a JSON array of `{cues, tags}`, one per chunk,
  in order — falling back to per-chunk extraction if the batched call or JSON
  parse fails.
- **Embeddings** are deferred during the per-chunk store loop (`embed=False`)
  and computed afterwards in one pass over every new chunk, via
  `embed_batch(..., workers=N)`, which both batches requests (50 texts each)
  and fires them concurrently instead of one request per chunk.

Together these cut the number of network round-trips for a 1000-chunk ingest
from ~2000 (1 extraction + 1 embedding call per chunk) to roughly
125 extraction calls + 20 embedding calls.

### Checkpointed OCR batches

`--batch-pages` splits a PDF into page-range batches OCR'd in parallel; each
batch's extracted text is written to `.ocr_cache/<docname>/` as soon as it
completes. If the process crashes or a batch fails (OpenRouter timeout,
malformed response after retries exhausted), re-running the same `ingest`
command loads every already-OCR'd batch straight from disk — no OCR cost is
paid twice — and only retries the batches that never finished.

### Graph deduplication

Cue and tag nodes are deduplicated at write time (`_find` before `_create`). A tag like "machine-learning" shared across 200 documents is stored as one node with 200 edges, not 200 nodes. This keeps the graph compact and tag-based recall O(edges) rather than O(nodes²).

### Step cap + forced answer

Deep recall is bounded at `max_steps=3`. If the LLM hasn't converged after 3 agentic steps, a final turn with no tools forces it to synthesize from what it already retrieved. This prevents runaway loops and caps API cost at 4 calls maximum.

### System prompt pre-injects the index

The cue and tag index is injected into the system prompt before the first deep-recall step. The LLM can plan its `search_memory` and `get_by_tag` calls without first calling `list_cues`/`list_tags`. The system prompt explicitly forbids those listing calls, saving at least one tool round-trip per query.

---

## Comparison: mragent vs llmwiki

[llmwiki](https://github.com/pakkio/llmwiki) is a companion project focused on building and querying a structured knowledge base from Wikipedia-style articles. Here is how the two projects differ:

| Dimension | mragent | llmwiki |
|---|---|---|
| **Primary input** | Arbitrary text, files, PDFs | Structured wiki / article content |
| **Memory model** | Graph (Cue → Tag → Content) | Flat or document-centric index |
| **Retrieval** | Hybrid BM25 + embeddings + RRF + rerank, or LLM-guided graph traversal | Keyword / semantic search over articles |
| **Agent loop** | Agentic tool use (search_memory, get_by_tag) with step cap | Direct retrieval, no multi-step agent loop |
| **Write path** | LLM extracts metadata on ingest | Metadata comes from document structure |
| **PDF support** | Yes — OCR + chunking pipeline | No |
| **Persistence** | Single SQLite `memory.db` (+ optional FAISS index) | Depends on wiki source |
| **Dependencies** | `python-dotenv`, `snowballstemmer` (FAISS optional) | Varies |
| **Best for** | Personal knowledge accumulation over time | Querying a fixed knowledge corpus |

In short: **mragent** is a general-purpose memory layer you write to incrementally; **llmwiki** is a read-optimized interface over a pre-existing structured corpus.

---

## Setup

**Requirements:** Python ≥ 3.12, an [OpenRouter](https://openrouter.ai) API key.

```bash
# Install dependencies
pip install python-dotenv snowballstemmer
# or with Poetry
poetry install

# Optional: FAISS vector search (otherwise brute-force cosine fallback)
pip install faiss-cpu numpy          # or: poetry install --extras vector

# Optional: parallel PDF batching for ingest (--batch-pages)
pip install pypdf                    # or: poetry install --extras pdf

# Set your API key
echo "OPENROUTER_APIKEY=sk-or-..." > .env

# Optional: Mistral OCR 4 direct (higher quality PDF extraction)
echo "MISTRAL_API_KEY=..." >> .env
```

The repository ships no pre-built memory: `memory.db`, the FAISS index, and source PDFs are gitignored. Build your own memory with `remember` / `ingest` — the FTS5 and FAISS indexes are created and maintained automatically.

The default model is `deepseek/deepseek-v4-flash` via OpenRouter. Override per-run with `--model`:

```bash
python main.py --model anthropic/claude-sonnet-4-6 ask "Explain monads."
```

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_APIKEY` | Yes | OpenRouter API key |
| `MISTRAL_API_KEY` | No | Required only for `--engine mistral-ocr-4` |

| Flag | Default | Description |
|---|---|---|
| `--model` | `deepseek/deepseek-v4-flash` | Any OpenRouter model slug |
| `--chunk-size` | `1500` | Max chars per PDF chunk |
| `--workers` | `16` | Parallel workers for cue/tag extraction, embedding, and OCR batches |
| `--extract-batch-size` | `8` | Chunks bundled into one cue/tag-extraction LLM call |
| `--batch-pages` | `0` (off) | Split PDF into N-page batches, OCR'd in parallel and checkpointed to `.ocr_cache/` |
| `--no-embed` | off | Skip embedding computation during ingest |
| `--resume` | off | Auto-detect existing chunks for the docname and continue from the next one |
| `--deep` | off | Enable iterative graph traversal on recall |
| `--no-rerank` | off | Skip the Cohere rerank step on fast recall |

---

## Example session

```bash
# Store some knowledge
python main.py remember "Transformers use self-attention to model long-range dependencies."
python main.py remember --file "attention_is_all_you_need.txt"
python main.py ingest research_papers.pdf --workers 8

# Check what's stored
python main.py memory
# Memory: 47 item(s), 183 cue(s), 31 tag(s)
# Tags : machine-learning, nlp, attention, ...

# Query it
python main.py recall "How does attention work in transformers?"

# Deeper investigation
python main.py recall --deep "What are the computational trade-offs of self-attention vs convolution?"
```

---

## Running tests

```bash
pytest                    # unit tests only
pytest -m live            # includes live API calls (requires OPENROUTER_APIKEY)
```

---

## Documents and copyright

This repository does not ship third-party documents. Put the PDFs you want to work with in the project folder yourself: `*.pdf` is in `.gitignore`, so they stay local and are never committed.
