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

### Fast recall (default) — hybrid pipeline

1. **Query expansion** — related cues found in the graph are appended to the query.
2. **Dual retrieval** — FTS5 BM25 keyword search and FAISS embedding similarity run on the same budget (2× top-N candidates each).
3. **Reciprocal rank fusion** — the two ranked lists are merged (`k=60`); chunks found by both retrievers float to the top.
4. **Rerank** (optional, on by default) — Cohere Rerank trims the fused candidates to top-N (default 32) by relevance.
5. **Synthesis** — one LLM call: "Answer using only this context."

**Cost:** 1 embedding call + 1 rerank call + 1 LLM call. Falls back to BM25-only if embeddings fail. Best for focused queries.

### Deep recall (`--deep`)

1. Inject the full cue and tag index into the system prompt so the LLM knows what's available.
2. The LLM calls tools (`search_memory`, `get_by_tag`) in parallel — multiple graph lookups in one round-trip via `ThreadPoolExecutor`.
3. Tool results are appended to the message history and the LLM synthesizes a final answer.
4. Capped at `max_steps=3` to bound cost; a forced-answer turn fires if the cap is reached.

**Cost:** 2–4 LLM calls. Better for open-ended or multi-faceted queries.

---

## PDF Ingestion Pipeline

```
PDF file
  │
  ▼
OCR (OpenRouter file-parser plugin or Mistral OCR 4 direct)
  │
  ▼
Split on blank lines → merge paragraphs up to chunk_size (default 800 chars)
  │
  ▼
ThreadPoolExecutor (default 5 workers)
  │  ├─ chunk 1 → LLM: extract cues+tags → embed → store in graph
  │  ├─ chunk 2 → LLM: extract cues+tags → embed → store in graph
  │  └─ ...
  ▼
memory.db + FTS5 + FAISS updated after each chunk
(interrupted? --resume continues from the last stored chunk)
```

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

PDF ingestion dispatches all chunk-processing tasks (OCR text → cue/tag extraction → graph write) to a thread pool. On a 50-page document split into 30 chunks with 5 workers, wall-clock time is ~6× faster than sequential processing.

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
| `--chunk-size` | `800` | Max chars per PDF chunk |
| `--workers` | `5` | Parallel workers for PDF ingestion |
| `--deep` | off | Enable iterative graph traversal on recall |

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
