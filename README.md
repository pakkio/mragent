# mragent

A lightweight, multi-role LLM agent with a persistent graph-based memory system. Designed for CLI-first workflows where you want an AI that can remember, retrieve, and reason over accumulated knowledge — without a database, vector store, or heavyweight framework.

---

## Why mragent?

Most LLM tools treat every session as stateless. You repeat context, re-explain background, or paste in documents every time. mragent fixes this by giving the agent a persistent, structured memory it can write to and read from across sessions.

The core insight: memory doesn't need semantic embeddings to be useful. A simple graph of **cues** (specific keywords), **tags** (broad topics), and **content** (the actual text) — traversed either locally or by the LLM itself — recovers the right information fast, with zero vector infrastructure.

---

## Architecture

```
main.py          — CLI entry point, command dispatch
llmwrapper.py    — thin HTTP wrapper over OpenRouter (no SDK dependency)
agent/
  agent.py       — MemoryAgent: store, recall (fast + deep), PDF ingest
memory/
  graph.py       — Cue-Tag-Content graph, JSON persistence, ranking
  controller.py  — LLM tool schemas (search_memory, get_by_tag, …)
memory.json      — persisted graph (auto-created on first write)
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

Nodes are deduplicated: storing two items with the same tag reuses the existing tag node and fans out its edges. The graph is persisted as a single `memory.json` after every write, protected by a threading lock for concurrent ingestion.

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

### Fast recall (default)

1. Tokenise the query into terms.
2. Score every content node by term-frequency overlap (pure Python, no LLM).
3. Take top-N (default 8) by score.
4. Feed those chunks to the LLM in a single call: "Answer using only this context."

**Cost:** 1 LLM call, sub-second graph traversal. Best for focused queries.

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
  │  ├─ chunk 1 → LLM: extract cues+tags → store in graph
  │  ├─ chunk 2 → LLM: extract cues+tags → store in graph
  │  └─ ...
  ▼
memory.json updated after each chunk
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

### No SDK, no embedding model

`llmwrapper.py` uses only `urllib.request` from the standard library. Zero third-party HTTP dependencies. The only runtime dependency is `python-dotenv`. This keeps cold-start time under 100 ms and the install footprint minimal.

### Local ranking before LLM

Fast recall scores content nodes with a term-frequency loop in pure Python. The LLM only sees the top-ranked chunks, not the entire memory. For a 1000-chunk graph, this reduces context sent to the LLM by 99% while keeping recall quality high for specific queries.

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
| **Retrieval** | Term-frequency ranking + LLM-guided graph traversal | Keyword / semantic search over articles |
| **Agent loop** | Agentic tool use (search_memory, get_by_tag) with step cap | Direct retrieval, no multi-step agent loop |
| **Write path** | LLM extracts metadata on ingest | Metadata comes from document structure |
| **PDF support** | Yes — OCR + chunking pipeline | No |
| **Persistence** | Single `memory.json`, schema-free | Depends on wiki source |
| **Dependencies** | `python-dotenv` only | Varies |
| **Best for** | Personal knowledge accumulation over time | Querying a fixed knowledge corpus |

In short: **mragent** is a general-purpose memory layer you write to incrementally; **llmwiki** is a read-optimized interface over a pre-existing structured corpus.

---

## Setup

**Requirements:** Python ≥ 3.12, an [OpenRouter](https://openrouter.ai) API key.

```bash
# Install dependencies
pip install python-dotenv
# or with Poetry
poetry install

# Set your API key
echo "OPENROUTER_APIKEY=sk-or-..." > .env

# Optional: Mistral OCR 4 direct (higher quality PDF extraction)
echo "MISTRAL_API_KEY=..." >> .env
```

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
