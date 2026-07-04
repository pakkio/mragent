"""MemoryAgent: active reconstruction of memory via iterative LLM-guided graph traversal."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llmwrapper import LLMWrapper, Prompt
from llmwrapper import RERANK_MODEL, RERANK_3_MODEL
from memory.graph import MemoryGraph
from memory.controller import TOOLS, dispatch

_SYSTEM = """\
You are a memory-augmented assistant. The available cues and tags are listed below.
STRICT RULES — follow exactly:
  1. Do NOT call list_cues or list_tags — the index is already provided.
  2. Step 1: call search_memory and get_by_tag in parallel (4-6 calls at once).
  3. Step 2: answer directly from what you retrieved. Do NOT search again.
  4. Maximum 2 steps total: one search step, then the answer.
Each memory item returned by a tool is prefixed with a citation number like [N] —
cite that number inline in your answer (e.g. [3]) instead of quoting the item's
text verbatim, to avoid duplicating content in your answer.
Only answer from what memory contains — say so if memory has nothing relevant."""

_EXTRACT_SYSTEM = """\
Extract structured metadata from text. Respond ONLY with valid JSON, no markdown.
Format: {"cues": ["keyword1", ...], "tags": ["topic1", ...]}
cues: 3-6 specific keywords/entities. tags: 2-3 broad topic categories."""

_EXTRACT_BATCH_SYSTEM = """\
Extract structured metadata for each numbered text below. Respond ONLY with a valid \
JSON array, no markdown, no extra keys. Each array element corresponds by position \
to a text: {"cues": ["keyword1", ...], "tags": ["topic1", ...]}
cues: 3-6 specific keywords/entities. tags: 2-3 broad topic categories.
Return exactly one JSON object per numbered text, in order, as a JSON array."""


def _rrf(ranked_a: list[str], ranked_b: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: merge two ranked lists into one."""
    scores: dict[str, float] = {}
    for rank, chunk in enumerate(ranked_a):
        scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)
    for rank, chunk in enumerate(ranked_b):
        scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class MemoryAgent:
    def __init__(self, llm: LLMWrapper, graph: MemoryGraph, max_steps: int = 3):
        self.llm = llm
        self.graph = graph
        self.max_steps = max_steps

    def store(
        self, content: str, cues: list[str], tags: list[str],
        docname: str | None = None,
        embed: bool = True,
    ) -> str:
        """Store content with manually specified cues and tags."""
        embedding_blob: bytes | None = None
        if embed:
            try:
                vec = self.llm.embed_text(content)
                import struct
                embedding_blob = struct.pack(f"{len(vec)}f", *vec)
            except Exception as e:
                print(f"[memory] embedding failed: {e}", flush=True)
        cid = self.graph.add_memory(content, cues, tags, docname=docname, embedding=embedding_blob)
        label = f" [{docname}]" if docname else ""
        emb_status = " +embed" if embedding_blob else ""
        print(f"[memory] stored {cid}{label}{emb_status} | cues={cues} | tags={tags}")
        return cid

    def _extract_metadata(self, content: str) -> tuple[list[str], list[str]]:
        """Single-chunk cue/tag extraction. Falls back to first-4-words heuristic."""
        try:
            raw = self.llm.call(Prompt(
                user=f"Text to index:\n\n{content}",
                system=_EXTRACT_SYSTEM,
            ))
            meta = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
            return meta.get("cues", []), meta.get("tags", [])
        except Exception as e:
            print(f"[memory] LLM extraction failed: {e} — using fallback cues", flush=True)
            words = content.split()
            return words[:4], ["general"]

    def _extract_metadata_batch(self, chunks: list[str]) -> list[tuple[list[str], list[str]]]:
        """Extract cues/tags for several chunks in a single LLM call.
        Falls back to per-chunk extraction if the batched call or JSON parse fails."""
        if len(chunks) == 1:
            return [self._extract_metadata(chunks[0])]

        numbered = "\n\n".join(f"[{i + 1}]\n{c}" for i, c in enumerate(chunks))
        try:
            raw = self.llm.call(Prompt(
                user=f"Texts to index:\n\n{numbered}",
                system=_EXTRACT_BATCH_SYSTEM,
            ))
            items = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
            if not isinstance(items, list) or len(items) != len(chunks):
                raise ValueError(
                    f"expected {len(chunks)} items, got "
                    f"{len(items) if isinstance(items, list) else type(items).__name__}"
                )
            return [(it.get("cues", []), it.get("tags", [])) for it in items]
        except Exception as e:
            print(f"[memory] batch extraction failed: {e} — falling back to per-chunk extraction", flush=True)
            return [self._extract_metadata(c) for c in chunks]

    def store_auto(self, content: str, docname: str | None = None, embed: bool = True) -> str:
        """Use LLM to extract cues/tags, then store.
        Falls back to first-4-words heuristic if the LLM call or JSON parse fails."""
        cues, tags = self._extract_metadata(content)
        return self.store(content, cues, tags, docname=docname, embed=embed)

    def embed_existing(self, workers: int = 5) -> int:
        """Compute & store embeddings for all chunks that don't have one yet."""
        missing = self.graph.chunks_without_embedding()
        if not missing:
            print("[embed] all chunks already have embeddings")
            return 0

        print(f"[embed] {len(missing)} chunk(s) need embeddings — {workers} workers")

        # Fetch all texts at once
        import struct
        with self.graph._lock:
            rows = {
                f"c{r[0]}": r[1]
                for r in self.graph._conn.execute(
                    f"SELECT id, text FROM nodes WHERE id IN ({','.join('?' * len(missing))})",
                    [int(c.lstrip('c')) for c in missing],
                ).fetchall()
            }

        # Batch embed
        ordered_ids = missing
        texts = [rows[cid] for cid in ordered_ids if cid in rows]
        if not texts:
            return 0

        vectors = self.llm.embed_batch(texts, workers=workers)

        # Store
        count = 0
        for cid, vec in zip(ordered_ids, vectors):
            blob = struct.pack(f"{len(vec)}f", *vec)
            nid = int(cid.lstrip("c"))
            with self.graph._lock:
                self.graph._conn.execute(
                    "UPDATE nodes SET embedding = ? WHERE id = ?", (blob, nid)
                )
                self.graph._conn.commit()
            count += 1

        print(f"[embed] stored {count} embedding(s)")

        # Rebuild FAISS index to include the new embeddings
        if hasattr(self.graph, '_faiss_index'):
            self.graph._build_faiss()

        return count

    def ingest_pdf(
        self, path: str, engine: str = "mistral-ocr",
        chunk_size: int = 800, workers: int = 5,
        alias: str | None = None,
        batch_pages: int = 0,
        start_chunk: int = 1,
        resume: bool = False,
        embed: bool = True,
        extract_batch_size: int = 8,
    ) -> int:
        """Extract text from a PDF, chunk it, and store each chunk in memory.

        If *alias* is given it becomes the docname for all chunks;
        otherwise the filename (without extension) is used.

        When *batch_pages* > 0 the PDF is split into sub-PDFs of that many
        pages and OCR runs in parallel across them — much faster for long
        documents. Requires ``pypdf`` (``pip install pypdf``).

        *start_chunk* (1-indexed): skip the first N-1 chunks and only store
        from chunk N onward. Useful for resuming an interrupted ingestion.

        *resume*: auto-detect how many chunks already exist for *docname* and
        continue from the next one. Overrides *start_chunk* if chunks are found.

        Cue/tag extraction is batched *extract_batch_size* chunks per LLM call
        (fewer round-trips); embeddings are computed afterwards in one batched
        pass over all newly-stored chunks (fewer, larger embedding requests).
        """
        docname = alias or os.path.splitext(os.path.basename(path))[0]

        # ── resolve start index ─────────────────────────────────────────────
        if resume:
            existing = self.graph.chunk_count(docname)
            if existing > 0:
                start_chunk = existing + 1
                print(f"[ingest] resuming '{docname}': {existing} chunk(s) already stored, starting at {start_chunk}")
            else:
                print(f"[ingest] --resume set but no existing chunks for '{docname}' — starting fresh")
        elif start_chunk > 1:
            print(f"[ingest] starting at chunk {start_chunk} for '{docname}'")

        # Plain-text files: read directly — no OCR needed
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            with open(path, encoding="utf-8") as f:
                text = f.read()
            print(f"[ingest] read {len(text)} chars directly from {os.path.basename(path)}")
        elif batch_pages > 0 and engine != "mistral-ocr-4":
            text = self._ingest_batched(path, engine, batch_pages, workers, docname)
        elif engine == "mistral-ocr-4":
            text = self.llm.extract_pdf_mistral_ocr4(path)
        else:
            text = self.llm.extract_pdf_openrouter(path, engine=engine)

        # Split on blank lines, merge small fragments up to chunk_size
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)

        total = len(chunks)
        if start_chunk > total:
            print(f"[ingest] start_chunk={start_chunk} > total chunks={total} — nothing to do")
            return 0
        chunks = chunks[start_chunk - 1:]

        print(f"[ingest] {total} total chunk(s) from {os.path.basename(path)} -> '{docname}'; "
              f"storing {len(chunks)} chunk(s) (chunk {start_chunk}–{total}) — {workers} workers, "
              f"extraction batched {extract_batch_size}/call")

        # group consecutive chunks so cue/tag extraction costs one LLM call per group
        numbered_chunks = list(enumerate(chunks, start_chunk))
        groups = [
            numbered_chunks[i:i + extract_batch_size]
            for i in range(0, len(numbered_chunks), extract_batch_size)
        ]

        done = [0]

        def process_group(group):
            texts = [c for _, c in group]
            metas = self._extract_metadata_batch(texts)
            for (idx, chunk), (cues, tags) in zip(group, metas):
                # embedding deferred to a single batched pass after all chunks are stored
                self.store(chunk, cues, tags, docname=docname, embed=False)
                done[0] += 1
                label = f"  [{idx}/{total}]" if start_chunk > 1 else f"  [{done[0]}/{total}]"
                print(f"{label} {chunk[:70].replace(chr(10), ' ')}...")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_group, g) for g in groups]
            for f in as_completed(futures):
                f.result()  # re-raise any exception

        if embed:
            print(f"[ingest] computing embeddings for {len(chunks)} chunk(s) in batches...", flush=True)
            self.embed_existing(workers=workers)

        return len(chunks)

    # ── batched PDF ingestion ────────────────────────────────────────────────

    def _ingest_batched(
        self, path: str, engine: str, batch_pages: int, workers: int, docname: str,
    ) -> str:
        """Split PDF into page batches, OCR each in parallel, return merged text.

        Each batch's extracted text is checkpointed to disk under
        ``.ocr_cache/<docname>/`` right after it's OCR'd. A batch whose
        checkpoint already exists is loaded from disk instead of re-sent to
        the OCR engine — so a crashed or interrupted run can be re-launched
        and only the batches that never completed get (re)charged/redone.
        """
        import tempfile

        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            raise ImportError(
                "--batch-pages requires pypdf. Install it: pip install pypdf"
            )

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(self.graph.path)) or ".", ".ocr_cache", docname)
        os.makedirs(cache_dir, exist_ok=True)

        reader = PdfReader(path)
        total = len(reader.pages)
        batches: list[tuple[int, int, str, str]] = []  # (start, end, temp_path, cache_path)

        tmpdir = tempfile.mkdtemp(prefix="mragent_pdf_")
        try:
            # split into page-range batches; skip writing sub-PDFs for already-cached batches
            for start in range(0, total, batch_pages):
                end = min(start + batch_pages, total)
                cache_path = os.path.join(cache_dir, f"batch_{start:04d}_{end:04d}.txt")
                tmp_path = os.path.join(tmpdir, f"batch_{start:04d}_{end:04d}.pdf")
                if not os.path.exists(cache_path):
                    writer = PdfWriter()
                    for i in range(start, end):
                        writer.add_page(reader.pages[i])
                    with open(tmp_path, "wb") as f:
                        writer.write(f)
                batches.append((start + 1, end, tmp_path, cache_path))

            n_cached = sum(1 for *_, cp in batches if os.path.exists(cp))
            n_pending = len(batches) - n_cached
            print(f"[ingest] {total} page(s) split into {len(batches)} batch(es) "
                  f"({batch_pages} pages each) — {n_cached} cached, {n_pending} to OCR "
                  f"({min(workers, n_pending) if n_pending else 0} parallel OCR workers)")

            # OCR each pending batch in parallel; cached ones load instantly
            results: dict[int, str] = {}

            def ocr_batch(item):
                idx, (pg_start, pg_end, tmp_path, cache_path) = item
                if os.path.exists(cache_path):
                    with open(cache_path, encoding="utf-8") as f:
                        txt = f.read()
                    print(f"  [batch {idx}/{len(batches)}] pages {pg_start}-{pg_end} "
                          f"cached ({len(txt)} chars)", flush=True)
                    return idx, txt
                t0 = time.time()
                size_kb = os.path.getsize(tmp_path) / 1024
                print(f"  [batch {idx}/{len(batches)}] pages {pg_start}-{pg_end} "
                      f"({size_kb:.0f} KB) OCR...", flush=True)
                txt = self.llm.extract_pdf_openrouter(tmp_path, engine=engine)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(txt)
                elapsed = time.time() - t0
                print(f"  [batch {idx}/{len(batches)}] pages {pg_start}-{pg_end} "
                      f"done in {elapsed:.1f}s ({len(txt)} chars)", flush=True)
                return idx, txt

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(ocr_batch, (i, b))
                    for i, b in enumerate(batches, 1)
                ]
                for f in as_completed(futures):
                    idx, txt = f.result()
                    results[idx] = txt

            # merge in order
            merged = "\n\n".join(results[i] for i in sorted(results))
            print(f"[ingest] merged {len(merged)} chars from {len(batches)} batch(es)")
            return merged

        finally:
            # clean up temp files
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def recall_fast(self, query: str, top_n: int = 32, rerank: bool = True) -> str:
        """Hybrid: query expansion → BM25 + embedding → reciprocal rank fusion → LLM re-rank → synthesize."""
        t0 = time.time()
        costs = {"embedding": 0.0, "rerank": 0.0, "sintesi": 0.0}

        # Step 1 — query expansion via graph cues
        expanded = self.graph.expand_query(query)
        if expanded != query:
            print(f"  [fast] query expanded: {query!r} → {expanded!r}", flush=True)

        # Step 2 — dual retrieval: BM25 + embedding
        fetch_n = top_n * 2 if rerank else top_n
        bm25_chunks = self.graph.rank_chunks(expanded, top_n=fetch_n)

        embed_chunks: list[str] = []
        try:
            cost_before = self.llm.total_cost
            q_vec = self.llm.embed_text(query)
            costs["embedding"] = self.llm.total_cost - cost_before
            embed_chunks = self.graph.rank_by_embedding(q_vec, top_n=fetch_n)
            print(f"  [fast] BM25={len(bm25_chunks)} + embed={len(embed_chunks)} chunks", flush=True)
        except Exception as e:
            print(f"  [fast] embedding retrieval failed: {e} — using BM25 only", flush=True)

        # Step 3 — reciprocal rank fusion
        if embed_chunks:
            chunks = _rrf(bm25_chunks, embed_chunks, k=60)
        else:
            chunks = bm25_chunks
        chunks = chunks[:fetch_n]  # trim to same budget

        if not chunks:
            return "Nothing relevant found in memory."

        # Step 4 — LLM re-ranking (optional)
        if rerank and len(chunks) > top_n:
            print(f"  [fast] re-ranking {len(chunks)} → {top_n} chunks via LLM...", flush=True)
            cost_before = self.llm.total_cost
            chunks = self._rerank(query, chunks, top_k=top_n)
            costs["rerank"] = self.llm.total_cost - cost_before
            print(f"  [fast] re-ranked to {len(chunks)} chunks", flush=True)

        # Step 5 — synthesize answer. Chunks are numbered [N] so the LLM can cite
        # them inline instead of quoting them verbatim (saves output tokens and
        # avoids duplicating the source text). The "Riferimenti" footer is built
        # afterwards from this same numbered list — not by the LLM — so it costs
        # no extra output tokens.
        context = "\n\n---\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
        print(f"  [fast] {len(chunks)} chunk(s), calling LLM...", flush=True)
        cost_before = self.llm.total_cost
        answer = self.llm.call(Prompt(
            user=f"Question: {query}\n\nRelevant memory:\n\n{context}",
            system=(
                "Answer the question using ONLY the provided memory. Be thorough and "
                "well-structured. Each memory item below is numbered like [N] — cite "
                "the relevant number(s) inline (e.g. [3]) instead of quoting the "
                "item's text verbatim, to avoid duplicating content in your answer."
            ),
        ))
        costs["sintesi"] = self.llm.total_cost - cost_before
        print(f"  [fast] done in {time.time() - t0:.1f}s total | cost ${sum(costs.values()):.5f}", flush=True)
        return self._finalize_answer(answer, chunks, costs)

    def _finalize_answer(self, answer: str, chunks: list[str], costs: dict[str, float]) -> str:
        """Frame the answer and append Riferimenti/Costi — shared by fast and deep recall."""
        return (
            self._frame(answer)
            + self._references_footer(answer, chunks)
            + self._costs_footer(costs)
        )

    @staticmethod
    def _frame(text: str) -> str:
        """Wrap *text* in a full terminal-width '=' border for readability."""
        import shutil
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        bar = "=" * width
        return f"{bar}\n{text}\n{bar}"

    @staticmethod
    def _costs_footer(costs: dict[str, float]) -> str:
        """Per-category cost breakdown, appended outside the answer frame."""
        label_width = max(len(label) for label in [*costs, "totale"])
        lines = ["\n\nCosti:"]
        for label, value in costs.items():
            lines.append(f"  {label:<{label_width}} : ${value:.5f}")
        lines.append(f"  {'totale':<{label_width}} : ${sum(costs.values()):.5f}")
        return "\n".join(lines)

    def _references_footer(self, answer: str, chunks: list[str], preview_len: int = 100) -> str:
        """Build the 'Riferimenti' footer, but only for chunks the LLM actually
        cited as [N] in its answer — a retrieved-but-ignored chunk gets no entry.
        Built entirely from data already in hand (no extra LLM call/output tokens)."""
        import re
        # Matches [3], [5,6], [5, 6, 7] — the model sometimes groups several
        # citations inside one bracket instead of writing [5][6][7].
        cited = sorted({
            n
            for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer)
            for n in (int(x) for x in group.split(","))
            if 1 <= n <= len(chunks)
        })
        if not cited:
            return ""
        lines = ["\n\nRiferimenti:"]
        for i in cited:
            chunk = chunks[i - 1]
            docname = self.graph.docname_for_text(chunk) or "?"
            preview = chunk[:preview_len].replace("\n", " ").strip()
            ellipsis = "..." if len(chunk) > preview_len else ""
            lines.append(f"[{i}] ({docname}) {preview}{ellipsis}")
        return "\n".join(lines)

    def _rerank(self, query: str, chunks: list[str], top_k: int,
                 model: str = RERANK_3_MODEL) -> list[str]:
        """Use Cohere Rerank to re-rank *chunks* by relevance to *query*."""
        return self.llm.rerank(query, chunks, top_n=top_k, model=model)

    def recall(self, query: str, deep: bool = False, top_n: int | None = None, rerank: bool = True) -> str:
        """Recall from memory. Fast mode (default): local ranking + single LLM call.
        Deep mode: iterative LLM-guided graph traversal."""
        if not deep:
            return self.recall_fast(query, top_n=top_n or 32, rerank=rerank)
        return self.recall_deep(query)

    def recall_deep(self, query: str) -> str:
        """Active reconstruction: LLM iteratively explores the graph to answer the query."""
        cues = self.graph.all_cues()
        tags = self.graph.all_tags()
        index = f"\nMemory index:\n  cues: {', '.join(cues)}\n  tags: {', '.join(tags)}"
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM + index},
            {"role": "user", "content": query},
        ]
        t0 = time.time()
        costs: dict[str, float] = {}
        # shared {chunk_text: citation_number} registry, filled in by dispatch()
        # as tool results come back, so every distinct chunk gets a stable [N]
        chunk_registry: dict[str, int] = {}

        for step in range(self.max_steps):
            print(f"  [step {step + 1}] thinking...", flush=True)
            t_step = time.time()
            cost_before = self.llm.total_cost
            msg = self.llm.call_with_tools(messages, TOOLS)
            step_cost = self.llm.total_cost - cost_before
            print(f"  [step {step + 1}] response in {time.time() - t_step:.1f}s", flush=True)
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # the LLM answered directly instead of requesting more searches
                costs["risposta finale"] = costs.get("risposta finale", 0.0) + step_cost
                print(f"  [recall] done in {time.time() - t0:.1f}s total | cost ${sum(costs.values()):.5f}", flush=True)
                return self._finalize_deep_answer(msg.get("content") or "", chunk_registry, costs)

            costs[f"ricerca in memoria (step {step + 1})"] = step_cost

            def _run_tool(tc):
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                print(f"  [step {step + 1}] {fn['name']}({args})")
                result = dispatch(self.graph, fn["name"], args, registry=chunk_registry)
                return tc["id"], result

            with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                results = dict(pool.map(_run_tool, tool_calls))

            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": results[tc["id"]],
                })

        # max steps reached — force final answer
        print(f"  [max steps] forcing answer...", flush=True)
        messages.append({"role": "user", "content": "Synthesize a final answer from what you found."})
        cost_before = self.llm.total_cost
        msg = self.llm.call_with_tools(messages, [])
        costs["risposta finale (max passi raggiunti)"] = self.llm.total_cost - cost_before
        print(f"  [recall] done in {time.time() - t0:.1f}s total | cost ${sum(costs.values()):.5f}", flush=True)
        return self._finalize_deep_answer(msg.get("content") or "", chunk_registry, costs)

    def _finalize_deep_answer(
        self, answer: str, chunk_registry: dict[str, int], costs: dict[str, float],
    ) -> str:
        """Turn the {text: N} citation registry into an ordered chunk list, then
        delegate to the same framing/footer logic used by recall_fast."""
        ordered_chunks: list[str] = [""] * len(chunk_registry)
        for text, num in chunk_registry.items():
            ordered_chunks[num - 1] = text
        return self._finalize_answer(answer, ordered_chunks, costs)
