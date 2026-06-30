"""MemoryAgent: active reconstruction of memory via iterative LLM-guided graph traversal."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llmwrapper import LLMWrapper, Prompt
from memory.graph import MemoryGraph
from memory.controller import TOOLS, dispatch

_SYSTEM = """\
You are a memory-augmented assistant. The available cues and tags are listed below.
STRICT RULES — follow exactly:
  1. Do NOT call list_cues or list_tags — the index is already provided.
  2. Step 1: call search_memory and get_by_tag in parallel (4-6 calls at once).
  3. Step 2: answer directly from what you retrieved. Do NOT search again.
  4. Maximum 2 steps total: one search step, then the answer.
Only answer from what memory contains — say so if memory has nothing relevant."""

_EXTRACT_SYSTEM = """\
Extract structured metadata from text. Respond ONLY with valid JSON, no markdown.
Format: {"cues": ["keyword1", ...], "tags": ["topic1", ...]}
cues: 3-6 specific keywords/entities. tags: 2-3 broad topic categories."""


class MemoryAgent:
    def __init__(self, llm: LLMWrapper, graph: MemoryGraph, max_steps: int = 3):
        self.llm = llm
        self.graph = graph
        self.max_steps = max_steps

    def store(
        self, content: str, cues: list[str], tags: list[str],
        docname: str | None = None,
    ) -> str:
        """Store content with manually specified cues and tags."""
        cid = self.graph.add_memory(content, cues, tags, docname=docname)
        label = f" [{docname}]" if docname else ""
        print(f"[memory] stored {cid}{label} | cues={cues} | tags={tags}")
        return cid

    def store_auto(self, content: str, docname: str | None = None) -> str:
        """Use LLM to extract cues/tags, then store."""
        raw = self.llm.call(Prompt(
            user=f"Text to index:\n\n{content}",
            system=_EXTRACT_SYSTEM,
        ))
        try:
            meta = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
            cues = meta.get("cues", [])
            tags = meta.get("tags", [])
        except Exception:
            words = content.split()
            cues = words[:4]
            tags = ["general"]
        return self.store(content, cues, tags, docname=docname)

    def ingest_pdf(
        self, path: str, engine: str = "mistral-ocr",
        chunk_size: int = 800, workers: int = 5,
        alias: str | None = None,
        batch_pages: int = 0,
    ) -> int:
        """Extract text from a PDF, chunk it, and store each chunk in memory.

        If *alias* is given it becomes the docname for all chunks;
        otherwise the filename (without extension) is used.

        When *batch_pages* > 0 the PDF is split into sub-PDFs of that many
        pages and OCR runs in parallel across them — much faster for long
        documents. Requires ``pypdf`` (``pip install pypdf``).
        """
        docname = alias or os.path.splitext(os.path.basename(path))[0]

        # Plain-text files: read directly — no OCR needed
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            with open(path, encoding="utf-8") as f:
                text = f.read()
            print(f"[ingest] read {len(text)} chars directly from {os.path.basename(path)}")
        elif batch_pages > 0 and engine != "mistral-ocr-4":
            text = self._ingest_batched(path, engine, batch_pages, workers)
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

        print(f"[ingest] {len(chunks)} chunk(s) from {os.path.basename(path)} -> '{docname}' — {workers} workers")

        done = [0]

        def process(item):
            i, chunk = item
            self.store_auto(chunk, docname=docname)
            done[0] += 1
            print(f"  [{done[0]}/{len(chunks)}] {chunk[:70].replace(chr(10), ' ')}...")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process, (i, c)) for i, c in enumerate(chunks, 1)]
            for f in as_completed(futures):
                f.result()  # re-raise any exception

        return len(chunks)

    # ── batched PDF ingestion ────────────────────────────────────────────────

    def _ingest_batched(
        self, path: str, engine: str, batch_pages: int, workers: int,
    ) -> str:
        """Split PDF into page batches, OCR each in parallel, return merged text."""
        import tempfile

        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            raise ImportError(
                "--batch-pages requires pypdf. Install it: pip install pypdf"
            )

        reader = PdfReader(path)
        total = len(reader.pages)
        batches: list[tuple[int, int, str]] = []  # (start, end, temp_path)

        tmpdir = tempfile.mkdtemp(prefix="mragent_pdf_")
        try:
            # split into page-range batches
            for start in range(0, total, batch_pages):
                end = min(start + batch_pages, total)
                writer = PdfWriter()
                for i in range(start, end):
                    writer.add_page(reader.pages[i])
                tmp_path = os.path.join(tmpdir, f"batch_{start:04d}_{end:04d}.pdf")
                with open(tmp_path, "wb") as f:
                    writer.write(f)
                batches.append((start + 1, end, tmp_path))

            print(f"[ingest] {total} page(s) split into {len(batches)} batch(es) "
                  f"({batch_pages} pages each) — {min(workers, len(batches))} parallel OCR workers")

            # OCR each batch in parallel
            results: dict[int, str] = {}

            def ocr_batch(item):
                idx, (pg_start, pg_end, tmp_path) = item
                t0 = time.time()
                size_kb = os.path.getsize(tmp_path) / 1024
                print(f"  [batch {idx}/{len(batches)}] pages {pg_start}-{pg_end} "
                      f"({size_kb:.0f} KB) OCR...", flush=True)
                txt = self.llm.extract_pdf_openrouter(tmp_path, engine=engine)
                elapsed = time.time() - t0
                print(f"  [batch {idx}/{len(batches)}] pages {pg_start}-{pg_end} "
                      f"done in {elapsed:.1f}s ({len(txt)} chars)", flush=True)
                return idx, txt

            with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as pool:
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

    def recall_fast(self, query: str, top_n: int = 8) -> str:
        """Single-shot: rank chunks locally, synthesize in one LLM call."""
        t0 = time.time()
        chunks = self.graph.rank_chunks(query, top_n=top_n)
        if not chunks:
            return "Nothing relevant found in memory."
        context = "\n\n---\n\n".join(chunks)
        print(f"  [fast] {len(chunks)} chunk(s) ranked locally, calling LLM...", flush=True)
        answer = self.llm.call(Prompt(
            user=f"Question: {query}\n\nRelevant memory:\n\n{context}",
            system="Answer the question using ONLY the provided memory. Be thorough and well-structured.",
        ))
        print(f"  [fast] done in {time.time() - t0:.1f}s total", flush=True)
        return answer

    def recall(self, query: str, deep: bool = False) -> str:
        """Recall from memory. Fast mode (default): local ranking + single LLM call.
        Deep mode: iterative LLM-guided graph traversal."""
        if not deep:
            return self.recall_fast(query)
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

        for step in range(self.max_steps):
            print(f"  [step {step + 1}] thinking...", flush=True)
            t_step = time.time()
            msg = self.llm.call_with_tools(messages, TOOLS)
            print(f"  [step {step + 1}] response in {time.time() - t_step:.1f}s", flush=True)
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                print(f"  [recall] done in {time.time() - t0:.1f}s total", flush=True)
                return msg.get("content") or ""

            def _run_tool(tc):
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                print(f"  [step {step + 1}] {fn['name']}({args})")
                result = dispatch(self.graph, fn["name"], args)
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
        msg = self.llm.call_with_tools(messages, [])
        print(f"  [recall] done in {time.time() - t0:.1f}s total", flush=True)
        return msg.get("content") or ""
