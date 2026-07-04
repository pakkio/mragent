import base64
import http.client
import os
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
EMBED_MODEL = "openai/text-embedding-3-small"
RERANK_MODEL = "cohere/rerank-4-pro"
RERANK_3_MODEL = "cohere/rerank-v3.5"


def _load_config() -> dict:
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


_config = _load_config()
DEFAULT_MODEL = _config.get("model", "deepseek/deepseek-v4-flash")


@dataclass
class Prompt:
    user: str
    system: Optional[str] = None
    history: list[dict] = field(default_factory=list)

    def to_messages(self) -> list[dict]:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": self.user})
        return messages


class LLMWrapper:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_APIKEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_APIKEY not set in environment or .env")

    def _request(self, payload: dict, timeout: int = 120, max_retries: int = 3) -> dict:
        model = payload.get("model", "?")
        plugins = payload.get("plugins")
        label = f"{model}" + (f" +plugins={[p['id'] for p in plugins]}" if plugins else "")
        last_error = None

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                OPENROUTER_API_URL,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            print(f"[openrouter] -> {label}", flush=True)
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    print(f"[openrouter] <- headers received ({time.time()-t0:.1f}s), reading body...", flush=True)
                    data = json.loads(resp.read())
                    elapsed = time.time() - t0
                    self.last_model = data.get("model", "unknown")
                    usage = data.get("usage", {})
                    print(f"[openrouter] done in {elapsed:.1f}s | model={self.last_model} | tokens={usage}", flush=True)
                    return data
            except TimeoutError:
                last_error = RuntimeError(
                    f"OpenRouter timed out after {timeout}s — try --engine mistral-ocr-4 or split the PDF"
                )
            except urllib.error.HTTPError as e:
                body = e.read().decode()
                last_error = RuntimeError(f"OpenRouter error {e.code}: {body}")
                # Retry on server errors (5xx) and rate limits (429)
                if e.code < 500 and e.code != 429:
                    raise last_error from e
            except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError, OSError) as e:
                last_error = e

            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"[openrouter] retry {attempt + 1}/{max_retries} in {wait}s: {last_error}", flush=True)
                time.sleep(wait)

        raise RuntimeError(f"OpenRouter failed after {max_retries} retries: {last_error}") from last_error

    def call(self, prompt: Prompt) -> str:
        data = self._request({"model": self.model, "messages": prompt.to_messages()})
        return data["choices"][0]["message"]["content"]

    def call_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Send messages with tool definitions; return the raw assistant message dict."""
        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._request(payload)
        if "choices" not in data:
            raise RuntimeError(f"Unexpected API response (no choices): {data}")
        return data["choices"][0]["message"]

    def chat(self, user_msg: str, system: Optional[str] = None) -> str:
        return self.call(Prompt(user=user_msg, system=system))

    def extract_pdf_openrouter(self, path: str, engine: str = "mistral-ocr") -> str:
        """Send a PDF through OpenRouter's file-parser plugin and return extracted text."""
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"[ocr] reading {os.path.basename(path)} ({size_mb:.1f} MB), engine={engine}", flush=True)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        print(f"[ocr] base64 encoded ({len(b64)//1024} KB), uploading to OpenRouter...", flush=True)
        payload = {
            "model": self.model,
            "plugins": [{"id": "file-parser", "pdf": {"engine": engine}}],
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": os.path.basename(path),
                            "file_data": f"data:application/pdf;base64,{b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Output ALL text content extracted from this document. "
                            "Preserve headings and paragraph breaks. "
                            "Output only the document text, nothing else."
                        ),
                    },
                ],
            }],
        }
        data = self._request(payload, timeout=300)
        text = data["choices"][0]["message"]["content"]
        print(f"[ocr] extracted {len(text)} chars", flush=True)
        return text

    def extract_pdf_mistral_ocr4(self, path: str) -> str:
        """Call Mistral OCR 4 directly (requires MISTRAL_API_KEY). Returns markdown."""
        mistral_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_key:
            raise ValueError("MISTRAL_API_KEY not set — required for Mistral OCR 4")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_base64",
                "document_base64": b64,
                "document_name": os.path.basename(path),
            },
            "include_image_base64": False,
        }
        req = urllib.request.Request(
            "https://api.mistral.ai/v1/ocr",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {mistral_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"Mistral OCR error {e.code}: {body}") from e
        pages = data.get("pages", [])
        print(f"[mistral-ocr-4] extracted {len(pages)} page(s)")
        return "\n\n".join(p.get("markdown", "") for p in pages)

    # ── reranking ────────────────────────────────────────────────────────────

    def rerank(self, query: str, documents: list[str], top_n: int = 32,
               model: str = RERANK_MODEL) -> list[str]:
        """Re-rank *documents* by relevance to *query* using Cohere Rerank.
        Defaults to Rerank 4 Pro; pass RERANK_3_MODEL for the cheaper v3.5.
        Returns *documents* reordered, trimmed to *top_n*."""
        if not documents:
            return []
        t0 = time.time()
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        req = urllib.request.Request(
            OPENROUTER_RERANK_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                break
            except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError, OSError) as e:
                if attempt == 3:
                    raise RuntimeError(f"Rerank failed after 3 retries: {e}") from e
                time.sleep(2 ** attempt)
        results = data.get("results", [])
        ordered = [documents[r["index"]] for r in sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)]
        elapsed = time.time() - t0
        label = model.split("/")[-1] if "/" in model else model
        print(f"[rerank] {len(documents)} → {len(ordered)} docs in {elapsed:.1f}s ({label})", flush=True)
        return ordered

    # ── embeddings ───────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Return a single embedding vector via text-embedding-3-small."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 50, workers: int = 1) -> list[list[float]]:
        """Return embedding vectors for *texts*, batching to respect API limits.
        When *workers* > 1, independent batches are requested concurrently."""
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        n_batches = len(batches)

        def fetch(indexed_batch: tuple[int, list[str]]) -> tuple[int, list[list[float]]]:
            i, batch = indexed_batch
            t0 = time.time()
            print(f"[embed] batch {i + 1}/{n_batches} ({len(batch)} texts)...", flush=True)
            payload = {
                "model": EMBED_MODEL,
                "input": batch,
            }
            req = urllib.request.Request(
                OPENROUTER_EMBED_URL,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = json.loads(resp.read())
                    break
                except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError, OSError) as e:
                    if attempt == 3:
                        raise RuntimeError(f"Embedding failed after 3 retries: {e}") from e
                    time.sleep(2 ** attempt)
            vectors = [item["embedding"] for item in data["data"]]
            elapsed = time.time() - t0
            print(f"[embed]   batch {i + 1}/{n_batches} done in {elapsed:.1f}s ({len(batch)} vectors)", flush=True)
            return i, vectors

        results: dict[int, list[list[float]]] = {}
        if workers > 1 and n_batches > 1:
            with ThreadPoolExecutor(max_workers=min(workers, n_batches)) as pool:
                for i, vectors in pool.map(fetch, enumerate(batches)):
                    results[i] = vectors
        else:
            for i, batch in enumerate(batches):
                _, vectors = fetch((i, batch))
                results[i] = vectors

        all_vectors: list[list[float]] = []
        for i in range(n_batches):
            all_vectors.extend(results[i])
        return all_vectors
