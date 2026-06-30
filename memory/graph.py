"""Cue-Tag-Content associative memory graph, persisted to JSON."""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    id: str
    type: str   # "cue" | "tag" | "content"
    text: str
    edges: list[str] = field(default_factory=list)


class MemoryGraph:
    def __init__(self, path: str = "memory.json"):
        self.path = path
        self.nodes: dict[str, Node] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._load()

    # ── write ────────────────────────────────────────────────────────────────

    def add_memory(self, content: str, cues: list[str], tags: list[str]) -> str:
        """Store content, linking it via tags to cue keywords. Returns content node id."""
        with self._lock:
            cid = self._new_id("content")
            content_node = Node(id=cid, type="content", text=content)

            tag_ids = []
            for tag in tags:
                tag_node = self._find("tag", tag) or self._create("tag", tag)
                if cid not in tag_node.edges:
                    tag_node.edges.append(cid)
                if tag_node.id not in content_node.edges:
                    content_node.edges.append(tag_node.id)
                tag_ids.append(tag_node.id)

            for cue in cues:
                cue_node = self._find("cue", cue) or self._create("cue", cue)
                for tid in tag_ids:
                    if tid not in cue_node.edges:
                        cue_node.edges.append(tid)

            self.nodes[cid] = content_node
            self._save()
            return cid

    # ── read ─────────────────────────────────────────────────────────────────

    def search_cue(self, query: str) -> list[dict]:
        """Walk cue → tag → content for any cue matching query substring."""
        q = query.lower()
        seen: set[str] = set()
        results = []
        for node in self.nodes.values():
            if node.type != "cue" or q not in node.text.lower():
                continue
            for tid in node.edges:
                tag = self.nodes.get(tid)
                if not tag:
                    continue
                for cid in tag.edges:
                    content = self.nodes.get(cid)
                    if content and content.type == "content" and cid not in seen:
                        seen.add(cid)
                        results.append({"cue": node.text, "tag": tag.text, "content": content.text})
        return results

    def get_by_tag(self, query: str) -> list[dict]:
        """Return contents linked to any tag matching query substring."""
        q = query.lower()
        seen: set[str] = set()
        results = []
        for node in self.nodes.values():
            if node.type != "tag" or q not in node.text.lower():
                continue
            for cid in node.edges:
                content = self.nodes.get(cid)
                if content and content.type == "content" and cid not in seen:
                    seen.add(cid)
                    results.append({"tag": node.text, "content": content.text})
        return results

    def rank_chunks(self, query: str, top_n: int = 8) -> list[str]:
        """BM25 over a virtual document = chunk_text + weighted cues + tags, return top-N."""
        import re
        import math

        raw_terms = re.sub(r"[^\w\s]", " ", query.lower()).split()
        terms = [t for t in raw_terms if len(t) > 2]
        if not terms:
            return []

        # BM25 parameters
        k1, b = 1.5, 0.75

        # pre-index: content_id → cue/tag texts
        cue_map: dict[str, list[str]] = {}
        tag_map: dict[str, list[str]] = {}
        for node in self.nodes.values():
            if node.type == "cue":
                for tid in node.edges:
                    tag = self.nodes.get(tid)
                    if tag and tag.type == "tag":
                        for cid in tag.edges:
                            cue_map.setdefault(cid, []).append(node.text.lower())
            elif node.type == "tag":
                for cid in node.edges:
                    tag_map.setdefault(cid, []).append(node.text.lower())

        # build virtual docs: chunk text + repeated cues (5×) + repeated tags (3×)
        content_nodes = [n for n in self.nodes.values() if n.type == "content"]
        N = len(content_nodes)
        if N == 0:
            return []

        def tokenize(text: str) -> list[str]:
            return re.sub(r"[^\w\s]", " ", text.lower()).split()

        docs: list[tuple[str, list[str]]] = []
        for node in content_nodes:
            cue_tokens = tokenize(" ".join(cue_map.get(node.id, []))) * 5
            tag_tokens = tokenize(" ".join(tag_map.get(node.id, []))) * 3
            tokens = tokenize(node.text) + cue_tokens + tag_tokens
            docs.append((node.text, tokens))

        avgdl = sum(len(tokens) for _, tokens in docs) / N

        # IDF per query term
        def idf(term: str) -> float:
            df = sum(1 for _, tokens in docs if term in tokens)
            return math.log((N - df + 0.5) / (df + 0.5) + 1)

        idfs = {t: idf(t) for t in terms}

        scored = []
        for text, tokens in docs:
            dl = len(tokens)
            tf_map: dict[str, int] = {}
            for tok in tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            score = sum(
                idfs[t] * (tf_map.get(t, 0) * (k1 + 1))
                / (tf_map.get(t, 0) + k1 * (1 - b + b * dl / avgdl))
                for t in terms
            )
            if score > 0:
                scored.append((score, text))

        scored.sort(reverse=True)
        return [text for _, text in scored[:top_n]]

    def all_cues(self) -> list[str]:
        return [n.text for n in self.nodes.values() if n.type == "cue"]

    def all_tags(self) -> list[str]:
        return [n.text for n in self.nodes.values() if n.type == "tag"]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _new_id(self, type: str) -> str:
        self._counter += 1
        return f"{type}_{self._counter}"

    def _find(self, type: str, text: str) -> Optional[Node]:
        for node in self.nodes.values():
            if node.type == type and node.text.lower() == text.lower():
                return node
        return None

    def _create(self, type: str, text: str) -> Node:
        nid = self._new_id(type)
        node = Node(id=nid, type=type, text=text)
        self.nodes[nid] = node
        return node

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            data = json.load(f)
        self._counter = data.get("counter", 0)
        for nid, nd in data.get("nodes", {}).items():
            self.nodes[nid] = Node(**nd)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(
                {"counter": self._counter, "nodes": {nid: vars(n) for nid, n in self.nodes.items()}},
                f, indent=2,
            )
