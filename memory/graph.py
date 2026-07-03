"""Cue-Tag-Content associative memory graph, persisted to SQLite."""

import math
import os
import sqlite3
import struct
import threading
import time
from typing import Optional

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    type    TEXT    NOT NULL CHECK(type IN ('cue', 'tag', 'content')),
    text    TEXT    NOT NULL,
    docname TEXT    DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source_id  INTEGER NOT NULL,
    target_id  INTEGER NOT NULL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nodes_type     ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_docname  ON nodes(docname);
CREATE INDEX IF NOT EXISTS idx_edges_source   ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target   ON edges(target_id);
"""

# SQLite FTS5 for ranked full-text search on *content* nodes.
# The virtual document = chunk text + cues + tags.
# FTS5 BM25 (rank) naturally boosts documents matching multiple fields.
# NB: standalone FTS5 (no content-sync) — we populate it manually.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
    text
);
"""


class MemoryGraph:
    def __init__(self, path: str = "memory.db"):
        self.path = path
        self._lock = threading.Lock()

        db_exists = os.path.exists(path)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_FTS_SCHEMA)
        self._conn.commit()

        # migrate: add docname column if upgrading from older schema
        self._migrate_docname()

        # migrate: add embedding column if upgrading
        self._migrate_embedding()

        # migrate: upgrade to content_fts (replaces old nodes_fts)
        self._migrate_content_fts()

        # migrate from legacy memory.json if it exists and db is brand-new
        if not db_exists:
            self._migrate_json()

        # FAISS vector index — persists to memory.faiss alongside the DB
        self._faiss_path = os.path.splitext(path)[0] + ".faiss"
        self._faiss_index: "faiss.Index | None" = None
        self._faiss_ids: list[int] = []  # node id for each FAISS position
        self._faiss_d: int | None = None
        self._faiss_ivfpq: bool = False  # True when using IVFPQ (needs retrain)
        if _FAISS_AVAILABLE:
            self._build_faiss()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        # persist FAISS index
        if _FAISS_AVAILABLE and self._faiss_index is not None:
            self._save_faiss()

    # ── write ────────────────────────────────────────────────────────────────

    def add_memory(
        self, content: str, cues: list[str], tags: list[str],
        docname: Optional[str] = None, embedding: Optional[bytes] = None,
    ) -> str:
        """Store content, linking it via tags to cue keywords. Returns content node id."""
        with self._lock:
            c = self._conn.cursor()

            # insert content node (docname only on content nodes — cues/tags are shared)
            if embedding is not None:
                c.execute(
                    "INSERT INTO nodes (type, text, docname, embedding) VALUES ('content', ?, ?, ?)",
                    (content, docname, embedding),
                )
            else:
                c.execute(
                    "INSERT INTO nodes (type, text, docname) VALUES ('content', ?, ?)",
                    (content, docname),
                )
            cid = c.lastrowid

            tag_ids: list[int] = []
            for tag in tags:
                tag_id = self._find_or_create(c, "tag", tag)
                tag_ids.append(tag_id)
                # tag → content (forward edge)
                self._add_edge(c, tag_id, cid)
                # content → tag (backlink)
                self._add_edge(c, cid, tag_id)

            for cue in cues:
                cue_id = self._find_or_create(c, "cue", cue)
                for tid in tag_ids:
                    # cue → tag
                    self._add_edge(c, cue_id, tid)

            self._conn.commit()

            # populate content_fts for BM25 retrieval
            self._fts_upsert(cid)

            # add to FAISS index if embedding was provided
            if embedding is not None:
                self._faiss_add_vector(cid, embedding)

            return f"c{cid}"

    # ── forget ───────────────────────────────────────────────────────────────

    def forget_doc(self, docname: str) -> int:
        """Delete all content nodes belonging to *docname*.

        Cascading foreign keys remove all edges incident on those nodes.
        Afterwards, orphaned tags (no remaining content edges) and orphaned
        cues (no remaining edges at all) are pruned.
        Returns the number of content nodes removed.
        """
        with self._lock:
            c = self._conn.cursor()

            # count before deleting
            row = c.execute(
                "SELECT COUNT(*) FROM nodes WHERE type = 'content' AND docname = ?",
                (docname,),
            ).fetchone()
            count = row[0] if row else 0

            # 1. delete content nodes (FOREIGN KEY CASCADE removes incident edges)
            c.execute(
                "DELETE FROM nodes WHERE type = 'content' AND docname = ?",
                (docname,),
            )

            # 2. prune tags that no longer point to any content
            c.execute(
                """
                DELETE FROM nodes
                WHERE type = 'tag'
                  AND id NOT IN (
                      SELECT DISTINCT e.source_id
                      FROM edges e
                      JOIN nodes c ON e.target_id = c.id AND c.type = 'content'
                  )
                """
            )

            # 3. prune cues with zero remaining edges (orphaned after step 2)
            c.execute(
                """
                DELETE FROM nodes
                WHERE type = 'cue'
                  AND id NOT IN (SELECT source_id FROM edges)
                  AND id NOT IN (SELECT target_id FROM edges)
                """
            )

            self._conn.commit()
            if count:
                print(f"[forget] removed {count} chunk(s) from '{docname}'")
            else:
                print(f"[forget] no chunks found for doc '{docname}'")
            return count

    def forget_content(self, node_id: int) -> int:
        """Delete a single content node by its integer id.

        Cascading foreign keys remove incident edges. Orphaned tags and
        cues are pruned. Returns 1 on success, 0 if not found.
        """
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "DELETE FROM nodes WHERE type = 'content' AND id = ?",
                (node_id,),
            )
            deleted = c.rowcount

            if deleted:
                # prune orphan tags (no content connections left)
                c.execute(
                    """
                    DELETE FROM nodes
                    WHERE type = 'tag'
                      AND id NOT IN (
                          SELECT DISTINCT e.source_id
                          FROM edges e
                          JOIN nodes c ON e.target_id = c.id AND c.type = 'content'
                      )
                    """
                )
                # prune orphan cues (zero edges left after tag pruning)
                c.execute(
                    """
                    DELETE FROM nodes
                    WHERE type = 'cue'
                      AND id NOT IN (SELECT source_id FROM edges)
                      AND id NOT IN (SELECT target_id FROM edges)
                    """
                )

            self._conn.commit()
            return deleted

    # ── read ─────────────────────────────────────────────────────────────────

    def search_cue(self, query: str) -> list[dict]:
        """Walk cue → tag → content for any cue matching query substring."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT cu.text AS cue_text, t.text AS tag_text, c.text AS content_text
                FROM nodes cu
                JOIN edges  cue_tag ON cu.id = cue_tag.source_id
                JOIN nodes t        ON cue_tag.target_id = t.id AND t.type = 'tag'
                JOIN edges  tag_con ON t.id = tag_con.source_id
                JOIN nodes c        ON tag_con.target_id = c.id AND c.type = 'content'
                WHERE cu.type = 'cue' AND cu.text LIKE ?
                """,
                (f"%{query}%",),
            ).fetchall()
            return [{"cue": r[0], "tag": r[1], "content": r[2]} for r in rows]

    def get_by_tag(self, query: str) -> list[dict]:
        """Return contents linked to any tag matching query substring."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT t.text AS tag_text, c.text AS content_text
                FROM nodes t
                JOIN edges  tag_con ON t.id = tag_con.source_id
                JOIN nodes c        ON tag_con.target_id = c.id AND c.type = 'content'
                WHERE t.type = 'tag' AND t.text LIKE ?
                """,
                (f"%{query}%",),
            ).fetchall()
            return [{"tag": r[0], "content": r[1]} for r in rows]

    def rank_chunks(self, query: str, top_n: int = 8) -> list[str]:
        """Return top-N content texts ranked by FTS5 BM25.

        Uses the persisted ``content_fts`` index — no per-query tokenization.
        Falls back to in-memory BM25 if FTS5 is unavailable.
        """
        # Try FTS5 first (fast, persisted)
        try:
            fts_query = self._fts_query(query)
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT n.text
                    FROM content_fts f
                    JOIN nodes n ON f.rowid = n.id
                    WHERE content_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, top_n),
                ).fetchall()
            if rows:
                return [r[0] for r in rows]
        except Exception as e:
            print(f"[fts] search failed: {e} — falling back to in-memory BM25", flush=True)

        # In-memory BM25 fallback (kept for robustness)
        return self._rank_chunks_bm25(query, top_n)

    @staticmethod
    def _fts_query(user_query: str) -> str:
        """Convert a user query to an FTS5-compatible MATCH string.

        Each word becomes a prefix term, OR-ed together, so partial matches work.
        Example: 'maxwell equations' → 'maxwell* OR equations*'
        """
        import re
        tokens = re.findall(r"\w+", user_query.lower())
        if not tokens:
            tokens = [user_query.lower()]
        # FTS5 prefix queries + OR
        return " OR ".join(f'"{t}"*' for t in tokens)

    def _rank_chunks_bm25(self, query: str, top_n: int = 8) -> list[str]:
        """In-memory BM25 — original implementation kept as fallback."""
        import re
        import math
        import unicodedata
        import snowballstemmer

        def _fold(word: str) -> str:
            """Strip diacritics via NFKD: schrödinger → schrodinger."""
            return unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode()

        raw_terms = re.sub(r"[^\w\s]", " ", _fold(query.lower())).split()
        terms = [t for t in raw_terms if len(t) > 2]
        if not terms:
            return []

        # Snowball stemmers — Italian handles verb/noun variation (consumato ↔ consumo);
        # English as fallback so non-Italian tokens pass through.
        _stem_it = snowballstemmer.stemmer("italian")
        _stem_en = snowballstemmer.stemmer("english")

        def _stem(word: str) -> str:
            """Stem a word: Italian first, fall back to English."""
            s = _stem_it.stemWord(word)
            if s != word.lower():
                return s
            return _stem_en.stemWord(word)

        # Expand query terms with their stems
        stem_terms = [_stem(t) for t in terms]

        k1, b = 1.5, 0.75

        with self._lock:
            # fetch all content nodes with their cues and tags via scalar subqueries
            rows = self._conn.execute(
                """
                SELECT c.id, c.text,
                       (SELECT GROUP_CONCAT(cu.text, ' ')
                        FROM edges c2t
                        JOIN nodes t ON c2t.target_id = t.id AND t.type = 'tag'
                        JOIN edges cu2t ON t.id = cu2t.target_id
                        JOIN nodes cu ON cu2t.source_id = cu.id AND cu.type = 'cue'
                        WHERE c2t.source_id = c.id) AS cue_texts,
                       (SELECT GROUP_CONCAT(t.text, ' ')
                        FROM edges c2t
                        JOIN nodes t ON c2t.target_id = t.id AND t.type = 'tag'
                        WHERE c2t.source_id = c.id) AS tag_texts
                FROM nodes c
                WHERE c.type = 'content'
                """
            ).fetchall()

        if not rows:
            return []

        N = len(rows)

        def tokenize(text: str) -> list[str]:
            return re.sub(r"[^\w\s]", " ", _fold((text or "").lower())).split()

        # build virtual docs: text + cues repeated 5× + tags repeated 3×
        # Use stem cache to avoid re-stemming identical tokens millions of times.
        _stem_cache: dict[str, str] = {}
        def _stem_cached(word: str) -> str:
            s = _stem_cache.get(word)
            if s is None:
                s = _stem(word)
                _stem_cache[word] = s
            return s

        docs: list[tuple[str, list[str]]] = []
        for _cid, text, cue_texts, tag_texts in rows:
            tokens = tokenize(text)
            tokens += tokenize(cue_texts or "") * 5
            tokens += tokenize(tag_texts or "") * 3
            # append stems of every token so morphological variants match
            tokens += [_stem_cached(tok) for tok in tokens]
            docs.append((text, tokens))

        avgdl = sum(len(tokens) for _, tokens in docs) / N

        def idf(term: str) -> float:
            df = sum(1 for _, tokens in docs if term in tokens)
            return math.log((N - df + 0.5) / (df + 0.5) + 1)

        idfs = {t: idf(t) for t in stem_terms}

        scored = []
        for text, tokens in docs:
            dl = len(tokens)
            tf_map: dict[str, int] = {}
            for tok in tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            score = sum(
                idfs[t] * (tf_map.get(t, 0) * (k1 + 1))
                / (tf_map.get(t, 0) + k1 * (1 - b + b * dl / avgdl))
                for t in stem_terms
            )
            if score > 0:
                scored.append((score, text))

        scored.sort(reverse=True)
        return [text for _, text in scored[:top_n]]

    def expand_query(self, query: str, max_terms: int = 12) -> str:
        """Expand *query* with related cues found in the graph.

        Finds content nodes matching the query, then collects their associated
        cues (via shared tags) ranked by co-occurrence frequency.  Returns the
        original query plus the top expansion terms that are not already in it.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM nodes WHERE type = 'content' AND text LIKE ?",
                (f"%{query}%",),
            ).fetchall()
            if not rows:
                return query

            placeholders = ",".join("?" * len(rows))
            cue_rows = self._conn.execute(
                f"""
                SELECT cu.text, COUNT(DISTINCT c.id) AS cnt
                FROM nodes c
                JOIN edges c2t ON c.id = c2t.source_id
                JOIN nodes t  ON c2t.target_id = t.id AND t.type = 'tag'
                JOIN edges cu2t ON t.id = cu2t.target_id
                JOIN nodes cu ON cu2t.source_id = cu.id AND cu.type = 'cue'
                WHERE c.id IN ({placeholders})
                GROUP BY cu.text
                ORDER BY cnt DESC
                LIMIT ?
                """,
                [r[0] for r in rows] + [max_terms],
            ).fetchall()

        query_lower = query.lower()
        expansion = [r[0] for r in cue_rows if r[0].lower() not in query_lower]
        if expansion:
            return query + " " + " ".join(expansion)
        return query

    def all_cues(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT text FROM nodes WHERE type = 'cue'").fetchall()
            return [r[0] for r in rows]

    def all_tags(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT text FROM nodes WHERE type = 'tag'").fetchall()
            return [r[0] for r in rows]

    def chunk_count(self, docname: str) -> int:
        """Return how many content chunks exist for a given docname."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type = 'content' AND docname = ?",
                (docname,),
            ).fetchone()
            return row[0] if row else 0

    def list_docs(self) -> list[dict]:
        """Return document summaries: docname → chunk count, total chars."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT docname,
                       COUNT(*)        AS chunks,
                       SUM(LENGTH(text)) AS total_chars
                FROM nodes
                WHERE type = 'content' AND docname IS NOT NULL
                GROUP BY docname
                ORDER BY docname
                """
            ).fetchall()
            return [
                {"docname": r[0], "chunks": r[1], "total_chars": r[2]}
                for r in rows
            ]

    def content_nodes(self) -> list[dict]:
        """Return all content nodes with ids (for memory display)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, docname FROM nodes WHERE type = 'content' ORDER BY id"
            ).fetchall()
            return [
                {"id": f"c{r[0]}", "text": r[1], "docname": r[2]}
                for r in rows
            ]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_or_create(self, cursor, ntype: str, text: str) -> int:
        """Return existing node id or insert a new one (case-insensitive).
        Cue and tag nodes are shared across documents — no docname is set.
        """
        row = cursor.execute(
            "SELECT id FROM nodes WHERE type = ? AND LOWER(text) = LOWER(?)",
            (ntype, text),
        ).fetchone()
        if row:
            return row[0]
        cursor.execute(
            "INSERT INTO nodes (type, text) VALUES (?, ?)", (ntype, text)
        )
        return cursor.lastrowid

    @staticmethod
    def _add_edge(cursor, source_id: int, target_id: int) -> None:
        cursor.execute(
            "INSERT OR IGNORE INTO edges (source_id, target_id) VALUES (?, ?)",
            (source_id, target_id),
        )

    # ── schema migration ─────────────────────────────────────────────────────

    def _migrate_docname(self) -> None:
        """Add the docname column if it doesn't exist (upgrade from older schema)."""
        try:
            self._conn.execute("SELECT docname FROM nodes LIMIT 0")
        except sqlite3.OperationalError:
            with self._lock:
                self._conn.execute("ALTER TABLE nodes ADD COLUMN docname TEXT DEFAULT NULL")
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_nodes_docname ON nodes(docname)"
                )
                self._conn.commit()
            print("[migrate] added docname column to nodes table")

    def _migrate_json(self) -> None:
        """One-shot: import an old memory.json into SQLite."""
        import json as _json

        json_path = os.path.join(os.path.dirname(self.path), "memory.json")
        if not os.path.exists(json_path):
            return

        with open(json_path) as f:
            data = _json.load(f)

        nodes: dict[str, dict] = data.get("nodes", {})
        if not nodes:
            return

        # map old string ids (content_1, tag_2, …) to new integer ids
        id_map: dict[str, int] = {}

        with self._lock:
            c = self._conn.cursor()
            for old_id, nd in nodes.items():
                c.execute(
                    "INSERT INTO nodes (type, text) VALUES (?, ?)",
                    (nd["type"], nd["text"]),
                )
                id_map[old_id] = c.lastrowid

            for old_id, nd in nodes.items():
                src = id_map[old_id]
                for target_old in nd.get("edges", []):
                    tgt = id_map.get(target_old)
                    if tgt is not None:
                        self._add_edge(c, src, tgt)

            self._conn.commit()
            total = len(nodes)
            print(f"[migrate] imported {total} node(s) from memory.json into {self.path}")

    def _migrate_embedding(self) -> None:
        """Add the embedding column if it doesn't exist."""
        try:
            self._conn.execute("SELECT embedding FROM nodes LIMIT 0")
        except sqlite3.OperationalError:
            with self._lock:
                self._conn.execute("ALTER TABLE nodes ADD COLUMN embedding BLOB DEFAULT NULL")
                self._conn.commit()
            print("[migrate] added embedding column to nodes table")

    def _migrate_content_fts(self) -> None:
        """Ensure content_fts exists and is populated."""
        # Check if we need to rebuild (old content-sync table or missing)
        needs_rebuild = False
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0]
            if count == 0:
                needs_rebuild = True
        except sqlite3.OperationalError:
            # content_fts doesn't exist — drop old nodes_fts and create new one
            with self._lock:
                self._conn.execute("DROP TABLE IF EXISTS nodes_fts")
                self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(text)")
                self._conn.commit()
            needs_rebuild = True
            print("[migrate] created content_fts (standalone FTS5)")

        if needs_rebuild:
            self._fts_rebuild()

    def _fts_virtual_doc(self, content_id: int) -> str:
        """Build the FTS5 virtual document for a content node: text + cues + tags."""
        with self._lock:
            # get the content text
            row = self._conn.execute(
                "SELECT text FROM nodes WHERE id = ?", (content_id,)
            ).fetchone()
            if not row:
                return ""
            parts = [row[0]]
            # get cues via tags
            cue_rows = self._conn.execute(
                """
                SELECT DISTINCT cu.text
                FROM edges c2t
                JOIN nodes t  ON c2t.target_id = t.id AND t.type = 'tag'
                JOIN edges cu2t ON t.id = cu2t.target_id
                JOIN nodes cu ON cu2t.source_id = cu.id AND cu.type = 'cue'
                WHERE c2t.source_id = ?
                """,
                (content_id,),
            ).fetchall()
            for (cue,) in cue_rows:
                parts.append(cue)
            # get tags
            tag_rows = self._conn.execute(
                """
                SELECT DISTINCT t.text
                FROM edges c2t
                JOIN nodes t ON c2t.target_id = t.id AND t.type = 'tag'
                WHERE c2t.source_id = ?
                """,
                (content_id,),
            ).fetchall()
            for (tag,) in tag_rows:
                parts.append(tag)
            return " ".join(parts)

    def _fts_upsert(self, content_id: int) -> None:
        """Insert or update the FTS5 entry for a content node."""
        doc = self._fts_virtual_doc(content_id)
        if not doc:
            return
        with self._lock:
            # Check if row exists (FTS5 errors on delete of missing rowid)
            exists = self._conn.execute(
                "SELECT COUNT(*) FROM content_fts WHERE rowid = ?",
                (content_id,),
            ).fetchone()[0] > 0
            if exists:
                self._conn.execute(
                    "INSERT INTO content_fts(content_fts, rowid, text) VALUES ('delete', ?, '')",
                    (content_id,),
                )
            self._conn.execute(
                "INSERT INTO content_fts(rowid, text) VALUES (?, ?)",
                (content_id, doc),
            )
            self._conn.commit()

    def _fts_rebuild(self) -> None:
        """Rebuild the entire content_fts index from all content nodes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM nodes WHERE type = 'content' ORDER BY id"
            ).fetchall()
        count = 0
        for (nid,) in rows:
            self._fts_upsert(nid)
            count += 1
        print(f"[fts] rebuilt content_fts: {count} documents", flush=True)

    def set_embedding(self, node_id: str, vector: list[float]) -> None:
        """Store an embedding vector for a content node (as float32 BLOB)."""
        import struct
        nid = int(node_id.lstrip("c"))
        blob = struct.pack(f"{len(vector)}f", *vector)
        with self._lock:
            self._conn.execute(
                "UPDATE nodes SET embedding = ? WHERE id = ?",
                (blob, nid),
            )
            self._conn.commit()

    def _build_faiss(self) -> None:
        """Build or load FAISS index.

        1. Try to load ``memory.faiss`` from disk (fast).
        2. If missing or fewer vectors than in SQLite, rebuild from SQLite.
        3. Choose IndexFlatIP (< 10K vectors) or IndexIVFPQ (≥ 10K).
        4. Persist to ``memory.faiss``.
        """
        if not _FAISS_AVAILABLE:
            return

        # ── try loading from disk ───────────────────────────────────────────
        if os.path.exists(self._faiss_path):
            try:
                import numpy as np
                idx = faiss.read_index(self._faiss_path)
                n_total = idx.ntotal
                # Also load the id mapping
                id_path = self._faiss_path + ".ids.npy"
                if os.path.exists(id_path):
                    self._faiss_ids = [int(x) for x in np.load(id_path)]
                else:
                    self._faiss_ids = []
                self._faiss_d = idx.d
                self._faiss_index = idx
                self._faiss_ivfpq = not isinstance(idx, faiss.IndexFlatIP)
                kind = "IVFPQ" if self._faiss_ivfpq else "FlatIP"
                print(f"[faiss] loaded {kind} index from disk: {n_total} vectors, dim={idx.d}", flush=True)
                # Check if DB has more embeddings than index → rebuild
                with self._lock:
                    db_count = self._conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE type = 'content' AND embedding IS NOT NULL"
                    ).fetchone()[0]
                if db_count <= n_total:
                    return
                print(f"[faiss] DB has {db_count} vectors but index has {n_total} — rebuilding", flush=True)
            except Exception as e:
                print(f"[faiss] failed to load from disk: {e} — rebuilding", flush=True)

        # ── build from SQLite ───────────────────────────────────────────────
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM nodes WHERE type = 'content' AND embedding IS NOT NULL ORDER BY id"
            ).fetchall()

        if not rows:
            self._faiss_index = None
            self._faiss_ids = []
            return

        n = len(rows)
        d = len(rows[0][1]) // 4
        self._faiss_d = d

        import numpy as np
        vectors = np.empty((n, d), dtype=np.float32)
        self._faiss_ids = []
        for i, (nid, blob) in enumerate(rows):
            vec = list(struct.unpack(f"{d}f", blob))
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vectors[i] = [v / norm for v in vec]
            else:
                vectors[i] = list(vec)
            self._faiss_ids.append(nid)

        # ── choose index type ──────────────────────────────────────────────
        IVF_THRESHOLD = 10_000

        if n >= IVF_THRESHOLD:
            # IndexIVFPQ: trained IVF + product quantization
            nlist = max(64, int(4 * math.sqrt(n)))       # ~9K for 5M
            m_sub = min(64, d // 2)                       # sub-vectors for PQ
            bits = 8
            quantizer = faiss.IndexFlatIP(d)
            idx = faiss.IndexIVFPQ(quantizer, d, nlist, m_sub, bits)
            idx.own_fields = True
            idx.metric_type = faiss.METRIC_INNER_PRODUCT

            # Train on a sample (IVFPQ needs ~30 × nlist × 256 vectors minimum;
            # we cap at 100K for speed)
            train_n = min(n, 100_000)
            print(f"[faiss] training IVFPQ on {train_n} vectors ({nlist} clusters, PQ {m_sub}x{bits}bit)...",
                  flush=True)
            t0 = time.time()
            idx.train(vectors[:train_n])
            idx.add(vectors)
            elapsed = time.time() - t0
            self._faiss_ivfpq = True
            print(f"[faiss] IVFPQ index built: {n} vectors, dim={d}, "
                  f"nlist={nlist}, m={m_sub} — trained in {elapsed:.1f}s", flush=True)
        else:
            # IndexFlatIP: exact search, fast enough for small N
            idx = faiss.IndexFlatIP(d)
            idx.metric_type = faiss.METRIC_INNER_PRODUCT
            idx.add(vectors)
            self._faiss_ivfpq = False
            print(f"[faiss] FlatIP index built: {n} vectors, dim={d}", flush=True)

        self._faiss_index = idx
        self._save_faiss()

    def _save_faiss(self) -> None:
        """Persist FAISS index + id mapping to disk."""
        if self._faiss_index is None or not _FAISS_AVAILABLE:
            return
        import numpy as np
        faiss.write_index(self._faiss_index, self._faiss_path)
        np.save(self._faiss_path + ".ids", np.array(self._faiss_ids, dtype=np.int64), allow_pickle=False)

    def _faiss_add_vector(self, nid: int, blob: bytes) -> None:
        """Add a single L2-normalized vector to the FAISS index."""
        if self._faiss_index is None:
            return
        import numpy as np
        d = len(blob) // 4
        vec = list(struct.unpack(f"{d}f", blob))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        self._faiss_index.add(np.array([vec], dtype=np.float32))
        self._faiss_ids.append(nid)

    def rank_by_embedding(self, query_vector: list[float], top_n: int = 32) -> list[str]:
        """Return top-N content texts ranked by cosine similarity.
        Uses FAISS (IndexFlatIP on L2-normalized vectors) for O(n) BLAS-optimised search.
        Falls back to brute-force Python loop if FAISS is unavailable."""
        if not query_vector:
            return []

        # Normalize query
        q_norm = math.sqrt(sum(v * v for v in query_vector))
        if q_norm == 0:
            return []
        q = [v / q_norm for v in query_vector]

        # FAISS path
        if self._faiss_index is not None and self._faiss_ids:
            import numpy as np
            # For IVFPQ: set nprobe (number of clusters to search)
            if self._faiss_ivfpq:
                nprobe = max(16, min(256, int(self._faiss_index.nlist * 0.01)))
                self._faiss_index.nprobe = nprobe
            q_np = np.array([q], dtype=np.float32)
            k = min(top_n, self._faiss_index.ntotal)
            distances, indices = self._faiss_index.search(q_np, k)
            faiss_node_ids = [int(self._faiss_ids[i]) for i in indices[0] if i >= 0]
            if not faiss_node_ids:
                return []
            with self._lock:
                placeholders = ",".join("?" * len(faiss_node_ids))
                rows = dict(self._conn.execute(
                    f"SELECT id, text FROM nodes WHERE id IN ({placeholders})",
                    faiss_node_ids,
                ).fetchall())
            # Preserve FAISS ranking order
            return [rows[nid] for nid in faiss_node_ids if nid in rows]

        # Brute-force fallback
        with self._lock:
            rows = self._conn.execute(
                "SELECT text, embedding FROM nodes WHERE type = 'content' AND embedding IS NOT NULL"
            ).fetchall()

        scored: list[tuple[float, str]] = []
        for text, blob in rows:
            if blob is None:
                continue
            n = len(blob) // 4
            vec = list(struct.unpack(f"{n}f", blob))
            dot = sum(a * b for a, b in zip(q, vec))
            v_norm = math.sqrt(sum(v * v for v in vec))
            if v_norm == 0:
                continue
            sim = dot / v_norm  # q already normalized
            scored.append((sim, text))

        scored.sort(reverse=True)
        return [text for _, text in scored[:top_n]]

    def chunk_has_embedding(self, node_id: str) -> bool:
        """Check if a content node already has an embedding."""
        nid = int(node_id.lstrip("c"))
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM nodes WHERE id = ?", (nid,)
            ).fetchone()
            return row is not None and row[0] is not None

    def chunks_without_embedding(self) -> list[str]:
        """Return node ids of content nodes missing embeddings."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM nodes WHERE type = 'content' AND embedding IS NULL"
            ).fetchall()
            return [f"c{r[0]}" for r in rows]

    # ── compatibility shim — no-op _load / _save, replaced by SQLite ─────────

    def _load(self):
        pass  # SQLite handles persistence

    def _save(self):
        pass  # committed per-write in add_memory
