"""Cue-Tag-Content associative memory graph, persisted to SQLite."""

import os
import sqlite3
import threading
from typing import Optional


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

# SQLite FTS5 for ranked full-text search; kept as a separate virtual table
# so we preserve the simpler LIKE-based exact/substring searches too.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    text,
    content=nodes,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS nodes_fts_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO nodes_fts(rowid, text) VALUES (new.id, new.text);
END;
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

        # migrate from legacy memory.json if it exists and db is brand-new
        if not db_exists:
            self._migrate_json()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── write ────────────────────────────────────────────────────────────────

    def add_memory(
        self, content: str, cues: list[str], tags: list[str], docname: Optional[str] = None
    ) -> str:
        """Store content, linking it via tags to cue keywords. Returns content node id."""
        with self._lock:
            c = self._conn.cursor()

            # insert content node (docname only on content nodes — cues/tags are shared)
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
        """BM25 over a virtual document = chunk_text + weighted cues + tags, return top-N."""
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
        docs: list[tuple[str, list[str]]] = []
        for _cid, text, cue_texts, tag_texts in rows:
            tokens = tokenize(text)
            tokens += tokenize(cue_texts or "") * 5
            tokens += tokenize(tag_texts or "") * 3
            # append stems of every token so morphological variants match
            tokens += [_stem(tok) for tok in tokens]
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

    def all_cues(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT text FROM nodes WHERE type = 'cue'").fetchall()
            return [r[0] for r in rows]

    def all_tags(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT text FROM nodes WHERE type = 'tag'").fetchall()
            return [r[0] for r in rows]

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

    # ── compatibility shim — no-op _load / _save, replaced by SQLite ─────────

    def _load(self):
        pass  # SQLite handles persistence

    def _save(self):
        pass  # committed per-write in add_memory
