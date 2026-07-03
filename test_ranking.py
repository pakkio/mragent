"""Unit tests for the retrieval/reranking stack: RRF fusion, FTS5 BM25,
in-memory BM25 fallback (stemming + diacritic folding), embedding ranking,
and query expansion. No API key required — everything runs locally."""

import pytest

from agent.agent import _rrf
from memory.graph import MemoryGraph


# ── RRF fusion ───────────────────────────────────────────────────────────────

class TestRRF:
    def test_item_in_both_lists_ranks_first(self):
        # "x" appears in both lists → its RRF score (1/61 + 1/62) beats
        # any single-list item, including "w" ranked first in list b (1/61).
        fused = _rrf(["x", "y", "z"], ["w", "x"])
        assert fused[0] == "x"

    def test_full_ordering(self):
        fused = _rrf(["x", "y", "z"], ["w", "x"])
        # x: 1/61+1/62 > w: 1/61 > y: 1/62 > z: 1/63
        assert fused == ["x", "w", "y", "z"]

    def test_single_list_preserves_order(self):
        assert _rrf(["a", "b", "c"], []) == ["a", "b", "c"]

    def test_empty_inputs(self):
        assert _rrf([], []) == []

    def test_no_duplicates_in_output(self):
        fused = _rrf(["a", "b"], ["b", "a"])
        assert sorted(fused) == ["a", "b"]


# ── graph fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def graph(tmp_path):
    g = MemoryGraph(str(tmp_path / "test.db"))
    yield g
    g.close()


def _seed_physics(graph):
    graph.add_memory(
        "Quantum entanglement links the states of two particles. "
        "Measuring one quantum system instantly constrains the other.",
        cues=["entanglement", "EPR"], tags=["quantum-mechanics"],
    )
    graph.add_memory(
        "The Schrödinger equation describes how the wave function evolves in time.",
        cues=["wave function", "Schrödinger"], tags=["quantum-mechanics"],
    )
    graph.add_memory(
        "General relativity models gravity as curvature of spacetime.",
        cues=["spacetime", "curvature"], tags=["relativity"],
    )


# ── _fts_query ───────────────────────────────────────────────────────────────

class TestFtsQuery:
    def test_multi_word_becomes_prefix_or(self):
        assert MemoryGraph._fts_query("maxwell equations") == '"maxwell"* OR "equations"*'

    def test_lowercases(self):
        assert MemoryGraph._fts_query("Maxwell") == '"maxwell"*'


# ── rank_chunks (FTS5 BM25 path) ─────────────────────────────────────────────

class TestRankChunksFTS:
    def test_returns_only_matching_chunks(self, graph):
        _seed_physics(graph)
        results = graph.rank_chunks("entanglement")
        assert len(results) >= 1
        assert "entanglement" in results[0]
        assert all("relativity" not in r for r in results)

    def test_prefix_matching(self, graph):
        _seed_physics(graph)
        # 'quant' should prefix-match 'quantum'
        results = graph.rank_chunks("quant")
        assert any("Quantum entanglement" in r for r in results)

    def test_diacritic_folding(self, graph):
        _seed_physics(graph)
        # FTS5 unicode61 folds ö → o: 'schrodinger' must match 'Schrödinger'
        results = graph.rank_chunks("schrodinger")
        assert any("Schrödinger" in r for r in results)

    def test_top_n_limit(self, graph):
        _seed_physics(graph)
        results = graph.rank_chunks("the", top_n=1)
        assert len(results) <= 1

    def test_no_match_returns_empty(self, graph):
        _seed_physics(graph)
        assert graph.rank_chunks("xylophone") == []


# ── _rank_chunks_bm25 (in-memory fallback) ───────────────────────────────────

class TestRankChunksBM25Fallback:
    def test_basic_match(self, graph):
        _seed_physics(graph)
        results = graph._rank_chunks_bm25("entanglement")
        assert len(results) >= 1
        assert "entanglement" in results[0]

    def test_italian_stemming(self, graph):
        # 'consumato' must match a chunk containing 'consumo' via Snowball stems
        graph.add_memory(
            "Il consumo di energia elettrica è aumentato del dieci percento.",
            cues=["energia"], tags=["economia"],
        )
        graph.add_memory(
            "La fotosintesi converte la luce in energia chimica.",
            cues=["fotosintesi"], tags=["biologia"],
        )
        results = graph._rank_chunks_bm25("consumato")
        assert len(results) >= 1
        assert "consumo" in results[0]

    def test_diacritic_folding(self, graph):
        _seed_physics(graph)
        results = graph._rank_chunks_bm25("schrodinger")
        assert any("Schrödinger" in r for r in results)

    def test_cue_match_outranks_single_text_mention(self, graph):
        # Cue tokens are weighted 5× in the virtual document, so a chunk whose
        # *cue* is 'thermodynamics' must outrank one that merely mentions it once.
        graph.add_memory(
            "The engine efficiency is bounded by the Carnot limit.",
            cues=["thermodynamics", "Carnot"], tags=["physics"],
        )
        graph.add_memory(
            "The word thermodynamics appears here once in passing text.",
            cues=["trivia"], tags=["misc"],
        )
        results = graph._rank_chunks_bm25("thermodynamics")
        assert len(results) == 2
        assert "Carnot" in results[0]

    def test_no_match_returns_empty(self, graph):
        _seed_physics(graph)
        assert graph._rank_chunks_bm25("xylophone") == []

    def test_short_terms_filtered(self, graph):
        _seed_physics(graph)
        # all terms ≤ 2 chars are dropped → no usable query terms → []
        assert graph._rank_chunks_bm25("of to") == []


# ── rank_by_embedding ────────────────────────────────────────────────────────

class TestRankByEmbedding:
    def _seed_vectors(self, graph):
        ids = [
            graph.add_memory("chunk aligned", cues=["a"], tags=["t"]),
            graph.add_memory("chunk diagonal", cues=["b"], tags=["t"]),
            graph.add_memory("chunk orthogonal", cues=["c"], tags=["t"]),
        ]
        graph.set_embedding(ids[0], [1.0, 0.0])
        graph.set_embedding(ids[1], [0.6, 0.8])
        graph.set_embedding(ids[2], [0.0, 1.0])

    def test_cosine_ordering(self, graph):
        self._seed_vectors(graph)
        results = graph.rank_by_embedding([1.0, 0.0])
        assert results[0] == "chunk aligned"      # sim 1.0
        assert results[1] == "chunk diagonal"     # sim 0.6
        assert results[2] == "chunk orthogonal"   # sim 0.0

    def test_top_n_limit(self, graph):
        self._seed_vectors(graph)
        assert len(graph.rank_by_embedding([1.0, 0.0], top_n=2)) == 2

    def test_zero_or_empty_query_vector(self, graph):
        self._seed_vectors(graph)
        assert graph.rank_by_embedding([0.0, 0.0]) == []
        assert graph.rank_by_embedding([]) == []


# ── expand_query ─────────────────────────────────────────────────────────────

class TestExpandQuery:
    def test_expands_with_related_cues(self, graph):
        graph.add_memory(
            "The mitochondria is the powerhouse of the cell.",
            cues=["ATP", "cristae"], tags=["biology"],
        )
        expanded = graph.expand_query("mitochondria")
        assert expanded.startswith("mitochondria")
        assert "ATP" in expanded
        assert "cristae" in expanded

    def test_does_not_duplicate_terms_already_in_query(self, graph):
        graph.add_memory(
            "The mitochondria is the powerhouse of the cell.",
            cues=["mitochondria", "ATP"], tags=["biology"],
        )
        expanded = graph.expand_query("mitochondria")
        # the cue equal to the query must not be appended again
        assert expanded.lower().count("mitochondria") == 1

    def test_no_match_returns_query_unchanged(self, graph):
        assert graph.expand_query("nonexistent topic") == "nonexistent topic"
