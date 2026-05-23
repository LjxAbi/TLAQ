"""
Unit tests for TLAQ Part 3:
  - Seven-level time encoding  (src/utils/time_encoding.py)
  - Temporal predicate encoder (src/models/temporal_predicate_encoder.py)
  - Graph-level embedding       (src/models/graph_level_embedding.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import torch.nn.functional as F
import pytest

from src.utils.time_encoding import (
    TIME_DIM,
    encode_timestamp,
    encode_interval,
    batch_encode_timestamps,
    decode_timestamp,
    _WIDTHS,
    _OFFSETS,
)
from src.models.temporal_predicate_encoder import (
    TemporalPredicateEncoder,
    PredicateTimeIndex,
)
from src.models.graph_level_embedding import (
    ContextVector,
    attention_aggregate,
    GraphLevelEmbedding,
    graph_similarity,
    batch_graph_similarity,
    ApproximateResultsGenerator,
)
from src.data.tkg_dataset import TKGDataset


# ═══════════════════════════════════════════════════════════════════════════
# ① Seven-level time encoding
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeEncoding:

    def test_total_dimension(self):
        assert TIME_DIM == 52, f"Expected 52, got {TIME_DIM}"

    def test_sub_vector_widths(self):
        expected = {"C_millennium": 3, "C_century": 10, "D": 10, "Y": 10,
                    "Q": 4, "M": 3, "W": 5, "DS": 7}
        assert _WIDTHS == expected

    def test_offsets_monotone(self):
        offsets = list(_OFFSETS.values())
        assert offsets == sorted(offsets)

    # --- encode_timestamp shape & dtype ---
    def test_output_shape(self):
        v = encode_timestamp(2004)
        assert v.shape == (TIME_DIM,)

    def test_output_dtype(self):
        v = encode_timestamp(2004)
        assert v.dtype == torch.float32

    # --- one-hot property: each sub-vector has exactly one 1 ---
    def test_each_sub_vector_one_hot(self):
        v = encode_timestamp(2004, month=1, day=1)
        for name, size in _WIDTHS.items():
            offset  = _OFFSETS[name]
            segment = v[offset: offset + size]
            assert segment.sum().item() == 1.0, \
                f"Sub-vector '{name}' is not one-hot: {segment.tolist()}"

    def test_binary_values(self):
        v = encode_timestamp(1990, month=6, day=15)
        assert ((v == 0) | (v == 1)).all()

    # --- specific encodings verified by hand ---
    def test_millennium_2000s(self):
        """Year 2025 → millennium index = 2."""
        v = encode_timestamp(2025)
        decoded = decode_timestamp(v)
        assert decoded["C_millennium"] == 2

    def test_millennium_1000s(self):
        """Year 1990 → millennium index = 1."""
        v = encode_timestamp(1990)
        decoded = decode_timestamp(v)
        assert decoded["C_millennium"] == 1

    def test_century_in_millennium(self):
        """Year 1990 → century_in_millennium = 9 (years 1900–1999)."""
        v = encode_timestamp(1990)
        decoded = decode_timestamp(v)
        assert decoded["C_century"] == 9

    def test_decade(self):
        """Year 1990 → decade_in_century = 9 (years 1990–1999)."""
        v = encode_timestamp(1990)
        decoded = decode_timestamp(v)
        assert decoded["D"] == 9

    def test_year_in_decade(self):
        """Year 1993 → year_in_decade = 3."""
        v = encode_timestamp(1993)
        decoded = decode_timestamp(v)
        assert decoded["Y"] == 3

    def test_quarter_q1(self):
        """January → quarter = 0 (Q1)."""
        v = encode_timestamp(2000, month=1)
        assert decode_timestamp(v)["Q"] == 0

    def test_quarter_q2(self):
        """April → quarter = 1 (Q2)."""
        v = encode_timestamp(2000, month=4)
        assert decode_timestamp(v)["Q"] == 1

    def test_quarter_q4(self):
        """October → quarter = 3 (Q4)."""
        v = encode_timestamp(2000, month=10)
        assert decode_timestamp(v)["Q"] == 3

    def test_month_in_quarter(self):
        """February is the 2nd month of Q1 → month_in_quarter = 1."""
        v = encode_timestamp(2000, month=2)
        assert decode_timestamp(v)["M"] == 1

    def test_week_in_month_first(self):
        """Day 1 → week 0."""
        v = encode_timestamp(2000, day=1)
        assert decode_timestamp(v)["W"] == 0

    def test_week_in_month_third(self):
        """Day 15 → week 2 (days 15–21)."""
        v = encode_timestamp(2000, day=15)
        assert decode_timestamp(v)["W"] == 2

    def test_different_years_differ(self):
        """Two different years must produce different encodings."""
        v1 = encode_timestamp(2004)
        v2 = encode_timestamp(2021)
        assert not torch.equal(v1, v2)

    # --- encode_interval ---
    def test_interval_shape(self):
        v = encode_interval(2004, 2021)
        assert v.shape == (TIME_DIM * 2,)

    def test_interval_first_half_is_start(self):
        ts = encode_timestamp(2004)
        iv = encode_interval(2004, 2021)
        assert torch.equal(iv[:TIME_DIM], ts)

    def test_interval_second_half_is_end(self):
        te = encode_timestamp(2021)
        iv = encode_interval(2004, 2021)
        assert torch.equal(iv[TIME_DIM:], te)

    def test_interval_none_end(self):
        """Open-ended interval (end=None) should not raise."""
        v = encode_interval(2017, None)
        assert v.shape == (TIME_DIM * 2,)

    # --- batch encoding ---
    def test_batch_shape(self):
        years = [1990, 2004, 2021]
        v = batch_encode_timestamps(years)
        assert v.shape == (3, TIME_DIM)

    def test_batch_matches_single(self):
        years = [1990, 2004]
        batch = batch_encode_timestamps(years)
        for i, y in enumerate(years):
            assert torch.equal(batch[i], encode_timestamp(y))

    # --- decode round-trip ---
    def test_decode_encode_roundtrip(self):
        v = encode_timestamp(1987, month=3, day=14)
        d = decode_timestamp(v)
        assert d["C_millennium"] == 1   # 1000s
        assert d["C_century"]    == 9   # 1900s
        assert d["D"]            == 8   # 1980s
        assert d["Y"]            == 7   # 1987


# ═══════════════════════════════════════════════════════════════════════════
# ② Temporal predicate encoder (LSTM)
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalPredicateEncoder:

    @pytest.fixture
    def encoder(self):
        return TemporalPredicateEncoder(embed_dim=32, time_dim=TIME_DIM)

    @pytest.fixture
    def encoder_interval(self):
        return TemporalPredicateEncoder(embed_dim=32, time_dim=TIME_DIM * 2)

    def test_output_shape(self, encoder):
        pred = torch.randn(4, 32)
        time = torch.zeros(4, TIME_DIM)
        out  = encoder(pred, time)
        assert out.shape == (4, 32)

    def test_output_finite(self, encoder):
        pred = torch.randn(8, 32)
        time = encode_timestamp(2004).unsqueeze(0).expand(8, -1)
        out  = encoder(pred, time)
        assert torch.isfinite(out).all()

    def test_encode_single_shape(self, encoder):
        pred = torch.randn(32)
        time = encode_timestamp(2021)
        out  = encoder.encode_single(pred, time)
        assert out.shape == (32,)

    def test_interval_encoder(self, encoder_interval):
        pred = torch.randn(3, 32)
        time = encode_interval(2004, 2021).unsqueeze(0).expand(3, -1)
        out  = encoder_interval(pred, time)
        assert out.shape == (3, 32)

    def test_different_times_produce_different_outputs(self, encoder):
        """Same predicate, different timestamps → different representations."""
        pred = torch.randn(32)
        t1   = encoder.encode_single(pred, encode_timestamp(2004))
        t2   = encoder.encode_single(pred, encode_timestamp(2021))
        assert not torch.allclose(t1, t2), \
            "Different timestamps must produce different predicate embeddings"

    def test_same_time_consistent(self, encoder):
        """Deterministic: same inputs → same outputs (eval mode)."""
        encoder.eval()
        pred = torch.randn(32)
        time = encode_timestamp(2004)
        out1 = encoder.encode_single(pred, time)
        out2 = encoder.encode_single(pred, time)
        assert torch.allclose(out1, out2)

    def test_multi_layer_encoder(self):
        encoder = TemporalPredicateEncoder(embed_dim=16, time_dim=TIME_DIM,
                                           num_layers=2)
        pred = torch.randn(4, 16)
        time = torch.zeros(4, TIME_DIM)
        out  = encoder(pred, time)
        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()

    def test_predicate_time_index(self):
        """Smoke-test PredicateTimeIndex with tiny TKG."""
        from src.data.tkg_dataset import TKGDataset
        ds = TKGDataset()
        ds.add_relation_triple("Messi", "playFor", "FCBarcelona", 2004, 2021)
        ds.add_relation_triple("Neymar", "playFor", "PSG", 2017, None)
        ds.build_indices()

        d = 32
        pred_table = torch.randn(ds.num_relations, d)
        encoder    = TemporalPredicateEncoder(embed_dim=d, time_dim=TIME_DIM * 2)

        index = PredicateTimeIndex(
            encoder, pred_table, ds.relation_triples, use_interval=True
        )
        r_id = ds.relation2id["playFor"]
        emb  = index.get(r_id)
        assert emb.shape == (d,)
        assert torch.isfinite(emb).all()


# ═══════════════════════════════════════════════════════════════════════════
# ③ Graph-level embedding (Eq 14–16)
# ═══════════════════════════════════════════════════════════════════════════

class TestContextVector:

    def test_output_shape(self):
        ctx = ContextVector(embed_dim=16)
        X   = torch.randn(5, 16)
        c   = ctx(X)
        assert c.shape == (16,)

    def test_tanh_range(self):
        """Output must be in [-1, 1] (tanh saturates to ±1 in float32)."""
        ctx = ContextVector(embed_dim=32)
        X   = torch.randn(10, 32) * 100   # large magnitude
        c   = ctx(X)
        assert (c >= -1).all() and (c <= 1).all()

    def test_empty_input(self):
        """Empty set → zero vector."""
        ctx = ContextVector(embed_dim=8)
        X   = torch.zeros(0, 8)
        c   = ctx(X)
        assert c.shape == (8,)
        assert (c == 0).all()

    def test_single_node(self):
        ctx = ContextVector(embed_dim=16)
        X   = torch.randn(1, 16)
        c   = ctx(X)
        assert c.shape == (16,)


class TestAttentionAggregate:

    def test_output_shape(self):
        X   = torch.randn(6, 16)
        ctx = torch.randn(16)
        agg = attention_aggregate(X, ctx)
        assert agg.shape == (16,)

    def test_sigmoid_weights_in_0_1(self):
        """All sigmoid values are in (0,1), so weighted sum ≤ ||X||₁."""
        X   = torch.eye(5, 5)         # simple case
        ctx = torch.ones(5)
        agg = attention_aggregate(X, ctx)
        assert (agg >= 0).all()

    def test_empty_input(self):
        ctx = torch.randn(8)
        agg = attention_aggregate(torch.zeros(0, 8), ctx)
        assert (agg == 0).all()


class TestGraphLevelEmbedding:

    @pytest.fixture
    def embedder(self):
        return GraphLevelEmbedding(embed_dim=32)

    def test_output_shape(self, embedder):
        E = torch.randn(4, 32)
        A = torch.randn(2, 32)
        R = torch.randn(3, 32)
        g = embedder(E, A, R)
        assert g.shape == (32,)

    def test_output_finite(self, embedder):
        E = torch.randn(5, 32)
        A = torch.randn(3, 32)
        R = torch.randn(2, 32)
        g = embedder(E, A, R)
        assert torch.isfinite(g).all()

    def test_only_entities(self, embedder):
        E = torch.randn(4, 32)
        A = torch.zeros(0, 32)
        R = torch.zeros(0, 32)
        g = embedder(E, A, R)
        assert g.shape == (32,)
        assert torch.isfinite(g).all()

    def test_all_empty_raises_not_crash(self, embedder):
        E = torch.zeros(0, 32)
        A = torch.zeros(0, 32)
        R = torch.zeros(0, 32)
        g = embedder(E, A, R)
        assert g.shape == (32,)
        assert (g == 0).all()

    def test_embed_graph_convenience(self, embedder):
        E = torch.randn(3, 32)
        g = embedder.embed_graph(entity_embs=E)
        assert g.shape == (32,)

    def test_different_graphs_differ(self, embedder):
        """Two graphs with different entity sets should produce different g."""
        torch.manual_seed(0)
        E1 = torch.randn(4, 32)
        E2 = torch.randn(4, 32)
        g1 = embedder(E1, torch.zeros(0, 32), torch.zeros(0, 32))
        g2 = embedder(E2, torch.zeros(0, 32), torch.zeros(0, 32))
        assert not torch.allclose(g1, g2)


class TestGraphSimilarity:

    def test_self_similarity_is_zero(self):
        g = torch.randn(32)
        assert graph_similarity(g, g).item() == pytest.approx(0.0, abs=1e-5)

    def test_symmetric(self):
        g1 = torch.randn(16)
        g2 = torch.randn(16)
        assert graph_similarity(g1, g2).item() == pytest.approx(
            graph_similarity(g2, g1).item(), abs=1e-5
        )

    def test_non_negative(self):
        g1 = torch.randn(32)
        g2 = torch.randn(32)
        assert graph_similarity(g1, g2).item() >= 0.0

    def test_batch_shape(self):
        g_q   = torch.randn(16)
        g_all = torch.randn(10, 16)
        scores = batch_graph_similarity(g_q, g_all)
        assert scores.shape == (10,)

    def test_batch_matches_single(self):
        torch.manual_seed(42)
        g_q   = torch.randn(8)
        g_all = torch.randn(5, 8)
        batch = batch_graph_similarity(g_q, g_all)
        for i in range(5):
            expected = graph_similarity(g_q, g_all[i]).item()
            assert batch[i].item() == pytest.approx(expected, abs=1e-5)

    def test_batch_closest_is_self(self):
        """When query is in the candidate list, it should rank first."""
        g_q   = torch.randn(16)
        others = torch.randn(5, 16)
        g_all  = torch.cat([g_q.unsqueeze(0), others], dim=0)
        scores = batch_graph_similarity(g_q, g_all)
        assert scores.argmin().item() == 0


class TestApproximateResultsGenerator:

    @pytest.fixture
    def generator(self):
        d  = 16
        Ne = 8
        Na = 4
        Nr = 3
        embedder   = GraphLevelEmbedding(embed_dim=d)
        entity_tab = torch.randn(Ne, d)
        attr_tab   = torch.randn(Na, d)
        rel_tab    = torch.randn(Nr, d)
        return ApproximateResultsGenerator(embedder, entity_tab, attr_tab, rel_tab)

    def test_embed_graph_shape(self, generator):
        g = generator.embed_graph([0, 1, 2], [0], [0])
        assert g.shape == (16,)

    def test_embed_graph_empty(self, generator):
        g = generator.embed_graph([], [], [])
        assert g.shape == (16,)

    def test_rank_candidates_sorted(self, generator):
        g_q    = generator.embed_graph([0], [], [])
        cands  = [([i], [], []) for i in range(5)]
        ranked = generator.rank_candidates(g_q, cands, top_k=3)
        scores = [r[0] for r in ranked]
        assert scores == sorted(scores), "Candidates must be sorted by similarity"

    def test_rank_candidates_top_k(self, generator):
        g_q   = generator.embed_graph([0], [], [])
        cands = [([i], [], []) for i in range(8)]
        ranked = generator.rank_candidates(g_q, cands, top_k=3)
        assert len(ranked) == 3

    def test_rank_candidates_empty(self, generator):
        g_q    = generator.embed_graph([0], [], [])
        ranked = generator.rank_candidates(g_q, [], top_k=5)
        assert ranked == []

    def test_self_graph_is_closest(self, generator):
        """A query graph should be most similar to itself."""
        ids    = [0, 1, 2]
        g_self = generator.embed_graph(ids, [], [])
        cands  = [([i], [], []) for i in range(8)]
        # insert self as candidate 3
        cands.insert(3, (ids, [], []))
        ranked = generator.rank_candidates(g_self, cands, top_k=len(cands))
        best_idx = ranked[0][1]
        assert best_idx == 3, f"Self graph should rank first, got index {best_idx}"


# ═══════════════════════════════════════════════════════════════════════════
# ④ Full Part 3 integration: time encoding → LSTM → graph embedding
# ═══════════════════════════════════════════════════════════════════════════

class TestPart3Integration:

    def test_full_pipeline(self):
        """
        End-to-end: timestamp → LSTM predicate embedding → graph embedding → sim.
        """
        d      = 32
        n_rels = 4

        # time-aware predicate embeddings via LSTM
        encoder    = TemporalPredicateEncoder(embed_dim=d, time_dim=TIME_DIM * 2)
        pred_table = torch.randn(n_rels, d)
        time_start = encode_timestamp(2004)
        time_end   = encode_timestamp(2021)
        time_vec   = torch.cat([time_start, time_end])   # [104]

        rel_embs = torch.stack([
            encoder.encode_single(pred_table[i], time_vec)
            for i in range(n_rels)
        ])                                               # [4, 32]

        # graph embedding
        embedder = GraphLevelEmbedding(embed_dim=d)
        E  = torch.randn(5, d)
        A  = torch.randn(2, d)
        g1 = embedder(E, A, rel_embs)                   # [32]

        # different temporal context → different graph embedding
        time_vec2 = torch.cat([encode_timestamp(1990), encode_timestamp(2000)])
        rel_embs2 = torch.stack([
            encoder.encode_single(pred_table[i], time_vec2)
            for i in range(n_rels)
        ])
        g2 = embedder(E, A, rel_embs2)

        assert g1.shape == g2.shape == (d,)
        assert torch.isfinite(g1).all() and torch.isfinite(g2).all()
        # different time contexts must produce different graph embeddings
        assert not torch.allclose(g1, g2)

    def test_similarity_order_preserved(self):
        """
        Graph most similar to query (same entities) should score lower than
        a completely different graph.
        """
        d = 16
        embedder = GraphLevelEmbedding(embed_dim=d)

        torch.manual_seed(7)
        E_base  = torch.randn(4, d)
        E_close = E_base + 0.01 * torch.randn(4, d)   # nearly identical
        E_far   = torch.randn(4, d) * 5               # very different

        empty = torch.zeros(0, d)
        g_q     = embedder(E_base,  empty, empty)
        g_close = embedder(E_close, empty, empty)
        g_far   = embedder(E_far,   empty, empty)

        sim_close = graph_similarity(g_q, g_close).item()
        sim_far   = graph_similarity(g_q, g_far).item()
        assert sim_close < sim_far, (
            f"Close graph ({sim_close:.4f}) should be more similar "
            f"than far graph ({sim_far:.4f})"
        )
