"""
Integration tests for the TLAQ end-to-end pipeline and evaluation metrics.

Tiny TKG (same entities as previous test suites):
  Entities  : Messi(0), Suarez(1), FCBarcelona(2), Neymar(3), PSG(4)
  Relations : playFor(0), serves(1)
  AttrPreds : bornIn(0), nationality(1)
  Attributes: Argentina(0), Uruguay(1), Brazil(2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.data.tkg_dataset import TKGDataset, RelationTriple, AttributeTriple
from src.pipeline.tlaq_pipeline import TLAQPipeline, TLAQConfig, build_pipeline
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    evaluate_at_k,
    evaluate_dataset,
    macro_average,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_tkg() -> TKGDataset:
    ds = TKGDataset()
    ds.add_relation_triple("Messi",  "playFor", "FCBarcelona", 2004, 2021)
    ds.add_relation_triple("Suarez", "serves",  "FCBarcelona", 2014, 2020)
    ds.add_relation_triple("Neymar", "playFor", "PSG",         2017, None)
    ds.add_attribute_triple("Messi",  "bornIn",      "Argentina")
    ds.add_attribute_triple("Suarez", "bornIn",      "Uruguay")
    ds.add_attribute_triple("Neymar", "bornIn",      "Brazil")
    ds.add_attribute_triple("Messi",  "nationality", "Argentina")
    ds.build_indices()
    return ds


@pytest.fixture
def pipeline(tiny_tkg) -> TLAQPipeline:
    cfg = TLAQConfig(embed_dim=32, top_k=5, bfs_depth=2)
    return build_pipeline(tiny_tkg, cfg)


# ═══════════════════════════════════════════════════════════════════════════
# ① Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestPrecisionAtK:

    def test_all_relevant(self):
        ranked   = [1, 2, 3]
        relevant = {1, 2, 3}
        assert precision_at_k(ranked, relevant, k=3) == pytest.approx(1.0)

    def test_none_relevant(self):
        ranked   = [1, 2, 3]
        relevant = {4, 5}
        assert precision_at_k(ranked, relevant, k=3) == pytest.approx(0.0)

    def test_partial(self):
        ranked   = [1, 2, 3, 4]
        relevant = {1, 3}
        assert precision_at_k(ranked, relevant, k=4) == pytest.approx(0.5)

    def test_k_larger_than_list(self):
        ranked   = [1, 2]
        relevant = {1}
        # top-3 but only 2 items: hits=1, k=3 → 1/3
        assert precision_at_k(ranked, relevant, k=3) == pytest.approx(1 / 3)

    def test_k_zero(self):
        assert precision_at_k([1, 2], {1}, k=0) == pytest.approx(0.0)


class TestRecallAtK:

    def test_all_found(self):
        ranked   = [1, 2, 3]
        relevant = {1, 2}
        assert recall_at_k(ranked, relevant, k=3) == pytest.approx(1.0)

    def test_none_found(self):
        ranked   = [4, 5]
        relevant = {1, 2}
        assert recall_at_k(ranked, relevant, k=2) == pytest.approx(0.0)

    def test_partial(self):
        ranked   = [1, 4, 5, 6]
        relevant = {1, 2}
        assert recall_at_k(ranked, relevant, k=4) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert recall_at_k([1, 2], set(), k=2) == pytest.approx(0.0)


class TestF1AtK:

    def test_perfect(self):
        ranked   = [1, 2]
        relevant = {1, 2}
        assert f1_at_k(ranked, relevant, k=2) == pytest.approx(1.0)

    def test_zero_both(self):
        ranked   = [1, 2]
        relevant = {3, 4}
        assert f1_at_k(ranked, relevant, k=2) == pytest.approx(0.0)

    def test_harmonic_mean(self):
        # P=0.5, R=1.0  →  F1 = 2*0.5*1.0 / (0.5+1.0) = 2/3
        ranked   = [1, 99]
        relevant = {1}
        p = precision_at_k(ranked, relevant, k=2)
        r = recall_at_k(ranked, relevant, k=2)
        f = f1_at_k(ranked, relevant, k=2)
        assert f == pytest.approx(2 * p * r / (p + r))


class TestEvaluateAtK:

    def test_keys_present(self):
        scores = evaluate_at_k([1, 2, 3], {1}, ks=[5, 10])
        assert set(scores.keys()) == {"P@5", "R@5", "F1@5", "P@10", "R@10", "F1@10"}

    def test_values_in_range(self):
        scores = evaluate_at_k(list(range(100)), set(range(10)), ks=[20, 40])
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_default_ks(self):
        scores = evaluate_at_k([1], {1})
        assert "P@20" in scores and "R@200" in scores


class TestMacroAverage:

    def test_single_query(self):
        d = {"P@20": 0.4, "R@20": 0.6}
        assert macro_average([d]) == pytest.approx(d)

    def test_average_two(self):
        d1 = {"P@20": 0.0, "R@20": 1.0}
        d2 = {"P@20": 1.0, "R@20": 0.0}
        avg = macro_average([d1, d2])
        assert avg["P@20"] == pytest.approx(0.5)
        assert avg["R@20"] == pytest.approx(0.5)

    def test_empty(self):
        assert macro_average([]) == {}


class TestEvaluateDataset:

    def test_output_keys(self):
        ranked   = [[1, 2, 3], [4, 5]]
        relevant = [{1}, {4}]
        out = evaluate_dataset(ranked, relevant, ks=[5])
        assert "P@5" in out and "R@5" in out and "F1@5" in out

    def test_perfect_retrieval(self):
        """If every query retrieves its answer first, P/R/F1@k = 1 for all k."""
        ranked   = [[i] for i in range(5)]
        relevant = [{i} for i in range(5)]
        out = evaluate_dataset(ranked, relevant, ks=[1])
        assert out["P@1"] == pytest.approx(1.0)
        assert out["R@1"] == pytest.approx(1.0)
        assert out["F1@1"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# ② Pipeline construction
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPipeline:

    def test_builds_without_error(self, tiny_tkg):
        cfg = TLAQConfig(embed_dim=16)
        pipe = build_pipeline(tiny_tkg, cfg)
        assert pipe is not None

    def test_rel_emb_table_shape(self, pipeline, tiny_tkg):
        table = pipeline._rel_emb_table
        assert table.shape == (tiny_tkg.num_relations, pipeline.cfg.embed_dim)

    def test_rel_emb_table_finite(self, pipeline):
        assert torch.isfinite(pipeline._rel_emb_table).all()


# ═══════════════════════════════════════════════════════════════════════════
# ③ Entity query (Algorithm 1 via pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityQuery:

    def test_returns_list(self, pipeline, tiny_tkg):
        messi_id  = tiny_tkg.entity2id["Messi"]
        play_id   = tiny_tkg.relation2id["playFor"]
        results   = pipeline.entity_query([(messi_id, play_id, -1, 2004, 2021)])
        assert isinstance(results, list)

    def test_results_are_relation_triples(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        play_id  = tiny_tkg.relation2id["playFor"]
        results  = pipeline.entity_query([(messi_id, play_id, -1, 2004, 2021)])
        for score, triple in results:
            assert isinstance(triple, RelationTriple)
            assert isinstance(score, float)

    def test_scores_ascending(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        play_id  = tiny_tkg.relation2id["playFor"]
        results  = pipeline.entity_query([(messi_id, play_id, -1, 2004, 2021)])
        scores   = [s for s, _ in results]
        assert scores == sorted(scores)

    def test_entity_query_ids(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        play_id  = tiny_tkg.relation2id["playFor"]
        ids      = pipeline.entity_query_ids([(messi_id, play_id, -1, 2004, 2021)])
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)


# ═══════════════════════════════════════════════════════════════════════════
# ④ Attribute query (Algorithm 2 via pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class TestAttributeQuery:

    def test_returns_list(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        born_id  = tiny_tkg.attr_pred2id["bornIn"]
        results  = pipeline.attribute_query([(messi_id, born_id, -1)])
        assert isinstance(results, list)

    def test_results_are_attribute_triples(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        born_id  = tiny_tkg.attr_pred2id["bornIn"]
        results  = pipeline.attribute_query([(messi_id, born_id, -1)])
        for score, triple in results:
            assert isinstance(triple, AttributeTriple)
            assert isinstance(score, float)

    def test_scores_ascending(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        born_id  = tiny_tkg.attr_pred2id["bornIn"]
        results  = pipeline.attribute_query([(messi_id, born_id, -1)])
        scores   = [s for s, _ in results]
        assert scores == sorted(scores)

    def test_attribute_query_ids(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        born_id  = tiny_tkg.attr_pred2id["bornIn"]
        ids      = pipeline.attribute_query_ids([(messi_id, born_id, -1)])
        assert isinstance(ids, list)


# ═══════════════════════════════════════════════════════════════════════════
# ⑤ Graph query (Part 3 via pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphQuery:

    def test_returns_list(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        fcb_id   = tiny_tkg.entity2id["FCBarcelona"]
        candidates = [
            ([messi_id], [], [0]),
            ([fcb_id],   [], [0]),
            ([messi_id, fcb_id], [], [0]),
        ]
        results = pipeline.graph_query([messi_id], [], [0], candidates)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_result_format(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        candidates = [([messi_id], [], []), ([], [], [])]
        results = pipeline.graph_query([messi_id], [], [], candidates)
        for score, idx in results:
            assert isinstance(score, float)
            assert 0 <= idx < len(candidates)

    def test_self_graph_closest(self, pipeline, tiny_tkg):
        """Query graph identical to first candidate should rank first."""
        messi_id = tiny_tkg.entity2id["Messi"]
        fcb_id   = tiny_tkg.entity2id["FCBarcelona"]
        query_e  = [messi_id]
        candidates = [
            ([messi_id], [], []),   # identical to query → should be closest
            ([fcb_id],   [], []),
        ]
        results = pipeline.graph_query(query_e, [], [], candidates, top_k=2)
        assert results[0][1] == 0   # index 0 should be closest

    def test_top_k_respected(self, pipeline, tiny_tkg):
        messi_id = tiny_tkg.entity2id["Messi"]
        candidates = [([messi_id], [], [])] * 10
        results = pipeline.graph_query([messi_id], [], [], candidates, top_k=3)
        assert len(results) == 3


# ═══════════════════════════════════════════════════════════════════════════
# ⑥ End-to-end evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndEvaluation:

    def test_entity_query_eval(self, pipeline, tiny_tkg):
        """
        Run entity query on two queries, collect ranked ids, compute P/R/F1.
        Checks that the evaluation loop completes without error and values
        are in [0, 1].
        """
        play_id   = tiny_tkg.relation2id["playFor"]
        messi_id  = tiny_tkg.entity2id["Messi"]
        neymar_id = tiny_tkg.entity2id["Neymar"]
        fcb_id    = tiny_tkg.entity2id["FCBarcelona"]
        psg_id    = tiny_tkg.entity2id["PSG"]

        queries   = [(messi_id, play_id, -1, 2004, 2021),
                     (neymar_id, play_id, -1, 2017, None)]
        ground_truth = [{fcb_id}, {psg_id}]

        all_ranked = [
            pipeline.entity_query_ids([q], top_k=10)
            for q in queries
        ]
        scores = evaluate_dataset(all_ranked, ground_truth, ks=[5, 10])

        for v in scores.values():
            assert 0.0 <= v <= 1.0, f"Metric out of range: {scores}"

    def test_attribute_query_eval(self, pipeline, tiny_tkg):
        born_id      = tiny_tkg.attr_pred2id["bornIn"]
        messi_id     = tiny_tkg.entity2id["Messi"]
        suarez_id    = tiny_tkg.entity2id["Suarez"]
        argentina_id = tiny_tkg.attribute2id["Argentina"]
        uruguay_id   = tiny_tkg.attribute2id["Uruguay"]

        queries      = [(messi_id, born_id, -1), (suarez_id, born_id, -1)]
        ground_truth = [{argentina_id}, {uruguay_id}]

        all_ranked = [
            pipeline.attribute_query_ids([q], top_k=10)
            for q in queries
        ]
        scores = evaluate_dataset(all_ranked, ground_truth, ks=[5, 10])

        for v in scores.values():
            assert 0.0 <= v <= 1.0, f"Metric out of range: {scores}"
