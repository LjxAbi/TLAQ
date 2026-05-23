"""
Unit tests for HyTE attribute embedding (Part 2 of TLAQ).

Tiny TKG:
  Entities  : Messi(0), Suarez(1), FCBarcelona(2), Neymar(3), PSG(4)
  AttrPreds : bornIn(0), nationality(1)
  Attributes: Argentina(0), Uruguay(1), Brazil(2)

  Attribute triples:
    Messi   --bornIn--> Argentina
    Suarez  --bornIn--> Uruguay
    Neymar  --bornIn--> Brazil
    Messi   --nationality--> Argentina
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import torch.nn.functional as F
import pytest

from src.data.tkg_dataset import TKGDataset
from src.models.hyte_attribute import (
    HyTEAttributeEmbedding,
    project_onto_hyperplane,
    hyte_attr_score,
    preliminary_attr_embedding,
    compute_atta,
    compute_final_attr_embedding,
    attribute_confidence,
)
from src.algorithms.algo2_attribute_query import (
    bfs_entity_neighbours,
    approximate_attribute_query,
)
from src.models.hyte_trainer import HyTETrainer


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_tkg() -> TKGDataset:
    ds = TKGDataset()
    # relation triples (needed for BFS neighbours)
    ds.add_relation_triple("Messi",  "playFor", "FCBarcelona", 2004, 2021)
    ds.add_relation_triple("Suarez", "serves",  "FCBarcelona", 2014, 2020)
    ds.add_relation_triple("Neymar", "playFor", "PSG",         2017, None)
    # attribute triples
    ds.add_attribute_triple("Messi",  "bornIn",      "Argentina")
    ds.add_attribute_triple("Suarez", "bornIn",      "Uruguay")
    ds.add_attribute_triple("Neymar", "bornIn",      "Brazil")
    ds.add_attribute_triple("Messi",  "nationality", "Argentina")
    ds.build_indices()
    return ds


@pytest.fixture
def tiny_model(tiny_tkg) -> HyTEAttributeEmbedding:
    return HyTEAttributeEmbedding(tiny_tkg, embed_dim=32)


# ---------------------------------------------------------------------------
# Tests: dataset
# ---------------------------------------------------------------------------

def test_attr_triple_count(tiny_tkg):
    assert len(tiny_tkg.attribute_triples) == 4


def test_attr_pred_vocab(tiny_tkg):
    assert tiny_tkg.num_attr_preds == 2
    assert "bornIn" in tiny_tkg.attr_pred2id
    assert "nationality" in tiny_tkg.attr_pred2id


def test_attribute_vocab(tiny_tkg):
    assert tiny_tkg.num_attributes == 3


# ---------------------------------------------------------------------------
# Tests: hyperplane projection  (Eq 10)
# ---------------------------------------------------------------------------

def test_projection_removes_normal_component():
    """P_w(w) = 0  for unit vector w."""
    w = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([3.0, 4.0, 5.0])
    projected = project_onto_hyperplane(v, w)
    # component along w should be zero
    assert abs(projected[0].item()) < 1e-5
    # other components unchanged
    assert abs(projected[1].item() - 4.0) < 1e-5
    assert abs(projected[2].item() - 5.0) < 1e-5


def test_projection_of_normal_is_zero():
    """P_w(w) = 0."""
    torch.manual_seed(42)
    w = F.normalize(torch.randn(16), p=2, dim=0)
    projected = project_onto_hyperplane(w, w)
    assert projected.norm().item() < 1e-5


def test_projection_non_unit_normal():
    """Projection should normalise w internally."""
    w = torch.tensor([2.0, 0.0, 0.0])   # non-unit
    v = torch.tensor([3.0, 4.0, 5.0])
    projected = project_onto_hyperplane(v, w)
    assert abs(projected[0].item()) < 1e-5   # component along w removed


def test_projection_output_shape():
    v = torch.randn(8, 32)
    w = torch.randn(32)
    projected = project_onto_hyperplane(v, w.unsqueeze(0).expand(8, -1))
    assert projected.shape == (8, 32)


# ---------------------------------------------------------------------------
# Tests: HyTE scoring
# ---------------------------------------------------------------------------

def test_hyte_score_positive():
    """Score of a positive triple should be ≥ 0 (it's a norm)."""
    d = 32
    e, ta_d, a, w_ta = [torch.randn(d) for _ in range(4)]
    score = hyte_attr_score(e.unsqueeze(0), ta_d.unsqueeze(0),
                            a.unsqueeze(0), w_ta.unsqueeze(0))
    assert score.item() >= 0.0


def test_hyte_score_self_is_zero():
    """||P_w(e) + d_ta - P_w(e)|| = ||d_ta|| which is ~0 only if d_ta=0."""
    d = 16
    e   = torch.randn(d)
    w   = F.normalize(torch.randn(d), p=2, dim=0)
    d_ta = torch.zeros(d)   # zero translation → score = 0
    score = hyte_attr_score(e.unsqueeze(0), d_ta.unsqueeze(0),
                            e.unsqueeze(0), w.unsqueeze(0))
    assert score.item() < 1e-5


def test_hyte_score_batch():
    B, d = 5, 16
    e    = torch.randn(B, d)
    ta_d = torch.randn(B, d)
    a    = torch.randn(B, d)
    w    = F.normalize(torch.randn(B, d), p=2, dim=-1)
    scores = hyte_attr_score(e, ta_d, a, w)
    assert scores.shape == (B,)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# Tests: model forward
# ---------------------------------------------------------------------------

def test_model_score_shape(tiny_model, tiny_tkg):
    B = len(tiny_tkg.attribute_triples)
    entity_ids    = torch.tensor([t.entity    for t in tiny_tkg.attribute_triples])
    attr_pred_ids = torch.tensor([t.attr_pred for t in tiny_tkg.attribute_triples])
    attr_ids      = torch.tensor([t.attribute for t in tiny_tkg.attribute_triples])
    scores = tiny_model.score(entity_ids, attr_pred_ids, attr_ids)
    assert scores.shape == (B,)


def test_hyperplane_normalisation(tiny_model):
    """After normalize_hyperplanes(), all w_ta should have unit norm."""
    tiny_model.normalize_hyperplanes()
    norms = tiny_model.attr_pred_w.weight.norm(p=2, dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Tests: preliminary embedding  (Eq 9)
# ---------------------------------------------------------------------------

def test_preliminary_embedding_shape(tiny_model, tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    born_id  = tiny_tkg.attr_pred2id["bornIn"]
    a_prelim = preliminary_attr_embedding(messi_id, born_id, tiny_model)
    assert a_prelim.shape == (32,)


def test_preliminary_embedding_eq9(tiny_model, tiny_tkg):
    """ã = e + ta  (Eq 9): verify the sum manually."""
    messi_id = tiny_tkg.entity2id["Messi"]
    born_id  = tiny_tkg.attr_pred2id["bornIn"]
    with torch.no_grad():
        e_emb  = tiny_model.entity_emb.weight[messi_id]
        d_ta   = tiny_model.attr_pred_d.weight[born_id]
        expected = e_emb + d_ta
    actual = preliminary_attr_embedding(messi_id, born_id, tiny_model)
    assert (actual - expected).norm().item() < 1e-5


# ---------------------------------------------------------------------------
# Tests: compute_atta  (Eq 11)
# ---------------------------------------------------------------------------

def test_atta_sums_to_one(tiny_model, tiny_tkg):
    messi_id  = tiny_tkg.entity2id["Messi"]
    suarez_id = tiny_tkg.entity2id["Suarez"]
    neymar_id = tiny_tkg.entity2id["Neymar"]
    born_id   = tiny_tkg.attr_pred2id["bornIn"]

    neighbours = [suarez_id, neymar_id]
    atta = compute_atta(messi_id, neighbours, born_id, tiny_model)

    total = sum(atta.values())
    assert abs(total - 1.0) < 1e-5, f"atta should sum to 1, got {total}"


def test_atta_all_positive(tiny_model, tiny_tkg):
    messi_id  = tiny_tkg.entity2id["Messi"]
    suarez_id = tiny_tkg.entity2id["Suarez"]
    born_id   = tiny_tkg.attr_pred2id["bornIn"]

    atta = compute_atta(messi_id, [suarez_id], born_id, tiny_model)
    for v in atta.values():
        assert v >= 0.0, "atta values must be non-negative"


def test_atta_empty_neighbours(tiny_model, tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    born_id  = tiny_tkg.attr_pred2id["bornIn"]
    atta = compute_atta(messi_id, [], born_id, tiny_model)
    assert atta == {}


# ---------------------------------------------------------------------------
# Tests: final attribute embedding  (Eq 12)
# ---------------------------------------------------------------------------

def test_final_attr_embedding_shape(tiny_model, tiny_tkg):
    messi_id  = tiny_tkg.entity2id["Messi"]
    suarez_id = tiny_tkg.entity2id["Suarez"]
    born_id   = tiny_tkg.attr_pred2id["bornIn"]
    a_hat = compute_final_attr_embedding(messi_id, [suarez_id], born_id, tiny_model)
    assert a_hat.shape == (32,)


def test_final_attr_embedding_no_neighbours_fallback(tiny_model, tiny_tkg):
    """With no neighbours, result should equal the preliminary embedding."""
    messi_id = tiny_tkg.entity2id["Messi"]
    born_id  = tiny_tkg.attr_pred2id["bornIn"]
    a_hat    = compute_final_attr_embedding(messi_id, [], born_id, tiny_model)
    a_prelim = preliminary_attr_embedding(messi_id, born_id, tiny_model)
    assert (a_hat - a_prelim).norm().item() < 1e-5


def test_final_attr_embedding_finite(tiny_model, tiny_tkg):
    messi_id  = tiny_tkg.entity2id["Messi"]
    fcb_id    = tiny_tkg.entity2id["FCBarcelona"]
    born_id   = tiny_tkg.attr_pred2id["bornIn"]
    a_hat = compute_final_attr_embedding(messi_id, [fcb_id], born_id, tiny_model)
    assert torch.isfinite(a_hat).all()


# ---------------------------------------------------------------------------
# Tests: attribute confidence  (Eq 13)
# ---------------------------------------------------------------------------

def test_ac_self_distance_is_zero(tiny_model):
    a_hat = torch.randn(32)
    A     = torch.zeros(5, 32)
    A[2]  = a_hat
    ac    = attribute_confidence(a_hat, A)
    assert ac[2].item() < 1e-5, "AC to itself must be 0"


def test_ac_output_length(tiny_model, tiny_tkg):
    a_hat = torch.randn(32)
    A     = tiny_model.all_attr_embeddings().detach()
    ac    = attribute_confidence(a_hat, A)
    assert ac.shape == (tiny_tkg.num_attributes,)


def test_ac_non_negative(tiny_model, tiny_tkg):
    a_hat = torch.randn(32)
    A     = tiny_model.all_attr_embeddings().detach()
    ac    = attribute_confidence(a_hat, A)
    assert (ac >= 0).all()


# ---------------------------------------------------------------------------
# Tests: BFS
# ---------------------------------------------------------------------------

def test_bfs_from_messi_depth1(tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    fcb_id   = tiny_tkg.entity2id["FCBarcelona"]
    nbrs = bfs_entity_neighbours(messi_id, tiny_tkg, max_depth=1)
    assert fcb_id in nbrs
    assert messi_id not in nbrs


def test_bfs_from_messi_depth2(tiny_tkg):
    """Via FCBarcelona, Suarez should be reachable at depth 2."""
    messi_id  = tiny_tkg.entity2id["Messi"]
    suarez_id = tiny_tkg.entity2id["Suarez"]
    nbrs = bfs_entity_neighbours(messi_id, tiny_tkg, max_depth=2)
    assert suarez_id in nbrs


# ---------------------------------------------------------------------------
# Tests: Algorithm 2 end-to-end
# ---------------------------------------------------------------------------

def test_algo2_returns_results(tiny_model, tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    born_id  = tiny_tkg.attr_pred2id["bornIn"]
    # av (target attribute) unknown — represented as -1 (placeholder)
    query_triples = [(messi_id, born_id, -1)]
    results = approximate_attribute_query(tiny_tkg, query_triples, tiny_model, top_k=3)
    assert len(results) > 0


def test_algo2_scores_ascending(tiny_model, tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    born_id  = tiny_tkg.attr_pred2id["bornIn"]
    query_triples = [(messi_id, born_id, -1)]
    results = approximate_attribute_query(tiny_tkg, query_triples, tiny_model, top_k=3)
    scores = [r[0] for r in results]
    assert scores == sorted(scores), "Results should be sorted by AC score"


def test_algo2_returns_attribute_triples(tiny_model, tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    born_id  = tiny_tkg.attr_pred2id["bornIn"]
    query_triples = [(messi_id, born_id, -1)]
    results = approximate_attribute_query(tiny_tkg, query_triples, tiny_model, top_k=2)
    from src.data.tkg_dataset import AttributeTriple
    for _, triple in results:
        assert isinstance(triple, AttributeTriple)


# ---------------------------------------------------------------------------
# Tests: training loop (smoke test)
# ---------------------------------------------------------------------------

def test_hyte_trainer_loss_decreases(tiny_tkg):
    model   = HyTEAttributeEmbedding(tiny_tkg, embed_dim=16)
    trainer = HyTETrainer(model, tiny_tkg, margin=1.0, lr=1e-2)
    losses  = trainer.fit(epochs=30, batch_size=4, log_every=999)
    assert losses[-1] <= losses[0] + 0.5, (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_hyte_trainer_tdns(tiny_tkg):
    """TDNS mode should also converge without errors."""
    model   = HyTEAttributeEmbedding(tiny_tkg, embed_dim=16)
    trainer = HyTETrainer(model, tiny_tkg, margin=1.0, lr=1e-2,
                          negative_sampling="tdns")
    losses  = trainer.fit(epochs=20, batch_size=4, log_every=999)
    assert all(math.isfinite(l) for l in losses), "Loss contains NaN/Inf"


def test_hyperplanes_remain_unit_after_training(tiny_tkg):
    """After training, all hyperplane normals must remain unit vectors."""
    model   = HyTEAttributeEmbedding(tiny_tkg, embed_dim=16)
    trainer = HyTETrainer(model, tiny_tkg, margin=1.0, lr=1e-2)
    trainer.fit(epochs=10, batch_size=4, log_every=999)
    norms = model.attr_pred_w.weight.norm(p=2, dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-5
