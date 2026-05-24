"""
Unit tests for the improved GCN module.

Uses a small hand-crafted TKG that mirrors the FC-Barcelona example in Figure 1
of the paper so we can verify matrix shapes and formula correctness by hand.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.data.tkg_dataset import TKGDataset
from src.models.improved_gcn import (
    ImprovedGCN,
    _build_weighted_adjacency,
    compute_attr,
    compute_final_entity_embedding,
    relational_reliability,
)
from src.algorithms.algo1_entity_query import bfs_neighbours, approximate_entity_query
from src.models.gcn_trainer import GCNTrainer


# ---------------------------------------------------------------------------
# Fixture: tiny TKG
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_tkg() -> TKGDataset:
    """
    Entities : Messi(0), Suarez(1), FCBarcelona(2), Neymar(3), PSG(4)
    Relations: playFor(0), serves(1)

    Triples:
      Messi   --playFor[2004,2021]--> FCBarcelona
      Suarez  --serves[2014,2020]--> FCBarcelona
      Neymar  --playFor[2017,+]---> PSG
    """
    ds = TKGDataset()
    ds.add_relation_triple("Messi",  "playFor", "FCBarcelona", 2004, 2021)
    ds.add_relation_triple("Suarez", "serves",  "FCBarcelona", 2014, 2020)
    ds.add_relation_triple("Neymar", "playFor", "PSG",         2017, None)
    ds.build_indices()
    return ds


# ---------------------------------------------------------------------------
# Tests: dataset
# ---------------------------------------------------------------------------

def test_dataset_sizes(tiny_tkg):
    assert tiny_tkg.num_entities == 5
    assert tiny_tkg.num_relations == 2
    assert len(tiny_tkg.relation_triples) == 3


def test_head_tail_counts(tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    fcb_id   = tiny_tkg.entity2id["FCBarcelona"]

    # Messi is head in 1 triple, tail in 0
    assert tiny_tkg.head_count[messi_id] == 1
    assert tiny_tkg.tail_count[messi_id] == 0

    # FCBarcelona is tail in 2 triples, head in 0
    assert tiny_tkg.head_count[fcb_id] == 0
    assert tiny_tkg.tail_count[fcb_id] == 2


# ---------------------------------------------------------------------------
# Tests: adjacency matrix construction
# ---------------------------------------------------------------------------

def test_adjacency_shape(tiny_tkg):
    A = _build_weighted_adjacency(tiny_tkg, torch.device("cpu"))
    N = tiny_tkg.num_entities
    assert A.shape == (N, N), f"Expected ({N},{N}), got {A.shape}"


def test_adjacency_values_positive(tiny_tkg):
    A = _build_weighted_adjacency(tiny_tkg, torch.device("cpu")).to_dense()
    assert (A >= 0).all(), "All adjacency weights should be non-negative"


def test_adjacency_diagonal_nonzero(tiny_tkg):
    """Self-loops must be present (diagonal should be non-zero after normalisation)."""
    A = _build_weighted_adjacency(tiny_tkg, torch.device("cpu")).to_dense()
    assert (A.diag() > 0).all(), "Diagonal (self-loops) should be positive"


# ---------------------------------------------------------------------------
# Tests: GCN forward pass
# ---------------------------------------------------------------------------

def test_gcn_output_shape(tiny_tkg):
    model = ImprovedGCN(tiny_tkg, embed_dim=32, num_layers=2)
    H = model()
    assert H.shape == (tiny_tkg.num_entities, 32)


def test_gcn_output_is_finite(tiny_tkg):
    model = ImprovedGCN(tiny_tkg, embed_dim=32, num_layers=2)
    H = model()
    assert torch.isfinite(H).all(), "GCN output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Tests: attr influence factor  (Eq 6)
# ---------------------------------------------------------------------------

def test_attr_returns_positive_values(tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    fcb_id   = tiny_tkg.entity2id["FCBarcelona"]
    play_id  = tiny_tkg.relation2id["playFor"]

    query_triples = [(messi_id, play_id, fcb_id)]
    scores = compute_attr(messi_id, query_triples, tiny_tkg)

    for v in scores.values():
        assert v > 0, "attr scores must be positive"
        assert v <= 1.0, "attr = exp(-x) so must be ≤ 1"


def test_attr_exp_formula(tiny_tkg):
    """Verify Eq 6 numerically for a known triple."""
    messi_id = tiny_tkg.entity2id["Messi"]
    fcb_id   = tiny_tkg.entity2id["FCBarcelona"]
    play_id  = tiny_tkg.relation2id["playFor"]

    query_triples = [(messi_id, play_id, fcb_id)]
    scores = compute_attr(messi_id, query_triples, tiny_tkg)

    # In our tiny TKG, <Messi, playFor, FCBarcelona> appears exactly once.
    # numerator=1, denominator=1  →  attr = exp(-1/1) = exp(-1)
    expected = math.exp(-1.0)
    actual = scores[(messi_id, play_id, fcb_id)]
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Tests: final entity embedding  (Eq 7)
# ---------------------------------------------------------------------------

def test_final_embedding_shape(tiny_tkg):
    model = ImprovedGCN(tiny_tkg, embed_dim=32, num_layers=2)
    H = model()
    messi_id = tiny_tkg.entity2id["Messi"]
    fcb_id   = tiny_tkg.entity2id["FCBarcelona"]
    play_id  = tiny_tkg.relation2id["playFor"]

    query_triples = [(messi_id, play_id, fcb_id)]
    emb = compute_final_entity_embedding(messi_id, query_triples, H, tiny_tkg)
    assert emb.shape == (32,)


# ---------------------------------------------------------------------------
# Tests: relational reliability  (Eq 8)
# ---------------------------------------------------------------------------

def test_rr_self_distance_is_zero(tiny_tkg):
    """Distance of an entity to itself must be zero."""
    model = ImprovedGCN(tiny_tkg, embed_dim=32, num_layers=2)
    H = model()
    messi_id = tiny_tkg.entity2id["Messi"]
    ev_emb = H[messi_id]
    rr = relational_reliability(ev_emb, H)

    assert rr[messi_id].item() < 1e-5, "Self-distance should be ~0"


def test_rr_output_length(tiny_tkg):
    model = ImprovedGCN(tiny_tkg, embed_dim=32, num_layers=2)
    H = model()
    rr = relational_reliability(H[0], H)
    assert rr.shape == (tiny_tkg.num_entities,)


# ---------------------------------------------------------------------------
# Tests: BFS
# ---------------------------------------------------------------------------

def test_bfs_neighbours_depth1(tiny_tkg):
    messi_id = tiny_tkg.entity2id["Messi"]
    fcb_id   = tiny_tkg.entity2id["FCBarcelona"]

    nbrs = bfs_neighbours(messi_id, tiny_tkg, max_depth=1)
    assert fcb_id in nbrs, "FCBarcelona should be a depth-1 neighbour of Messi"
    assert messi_id not in nbrs, "Root node must not be in its own neighbour set"


def test_bfs_neighbours_depth2(tiny_tkg):
    """At depth 2 from Messi we should also reach Suarez (via FCBarcelona)."""
    messi_id  = tiny_tkg.entity2id["Messi"]
    suarez_id = tiny_tkg.entity2id["Suarez"]

    nbrs = bfs_neighbours(messi_id, tiny_tkg, max_depth=2)
    assert suarez_id in nbrs


# ---------------------------------------------------------------------------
# Tests: Algorithm 1 end-to-end
# ---------------------------------------------------------------------------

def test_algo1_returns_top_k(tiny_tkg):
    model = ImprovedGCN(tiny_tkg, embed_dim=32, num_layers=2)
    messi_id = tiny_tkg.entity2id["Messi"]
    fcb_id   = tiny_tkg.entity2id["FCBarcelona"]
    play_id  = tiny_tkg.relation2id["playFor"]

    query_triples = [(messi_id, play_id, fcb_id)]
    results = approximate_entity_query(tiny_tkg, query_triples, model, top_k=2)

    assert len(results) <= 2
    # scores should be in ascending order
    if len(results) == 2:
        assert results[0][0] <= results[1][0]


# ---------------------------------------------------------------------------
# Tests: training loop (smoke test)
# ---------------------------------------------------------------------------

def test_trainer_loss_decreases(tiny_tkg):
    model   = ImprovedGCN(tiny_tkg, embed_dim=16, num_layers=1)
    trainer = GCNTrainer(model, tiny_tkg, margin=1.0, lr=1e-2)
    losses  = trainer.fit(epochs=20, batch_size=4, log_every=999)

    # Loss should not increase overall (allow small fluctuations)
    assert losses[-1] <= losses[0] + 0.5, (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
