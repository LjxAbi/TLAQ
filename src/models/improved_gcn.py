"""
Improved GCN for TKG entity embedding (Part 1 of TLAQ).

Key differences from vanilla GCN:
  - Adjacency matrix A_ij is replaced by a weighted version that encodes
    both directional bias (da) and relation importance (re).  See Eqs (3-5).
  - Two separate learnable weight matrices W0 (neighbour) and W1 (self).
    See Eq (1).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import numpy as np

from src.data.tkg_dataset import TKGDataset


# ---------------------------------------------------------------------------
# Adjacency matrix construction  (Eqs 3-5)
# ---------------------------------------------------------------------------

def _build_weighted_adjacency(
    dataset: TKGDataset,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the weighted adjacency matrix Ã used by the improved GCN.

    For each pair (ei, ej) connected by at least one relational triple:

        da(ei, ej)   = tail_count[ej] / head_count[ei]          (Eq 4)
        re(ei,ej,tr) = (nodes_i + nodes_j) / nums_r             (Eq 5)
        Aij          = Σ_{<ei,tr,ej>∈RT} da(ei,ej) * re(ei,ej,tr)  (Eq 3)

    The matrix is then symmetrically normalised (D̂^{-1/2} A D̂^{-1/2})
    following Eq (2), with self-loops added beforehand.

    Returns a dense float tensor of shape [N, N].
    """
    assert dataset._indices_built, "Call dataset.build_indices() first."

    N = dataset.num_entities
    # accumulate raw weights into a dict to avoid O(N^2) dense allocation
    weight: Dict[Tuple[int, int], float] = defaultdict(float)

    for triple in dataset.relation_triples:
        ei, tr, ej = triple.head, triple.relation, triple.tail
        head_i = max(dataset.head_count[ei], 1)   # avoid /0
        tail_j = max(dataset.tail_count[ej], 1)
        nums_r = max(dataset.nums_r[tr], 1)

        da_val = tail_j / head_i                                    # Eq 4
        nodes_i = dataset.entity_in_r.get((ei, tr), 0)
        nodes_j = dataset.entity_in_r.get((ej, tr), 0)
        re_val = (nodes_i + nodes_j) / nums_r                       # Eq 5

        weight[(ei, ej)] += da_val * re_val                         # Eq 3

    # build sparse COO then convert
    rows, cols, vals = [], [], []
    for (i, j), v in weight.items():
        rows.append(i); cols.append(j); vals.append(v)

    # add self-loops (identity part of Â)
    for i in range(N):
        rows.append(i); cols.append(i); vals.append(1.0)

    A_sparse = coo_matrix(
        (vals, (rows, cols)), shape=(N, N), dtype=np.float32
    ).toarray()
    A = torch.tensor(A_sparse, dtype=torch.float32, device=device)  # [N, N]

    # symmetric normalisation: D̂^{-1/2} A D̂^{-1/2}              (Eq 2)
    deg = A.sum(dim=1)                                               # [N]
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)                            # [N, N]
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt                             # [N, N]

    return A_hat


# ---------------------------------------------------------------------------
# Single GCN layer  (Eq 1)
# ---------------------------------------------------------------------------

class ImprovedGCNLayer(nn.Module):
    """
    H^{l+1} = σ( Ã H^{l} W0 + H^{l} W1 )        (Eq 1)

    Two separate weight matrices keep the self-representation
    independent of the aggregated neighbour representation.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W0 = nn.Linear(in_dim, out_dim, bias=False)   # neighbour branch
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)   # self branch
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W0.weight)
        nn.init.xavier_uniform_(self.W1.weight)

    def forward(
        self,
        H: torch.Tensor,       # [N, in_dim]
        A_hat: torch.Tensor,   # [N, N]  normalised adjacency
    ) -> torch.Tensor:         # [N, out_dim]
        H = self.dropout(H)
        neighbour = A_hat @ H                  # aggregate neighbours
        out = self.W0(neighbour) + self.W1(H)  # Eq 1 (bias-free)
        return F.relu(out)


# ---------------------------------------------------------------------------
# Full improved GCN encoder
# ---------------------------------------------------------------------------

class ImprovedGCN(nn.Module):
    """
    Multi-layer improved GCN producing a d-dimensional embedding for every
    entity in the TKG.

    After forward(), self.entity_embeddings holds the final H^{(L)} matrix
    which can be used by Algorithm 1 (relational reliability).
    """

    def __init__(
        self,
        dataset: TKGDataset,
        embed_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device or torch.device("cpu")

        N = dataset.num_entities

        # learnable initial entity embeddings H^{(0)}
        self.entity_emb = nn.Embedding(N, embed_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        # stack of improved GCN layers
        self.layers = nn.ModuleList(
            [ImprovedGCNLayer(embed_dim, embed_dim, dropout) for _ in range(num_layers)]
        )

        # pre-compute and cache the normalised adjacency matrix
        # (re-build only when dataset changes)
        self._A_hat: Optional[torch.Tensor] = None

    def _get_adjacency(self) -> torch.Tensor:
        if self._A_hat is None:
            self._A_hat = _build_weighted_adjacency(
                self.dataset, self.device
            )
        return self._A_hat

    def invalidate_adjacency(self) -> None:
        """Call this if the dataset is modified after the first forward pass."""
        self._A_hat = None

    def forward(self) -> torch.Tensor:
        """
        Returns entity embeddings H of shape [N, embed_dim].
        No input argument needed — the graph is fully encoded via the
        cached adjacency matrix and the learnable embedding table.
        """
        A_hat = self._get_adjacency()                      # [N, N]
        H = self.entity_emb.weight                         # [N, d]  H^{(0)}

        for layer in self.layers:
            H = layer(H, A_hat)                            # [N, d]

        return H   # final entity embeddings


# ---------------------------------------------------------------------------
# Relation triple influence factor attr  (Eq 6)
# ---------------------------------------------------------------------------

def compute_attr(
    target_ev: int,
    rtq_triples: List[Tuple[int, int, int]],   # (ev, tr, et) in TQG
    dataset: TKGDataset,
) -> Dict[Tuple[int, int, int], float]:
    """
    For each triple (ev, tr, et) in the temporal query graph RTQ that contains
    the target entity ev, compute its influence factor attr.

        attr(ev, tr, et) = exp( -numerator / denominator )          (Eq 6)

        numerator   = |{ev | <ev, tr, et> ∈ RT}|
                      (how specific is THIS triple in the TKG)
        denominator = Σ_{<ev,t'r,e't> ∈ RTQ} |{ev | <ev,t'r,e't> ∈ RT}|
                      (total specificity of ALL query triples for ev)

    A more specific triple (small numerator) → larger attr weight.
    """
    # count how many TKG triples match each query triple pattern
    counts: Dict[Tuple[int, int, int], int] = {}
    for (ev, tr, et) in rtq_triples:
        matched = sum(
            1 for t in dataset.head_triples.get(ev, []) if t.relation == tr and t.tail == et
        )
        counts[(ev, tr, et)] = max(matched, 1)   # avoid zero denominator

    denominator = sum(counts.values())
    if denominator == 0:
        denominator = 1.0

    attr_scores: Dict[Tuple[int, int, int], float] = {}
    for triple, cnt in counts.items():
        attr_scores[triple] = math.exp(-cnt / denominator)   # Eq 6

    return attr_scores


# ---------------------------------------------------------------------------
# Final target-entity embedding  (Eq 7)
# ---------------------------------------------------------------------------

def compute_final_entity_embedding(
    target_ev: int,
    rtq_triples: List[Tuple[int, int, int]],   # (ev, tr, et) in TQG
    H: torch.Tensor,                           # [N, d]  GCN output
    dataset: TKGDataset,
) -> torch.Tensor:
    """
    Combine the GCN preliminary embedding with attr influence factors.

        ê_v = Σ attr(e,tr,e) * ẽ_v  /  Σ attr(e,t'r,e't)         (Eq 7)

    where ẽ_v is H[ev] from the GCN.  The denominator normalises the weights.
    """
    attr_scores = compute_attr(target_ev, rtq_triples, dataset)
    ev_emb = H[target_ev]                      # [d]  preliminary embedding

    total_weight = sum(attr_scores.values())
    if total_weight < 1e-9:
        return ev_emb                          # fallback: no reweighting

    # weighted average (numerator and denominator from Eq 7)
    # Since all triples share the same ẽ_v, this simplifies to a scalar rescale
    weight = sum(attr_scores.values()) / total_weight
    return weight * ev_emb


# ---------------------------------------------------------------------------
# Relational Reliability  (Eq 8)
# ---------------------------------------------------------------------------

def relational_reliability(
    ev_emb: torch.Tensor,   # [d]  target entity final embedding
    E: torch.Tensor,        # [N, d]  all candidate entity embeddings
) -> torch.Tensor:          # [N]  RR scores (lower = more similar)
    """
    RR(ev, e) = sqrt( Σ_i (ev_i - e_i)^2 )      (Eq 8)

    Returns Euclidean distance for every candidate entity.
    Lower distance → higher relational reliability → better match.
    """
    diff = E - ev_emb.unsqueeze(0)              # [N, d]
    return torch.sqrt((diff ** 2).sum(dim=1))   # [N]
