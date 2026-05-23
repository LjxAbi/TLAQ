"""
Graph-level embedding and similarity for TLAQ Part 3  (Section 3.5, Figure 8).

Formulae implemented:

  Eq 14  c(X) = tanh( (1/s Σᵢ xᵢ) W )        context vector for a set X
  Eq 15  g = avg(  Σᵢ f(eᵢᵀ c(E)) eᵢ
                  + Σⱼ f(aⱼᵀ c(A)) aⱼ
                  + Σₖ f(rₖᵀ c(R)) rₖ  )       graph embedding
  Eq 16  sim(g₁, g₂) = Σᵢ |g₁ᵢ − g₂ᵢ|         L1 graph similarity

Dual-branch structure (Figure 8):
  Top    branch : linear average → W·tanh → C(E)^d     (global context)
  Bottom branch : sigmoid weights on entity scores → N(E)^d  (local attention)

The bottom branch in Figure 8 is the attention mechanism embedded in Eq 15:
  f(eᵢᵀ c(E)) uses sigmoid to map attention scores to [0, 1] weights.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Context vector  (Eq 14)
# ---------------------------------------------------------------------------

class ContextVector(nn.Module):
    """
    c(X) = tanh( (1/s Σᵢ xᵢ) W )              (Eq 14)

    Computes a single d-dimensional "graph summary" vector from a set of
    d-dimensional node embeddings X ∈ R^{s×d}.

    W is a learnable d×d weight matrix, shared across all node types.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : Tensor  [s, d]   set of embeddings (s nodes, d dimensions)

        Returns
        -------
        context : Tensor [d]   tanh( mean(X) @ W )
        """
        if X.size(0) == 0:
            return torch.zeros(X.size(-1), dtype=X.dtype, device=X.device)
        mean_x = X.mean(dim=0)          # [d]  linear average  (1/s Σxᵢ)
        return torch.tanh(self.W(mean_x))   # [d]


# ---------------------------------------------------------------------------
# Attention-weighted node aggregation  (bottom branch, Eq 15)
# ---------------------------------------------------------------------------

def attention_aggregate(
    X:   torch.Tensor,   # [s, d]  node embeddings
    ctx: torch.Tensor,   # [d]     context vector c(X)
) -> torch.Tensor:       # [d]     attention-weighted sum
    """
    Σᵢ f(xᵢᵀ c(X)) xᵢ      where  f = sigmoid           (Eq 15 inner sum)

    Each node's contribution is gated by how aligned it is with the
    global context vector.  sigmoid maps scores to [0,1] ensuring the
    bottom branch outputs N(E) ∈ [0,1]^d after normalisation.
    """
    if X.size(0) == 0:
        return torch.zeros_like(ctx)
    scores  = (X @ ctx)                 # [s]   eᵢᵀ c(X)
    weights = torch.sigmoid(scores)     # [s]   f(·) ∈ (0,1)
    return (weights.unsqueeze(1) * X).sum(dim=0)   # [d]


# ---------------------------------------------------------------------------
# Graph-level embedding module  (Eq 14–15)
# ---------------------------------------------------------------------------

class GraphLevelEmbedding(nn.Module):
    """
    Converts sets of vertex/edge embeddings into a single graph embedding g.

    The module maintains separate ContextVector instances for entities (E),
    attributes (A), and relations / time-aware predicates (R), each sharing
    the same embed_dim but with independent weight matrices W.

    Forward signature:
        g = model(E, A, R)         →  [d]
    where E, A, R are variable-length embedding matrices.

    Empty matrices (no nodes of a certain type) are handled gracefully.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ctx_E = ContextVector(embed_dim)   # entity context
        self.ctx_A = ContextVector(embed_dim)   # attribute context
        self.ctx_R = ContextVector(embed_dim)   # relation/predicate context

    def forward(
        self,
        E: torch.Tensor,   # [nE, d]  entity embeddings in this graph
        A: torch.Tensor,   # [nA, d]  attribute value embeddings
        R: torch.Tensor,   # [nR, d]  time-aware predicate embeddings
    ) -> torch.Tensor:     # [d]      graph embedding g
        """
        g = avg(  Σᵢ f(eᵢᵀ c(E)) eᵢ
                 + Σⱼ f(aⱼᵀ c(A)) aⱼ
                 + Σₖ f(rₖᵀ c(R)) rₖ )          (Eq 15)
        """
        # ── top branch: compute context vectors (Eq 14)
        c_E = self.ctx_E(E)   # [d]
        c_A = self.ctx_A(A)   # [d]
        c_R = self.ctx_R(R)   # [d]

        # ── bottom branch: attention-weighted sums (sigmoid gate, Eq 15)
        agg_E = attention_aggregate(E, c_E)   # [d]
        agg_A = attention_aggregate(A, c_A)   # [d]
        agg_R = attention_aggregate(R, c_R)   # [d]

        # ── average across the three node types  (a() in Eq 15)
        # collect only non-zero terms (skip types with no nodes)
        parts = []
        if E.size(0) > 0:
            parts.append(agg_E)
        if A.size(0) > 0:
            parts.append(agg_A)
        if R.size(0) > 0:
            parts.append(agg_R)

        if not parts:
            return torch.zeros(self.embed_dim, dtype=E.dtype, device=E.device)

        g = torch.stack(parts, dim=0).mean(dim=0)   # [d]
        return g

    def embed_graph(
        self,
        entity_embs:  Optional[torch.Tensor] = None,   # [nE, d]
        attr_embs:    Optional[torch.Tensor] = None,   # [nA, d]
        rel_embs:     Optional[torch.Tensor] = None,   # [nR, d]
    ) -> torch.Tensor:                                 # [d]
        """
        Convenience wrapper that accepts optional tensors (None → empty).
        """
        d = self.embed_dim
        tensors = [x for x in [entity_embs, attr_embs, rel_embs] if x is not None]
        dev = tensors[0].device if tensors else torch.device("cpu")

        E = entity_embs if entity_embs is not None else torch.zeros(0, d, device=dev)
        A = attr_embs   if attr_embs   is not None else torch.zeros(0, d, device=dev)
        R = rel_embs    if rel_embs    is not None else torch.zeros(0, d, device=dev)
        return self.forward(E, A, R)


# ---------------------------------------------------------------------------
# Graph similarity  (Eq 16)
# ---------------------------------------------------------------------------

def graph_similarity(
    g1: torch.Tensor,   # [d]
    g2: torch.Tensor,   # [d]
) -> torch.Tensor:      # scalar
    """
    sim(g₁, g₂) = Σᵢ |g₁ᵢ − g₂ᵢ|      (Eq 16  — L1 / Manhattan distance)

    Lower value → graphs are more similar.
    """
    return (g1 - g2).abs().sum()


def batch_graph_similarity(
    g_query:     torch.Tensor,   # [d]
    g_candidates: torch.Tensor,  # [N, d]
) -> torch.Tensor:               # [N]
    """
    Vectorised Eq 16 for ranking a query graph against N candidate graphs.
    Lower score → higher similarity.
    """
    return (g_candidates - g_query.unsqueeze(0)).abs().sum(dim=-1)   # [N]


# ---------------------------------------------------------------------------
# Top-level approximate results generator
# ---------------------------------------------------------------------------

class ApproximateResultsGenerator:
    """
    Part 3 orchestrator: builds graph embeddings from vertex-level outputs
    (Part 1 & 2) and time-aware predicate embeddings (LSTM), then ranks
    candidate result graphs by Eq 16 similarity to the query graph.

    Typical call flow:
        1. embed_graph(entity_ids, attr_ids, rel_ids, time_start, time_end)
           → graph embedding tensor g  [d]
        2. rank_candidates(g_query, candidate_graphs)
           → sorted list of (similarity_score, candidate_index)
    """

    def __init__(
        self,
        graph_embedder:  GraphLevelEmbedding,
        entity_emb_table: torch.Tensor,   # [N_e, d] from ImprovedGCN
        attr_emb_table:   torch.Tensor,   # [N_a, d] from HyTEAttributeEmbedding
        rel_emb_table:    torch.Tensor,   # [N_r, d] time-aware (from TemporalPredicateEncoder)
        device: Optional[torch.device] = None,
    ) -> None:
        self.embedder      = graph_embedder
        self.entity_table  = entity_emb_table
        self.attr_table    = attr_emb_table
        self.rel_table     = rel_emb_table
        self.device        = device or torch.device("cpu")

    @torch.no_grad()
    def embed_graph(
        self,
        entity_ids: list[int],
        attr_ids:   list[int],
        rel_ids:    list[int],
    ) -> torch.Tensor:   # [d]
        """
        Build graph embedding g from the vertex-level embeddings of the
        entities, attributes, and (time-aware) relations in a result graph.
        """
        d   = self.embedder.embed_dim
        dev = self.device

        E = (self.entity_table[torch.tensor(entity_ids, device=dev)]
             if entity_ids else torch.zeros(0, d, device=dev))
        A = (self.attr_table[torch.tensor(attr_ids, device=dev)]
             if attr_ids   else torch.zeros(0, d, device=dev))
        R = (self.rel_table[torch.tensor(rel_ids, device=dev)]
             if rel_ids    else torch.zeros(0, d, device=dev))

        return self.embedder(E, A, R)   # [d]

    @torch.no_grad()
    def rank_candidates(
        self,
        g_query:     torch.Tensor,        # [d]
        candidate_graphs: list[tuple],    # each = (entity_ids, attr_ids, rel_ids)
        top_k: int = 20,
    ) -> list[tuple[float, int]]:
        """
        Rank candidates by Eq 16 similarity.
        Returns list of (sim_score, candidate_index) in ascending order.
        """
        if not candidate_graphs:
            return []

        g_cands = torch.stack(
            [self.embed_graph(*cg) for cg in candidate_graphs]
        )                                              # [C, d]
        scores = batch_graph_similarity(g_query, g_cands)   # [C]

        k = min(top_k, len(candidate_graphs))
        topk = torch.topk(scores, k=k, largest=False)
        return [
            (topk.values[i].item(), topk.indices[i].item())
            for i in range(k)
        ]
