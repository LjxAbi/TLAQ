"""
HyTE-based attribute embedding for TLAQ Part 2.

Mapping from HyTE (timestamp hyperplanes) to TLAQ (attribute-predicate hyperplanes):
  HyTE                        TLAQ
  ─────────────────────────   ──────────────────────────────────────
  timestamp τ                 attribute predicate ta
  hyperplane normal w_τ       normalized ta embedding (w_ta)
  entity triple (h,r,t,τ)     attribute triple <e, ta, a>
  P_τ(e) = e - (w_τ^T e)w_τ  P_ta(e) = e - (w_ta^T e) w_ta  (Eq 10)

Key formulae implemented (TLAQ Section 3.4):
  Eq  9  : ã  = e + ta                        (preliminary embedding)
  Eq 10  : eᵢ = eᵢ - ta^T eᵢ ta              (hyperplane projection)
  Eq 11  : atta(e,eᵢ) = sim(e,eᵢ)/Σ sim      (entity influence factor)
  Eq 12  : â  = Σ atta(e,eᵢ) · (eᵢ + ta)    (final attribute embedding)
  Eq 13  : AC(aᵥ,a) = Σᵢ|aᵥᵢ − aᵢ|           (attribute confidence / L1)

Training uses TransH-style scoring in the projected hyperplane space
(each attribute predicate defines its own hyperplane, as in HyTE for timestamps).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tkg_dataset import AttributeTriple, TKGDataset


# ---------------------------------------------------------------------------
# Hyperplane projection  (Eq 10 / HyTE P_τ formula)
# ---------------------------------------------------------------------------

def project_onto_hyperplane(
    v: torch.Tensor,          # [..., d]
    w: torch.Tensor,          # [..., d]  hyperplane normal (need not be unit)
) -> torch.Tensor:            # [..., d]
    """
    Project v onto the hyperplane orthogonal to w.
        P_w(v) = v - (ŵ^T v) ŵ       where ŵ = w / ||w||₂
    """
    w_hat = F.normalize(w, p=2, dim=-1)                     # unit normal
    coeff = (v * w_hat).sum(dim=-1, keepdim=True)           # ŵ^T v
    return v - coeff * w_hat                                # Eq 10


# ---------------------------------------------------------------------------
# HyTE-style scoring function
# ---------------------------------------------------------------------------

def hyte_attr_score(
    e_emb:  torch.Tensor,   # [B, d]  entity embedding
    ta_d:   torch.Tensor,   # [B, d]  translation vector for attr predicate
    a_emb:  torch.Tensor,   # [B, d]  attribute value embedding
    w_ta:   torch.Tensor,   # [B, d]  hyperplane normal for attr predicate
    norm: int = 1,
) -> torch.Tensor:           # [B]
    """
    TransH-style score in the attribute-predicate hyperplane.
        f(e, ta, a) = ||P_w(e) + d_ta - P_w(a)||ₙₒᵣₘ

    Using separate hyperplane normals (w_ta) and translation vectors (ta_d)
    follows TransH exactly, which HyTE is built upon.
    """
    p_e = project_onto_hyperplane(e_emb, w_ta)   # P_w(e)
    p_a = project_onto_hyperplane(a_emb, w_ta)   # P_w(a)
    return torch.norm(p_e + ta_d - p_a, p=norm, dim=-1)


# ---------------------------------------------------------------------------
# HyTE attribute embedding model
# ---------------------------------------------------------------------------

class HyTEAttributeEmbedding(nn.Module):
    """
    Learns embeddings for entities, attribute predicates, and attribute values
    via HyTE-style hyperplane projection training.

    Each attribute predicate ta has:
      - A translation vector d_ta  (analogous to relation embedding in TransE)
      - A hyperplane normal  w_ta  (analogous to w_τ in HyTE)

    At inference time, w_ta ≈ normalize(ta_embed) (we tie them for simplicity);
    the TLAQ paper uses ta directly as the hyperplane normal in Eq 10.
    """

    def __init__(
        self,
        dataset: TKGDataset,
        embed_dim: int = 128,
        l2_reg: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.embed_dim = embed_dim
        self.l2_reg = l2_reg
        self.device = device or torch.device("cpu")

        N   = dataset.num_entities
        NTA = dataset.num_attr_preds
        NA  = dataset.num_attributes

        # entity embeddings  (shared vocabulary with GCN or standalone)
        self.entity_emb  = nn.Embedding(N,   embed_dim)
        # attribute predicate: translation vector d_ta
        self.attr_pred_d = nn.Embedding(NTA, embed_dim)
        # attribute predicate: hyperplane normal w_ta  (normalized after each step)
        self.attr_pred_w = nn.Embedding(NTA, embed_dim)
        # attribute value embeddings
        self.attr_emb    = nn.Embedding(NA,  embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in [self.entity_emb, self.attr_pred_d, self.attr_pred_w, self.attr_emb]:
            nn.init.xavier_uniform_(emb.weight)
        # start with unit hyperplane normals
        with torch.no_grad():
            self.attr_pred_w.weight.data = F.normalize(
                self.attr_pred_w.weight.data, p=2, dim=-1
            )

    def normalize_hyperplanes(self) -> None:
        """Enforce ||w_ta||_2 = 1 after each SGD step (HyTE constraint)."""
        with torch.no_grad():
            self.attr_pred_w.weight.data = F.normalize(
                self.attr_pred_w.weight.data, p=2, dim=-1
            )

    def score(
        self,
        entity_ids:    torch.Tensor,   # [B]
        attr_pred_ids: torch.Tensor,   # [B]
        attr_ids:      torch.Tensor,   # [B]
    ) -> torch.Tensor:                 # [B]
        """HyTE score — lower is better for positive triples."""
        e  = self.entity_emb(entity_ids)     # [B, d]
        d  = self.attr_pred_d(attr_pred_ids) # [B, d]
        w  = self.attr_pred_w(attr_pred_ids) # [B, d]
        a  = self.attr_emb(attr_ids)         # [B, d]
        return hyte_attr_score(e, d, a, w)   # [B]

    def forward(
        self,
        entity_ids:    torch.Tensor,
        attr_pred_ids: torch.Tensor,
        attr_ids:      torch.Tensor,
    ) -> torch.Tensor:
        return self.score(entity_ids, attr_pred_ids, attr_ids)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_entity_emb(self, entity_id: int) -> torch.Tensor:
        """Return [d] embedding for a single entity."""
        idx = torch.tensor([entity_id], device=self.device)
        return self.entity_emb(idx).squeeze(0)

    def get_attr_pred_emb(self, attr_pred_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (d_ta [d], w_ta [d]) for a single attribute predicate."""
        idx = torch.tensor([attr_pred_id], device=self.device)
        return (
            self.attr_pred_d(idx).squeeze(0),
            self.attr_pred_w(idx).squeeze(0),
        )

    def all_attr_embeddings(self) -> torch.Tensor:
        """Return all attribute value embeddings [NA, d]."""
        return self.attr_emb.weight


# ---------------------------------------------------------------------------
# Preliminary attribute embedding  (Eq 9)
# ---------------------------------------------------------------------------

def preliminary_attr_embedding(
    entity_id:    int,
    attr_pred_id: int,
    model: HyTEAttributeEmbedding,
) -> torch.Tensor:   # [d]
    """
    ã = e + ta                                          (Eq 9)

    The TransE-style sum of the entity embedding and the attribute
    predicate translation vector gives the approximate position of
    the target attribute in embedding space.
    """
    e   = model.get_entity_emb(entity_id)            # [d]
    d_ta, _ = model.get_attr_pred_emb(attr_pred_id)  # [d]
    return e + d_ta


# ---------------------------------------------------------------------------
# Entity influence factor  (Eq 11)
# ---------------------------------------------------------------------------

def compute_atta(
    query_entity_id: int,
    neighbour_ids:   List[int],
    attr_pred_id:    int,
    model: HyTEAttributeEmbedding,
) -> Dict[int, float]:
    """
    For each adjacent entity eᵢ of the query entity e (in the attribute query):
      1. Project eᵢ onto the hyperplane of attribute predicate ta  (Eq 10)
      2. Measure cosine similarity between projected eᵢ and (original) e
      3. Normalise to sum-1 weights

        atta(e, eᵢ) = sim(e, eᵢ_proj) / Σⱼ sim(e, eⱼ_proj)      (Eq 11)

    Uses cosine similarity so the scale of embeddings does not bias results.
    """
    if not neighbour_ids:
        return {}

    _, w_ta = model.get_attr_pred_emb(attr_pred_id)   # [d]
    e_emb   = model.get_entity_emb(query_entity_id)   # [d]

    nbr_ids_t = torch.tensor(neighbour_ids, dtype=torch.long, device=model.device)
    nbr_embs  = model.entity_emb(nbr_ids_t)           # [M, d]

    # project each neighbour onto hyperplane of ta  (Eq 10)
    nbr_proj = project_onto_hyperplane(nbr_embs, w_ta.unsqueeze(0))  # [M, d]

    # cosine similarity between (original) e and projected neighbours
    e_norm   = F.normalize(e_emb.unsqueeze(0), p=2, dim=-1)          # [1, d]
    nbr_norm = F.normalize(nbr_proj, p=2, dim=-1)                    # [M, d]
    sims     = (e_norm * nbr_norm).sum(dim=-1)                        # [M]

    # shift to [0,∞) before normalising (cosine ∈ [-1,1])
    sims = sims + 1.0                                                  # [M] ≥ 0
    total = sims.sum().item()
    if total < 1e-9:
        total = 1.0

    return {nbr_id: (sims[i].item() / total) for i, nbr_id in enumerate(neighbour_ids)}


# ---------------------------------------------------------------------------
# Final attribute embedding  (Eq 12)
# ---------------------------------------------------------------------------

def compute_final_attr_embedding(
    query_entity_id: int,
    neighbour_ids:   List[int],
    attr_pred_id:    int,
    model: HyTEAttributeEmbedding,
) -> torch.Tensor:   # [d]
    """
    â = Σᵢ atta(e, eᵢ) · (eᵢ + ta)                              (Eq 12)

    Weighted average of the preliminary attribute embeddings of each
    adjacent entity.  Neighbours whose embeddings (projected onto ta's
    hyperplane) are most similar to e receive the largest weight.
    If there are no neighbours, fall back to the preliminary embedding
    of the query entity itself.
    """
    if not neighbour_ids:
        return preliminary_attr_embedding(query_entity_id, attr_pred_id, model)

    atta = compute_atta(query_entity_id, neighbour_ids, attr_pred_id, model)

    d_ta, _ = model.get_attr_pred_emb(attr_pred_id)               # [d]
    nbr_ids_t = torch.tensor(neighbour_ids, dtype=torch.long, device=model.device)
    nbr_embs  = model.entity_emb(nbr_ids_t)                       # [M, d]

    # preliminary embeddings of each neighbour: eᵢ + ta
    prelim = nbr_embs + d_ta.unsqueeze(0)                          # [M, d]

    weights = torch.tensor(
        [atta[nid] for nid in neighbour_ids],
        dtype=torch.float32,
        device=model.device,
    )                                                               # [M]
    a_hat = (weights.unsqueeze(1) * prelim).sum(dim=0)             # [d]  Eq 12
    return a_hat


# ---------------------------------------------------------------------------
# Attribute Confidence  (Eq 13)
# ---------------------------------------------------------------------------

def attribute_confidence(
    a_hat: torch.Tensor,   # [d]  final target attribute embedding
    A:     torch.Tensor,   # [N, d]  all attribute value embeddings
) -> torch.Tensor:          # [N]
    """
    AC(aᵥ, a) = Σᵢ |aᵥᵢ − aᵢ|       (Eq 13  —  Manhattan / L1 distance)

    Lower AC → attribute a is more similar to the target aᵥ.
    """
    diff = A - a_hat.unsqueeze(0)                  # [N, d]
    return diff.abs().sum(dim=-1)                  # [N]
