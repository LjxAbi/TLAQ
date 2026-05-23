"""
Training loop for HyTEAttributeEmbedding.

Loss (HyTE paper Section 4.2):
    L = Σ_τ Σ_{x∈D+_τ} Σ_{y∈D-_τ} max(0, f(x) - f(y) + γ)

Adapted for attribute triples <e, ta, a>:
    L = Σ_{(e,ta,a)∈AT} max(0, f(e,ta,a) - f(e',ta,a') + γ)

Two negative-sampling strategies from HyTE:
  TANS (time-agnostic  → here: predicate-agnostic)  : corrupt entity or attribute value
  TDNS (time-dependent → here: predicate-dependent) : corrupt with triples that exist in
        the TKG but are NOT valid for the current attribute predicate
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.data.tkg_dataset import AttributeTriple, TKGDataset
from src.models.hyte_attribute import HyTEAttributeEmbedding


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def _tans_corrupt(
    triple: AttributeTriple,
    num_entities: int,
    num_attributes: int,
) -> Tuple[int, int, int]:
    """
    Time-Agnostic Negative Sampling (TANS, Eq 1 in HyTE).
    Randomly corrupt the entity or the attribute value.
    """
    if random.random() < 0.5:
        return (random.randint(0, num_entities - 1), triple.attr_pred, triple.attribute)
    else:
        return (triple.entity, triple.attr_pred, random.randint(0, num_attributes - 1))


def _tdns_corrupt(
    triple: AttributeTriple,
    dataset: TKGDataset,
    num_entities: int,
    num_attributes: int,
    max_tries: int = 10,
) -> Tuple[int, int, int]:
    """
    Predicate-Dependent Negative Sampling (analogous to TDNS, Eq 2 in HyTE).
    Prefers corrupted triples that exist in the TKG but with a DIFFERENT
    attribute predicate, pushing predicates apart in hyperplane space.
    Falls back to TANS if no such triple is found quickly.
    """
    ta = triple.attr_pred
    for _ in range(max_tries):
        # pick a triple that uses a different attribute predicate for the same entity
        candidates = [
            t for t in dataset.attribute_triples
            if t.entity == triple.entity and t.attr_pred != ta
        ]
        if candidates:
            c = random.choice(candidates)
            # inject this entity/attr from the "wrong" predicate as negative
            return (c.entity, ta, c.attribute)
    # fallback
    return _tans_corrupt(triple, num_entities, num_attributes)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HyTETrainer:
    """
    Trains HyTEAttributeEmbedding with margin-based ranking loss.

    After each parameter update, the hyperplane normals w_ta are
    re-normalised to satisfy ||w_ta||_2 = 1  (HyTE constraint).
    Entity embeddings are L2-regularised (||ep||_2 ≤ 1, enforced via
    the regularisation term in the loss rather than hard clipping, for
    gradient compatibility with PyTorch).
    """

    def __init__(
        self,
        model: HyTEAttributeEmbedding,
        dataset: TKGDataset,
        margin: float = 1.0,
        lr: float = 1e-3,
        negative_sampling: str = "tans",   # "tans" | "tdns"
        device: Optional[torch.device] = None,
    ) -> None:
        assert negative_sampling in ("tans", "tdns"), \
            "negative_sampling must be 'tans' or 'tdns'"
        self.model = model
        self.dataset = dataset
        self.margin = margin
        self.ns_mode = negative_sampling
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def _sample_batch(
        self,
        triples: List[AttributeTriple],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pos_e, pos_ta, pos_a, neg_e, neg_ta, neg_a) tensors [B]."""
        batch = random.sample(triples, min(batch_size, len(triples)))
        N, NA = self.dataset.num_entities, self.dataset.num_attributes

        pe, pta, pa, ne, nta, na = [], [], [], [], [], []
        for t in batch:
            pe.append(t.entity); pta.append(t.attr_pred); pa.append(t.attribute)
            if self.ns_mode == "tdns":
                ne_, nta_, na_ = _tdns_corrupt(t, self.dataset, N, NA)
            else:
                ne_, nta_, na_ = _tans_corrupt(t, N, NA)
            ne.append(ne_); nta.append(nta_); na.append(na_)

        to_t = lambda lst: torch.tensor(lst, dtype=torch.long, device=self.device)
        return to_t(pe), to_t(pta), to_t(pa), to_t(ne), to_t(nta), to_t(na)

    def train_epoch(self, batch_size: int = 256) -> float:
        """Run one training epoch; return mean loss."""
        self.model.train()
        triples = self.dataset.attribute_triples
        if not triples:
            return 0.0

        total_loss = 0.0
        num_batches = max(len(triples) // batch_size, 1)

        for _ in range(num_batches):
            self.optimizer.zero_grad()

            pe, pta, pa, ne, nta, na = self._sample_batch(triples, batch_size)
            pos_score = self.model.score(pe, pta, pa)    # [B]
            neg_score = self.model.score(ne, nta, na)    # [B]

            # margin ranking loss
            margin_loss = torch.clamp(self.margin + pos_score - neg_score, min=0.0).mean()

            # L2 regularisation on entity embeddings (||ep||₂ ≤ 1)
            e_norms = self.model.entity_emb.weight.norm(p=2, dim=-1)
            reg_loss = self.model.l2_reg * torch.clamp(e_norms - 1.0, min=0.0).mean()

            loss = margin_loss + reg_loss
            loss.backward()
            self.optimizer.step()

            # re-normalise hyperplane normals after each update (HyTE constraint)
            self.model.normalize_hyperplanes()

            total_loss += loss.item()

        return total_loss / num_batches

    def fit(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        log_every: int = 10,
    ) -> List[float]:
        losses = []
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(batch_size)
            losses.append(loss)
            if epoch % log_every == 0:
                print(f"Epoch {epoch:4d}/{epochs}  attr-loss={loss:.4f}")
        return losses
