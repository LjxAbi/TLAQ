"""
Algorithm 2 — Approximate attribute retrieval based on Attribute Confidence.

Input : TKG, TQG (as list of attribute-query triples)
Output: attribute sorted result set ATR (top-k approximate triples)

Steps (paper Section 3.4):
  01-03  BFS over entity e's neighbours  → set A
  04     Preliminary embedding of target attribute av
  05-07  Compute atta influence factor for each entity eᵢ in A
  08     Final embedding of av   (Eq 12)
  09-11  AC distance for every candidate attribute triple in TKG
  12-13  Sort and return top-k triples
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch

from src.data.tkg_dataset import AttributeTriple, TKGDataset
from src.models.hyte_attribute import (
    HyTEAttributeEmbedding,
    attribute_confidence,
    compute_atta,
    compute_final_attr_embedding,
    preliminary_attr_embedding,
)


# ---------------------------------------------------------------------------
# BFS (reuses same logic as Algorithm 1; kept local for clarity)
# ---------------------------------------------------------------------------

def bfs_entity_neighbours(
    root: int,
    dataset: TKGDataset,
    max_depth: int = 2,
) -> Set[int]:
    """
    BFS over RELATION edges from root entity.
    Returns all reachable entity ids within max_depth hops (root excluded).
    """
    visited: Set[int] = {root}
    queue: deque[Tuple[int, int]] = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nbr in dataset.neighbours(node):
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, depth + 1))
    visited.discard(root)
    return visited


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------

def approximate_attribute_query(
    tkg: TKGDataset,
    query_triples: List[Tuple[int, int, int]],   # (e, ta, av)  from TQG ATQ
    model: HyTEAttributeEmbedding,
    top_k: int = 20,
    bfs_depth: int = 2,
    device: Optional[torch.device] = None,
) -> List[Tuple[float, AttributeTriple]]:
    """
    Parameters
    ----------
    tkg           : the full temporal knowledge graph
    query_triples : attribute-query triples (e, ta, av) from the TQG.
                    av is the target attribute (to be approximated).
    model         : trained HyTEAttributeEmbedding instance
    top_k         : number of approximate results to return
    bfs_depth     : BFS hop depth for neighbour collection

    Returns
    -------
    List of (ac_score, AttributeTriple) sorted by ascending AC distance.
    Lower AC → more semantically similar to the query attribute.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    A_all_embs: torch.Tensor = model.all_attr_embeddings().detach()   # [NA, d]

    results: List[Tuple[float, AttributeTriple]] = []

    for (e, ta, _av) in query_triples:

        # --- lines 01-03: BFS to collect neighbouring entities ---
        neighbour_ids: List[int] = sorted(
            bfs_entity_neighbours(e, tkg, bfs_depth)
        )                                                              # list of int

        with torch.no_grad():
            # --- line 04: preliminary embedding of target attribute ---
            a_prelim = preliminary_attr_embedding(e, ta, model)       # [d]  Eq 9

            # --- lines 05-07: atta for each neighbour entity ---
            # (atta is computed inside compute_final_attr_embedding)

            # --- line 08: final embedding of target attribute ---
            a_hat = compute_final_attr_embedding(
                e, neighbour_ids, ta, model
            )                                                          # [d]  Eq 12

            # --- lines 09-11: AC distance over all TKG attribute triples ---
            # Collect candidate attribute triples for the same predicate ta
            candidate_triples: List[AttributeTriple] = [
                t for t in tkg.attribute_triples if t.attr_pred == ta
            ]

            if not candidate_triples:
                # broaden search to all attribute triples
                candidate_triples = tkg.attribute_triples

            if not candidate_triples:
                continue

            cand_attr_ids = torch.tensor(
                [t.attribute for t in candidate_triples],
                dtype=torch.long,
                device=device,
            )                                                          # [M]
            cand_embs = A_all_embs[cand_attr_ids]                     # [M, d]
            ac_scores = attribute_confidence(a_hat, cand_embs)        # [M]  Eq 13

        # --- lines 12-13: sort by AC and collect top-k ---
        k = min(top_k, len(candidate_triples))
        topk_idx = torch.topk(ac_scores, k=k, largest=False).indices.tolist()

        for idx in topk_idx:
            results.append((ac_scores[idx].item(), candidate_triples[idx]))

    # global sort across all query triples, deduplicate by (entity, attr_pred, attribute)
    results.sort(key=lambda x: x[0])
    seen: Set[Tuple[int, int, int]] = set()
    unique: List[Tuple[float, AttributeTriple]] = []
    for score, triple in results:
        key = (triple.entity, triple.attr_pred, triple.attribute)
        if key not in seen:
            seen.add(key)
            unique.append((score, triple))
        if len(unique) >= top_k:
            break

    return unique[:top_k]
