"""
Algorithm 1 — Approximate entity retrieval based on Relational Reliability.

Input : TKG, TQG (as list of entity-query triples)
Output: entity sorted result set RTR (top-k approximate triples)

Steps (paper Section 3.3):
  01-03  BFS over target-entity neighbours  → set E
  04     Preliminary GCN embedding of ev
  05-07  Compute attr influence factor for each ei in E
  08     Final embedding of ev   (Eq 7)
  09-11  RR distance for every candidate in TKG
  12-13  Sort and return top-k triples
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch

from src.data.tkg_dataset import RelationTriple, TKGDataset
from src.models.improved_gcn import (
    ImprovedGCN,
    compute_attr,
    compute_final_entity_embedding,
    relational_reliability,
)


# ---------------------------------------------------------------------------
# BFS neighbour collector  (lines 01-03)
# ---------------------------------------------------------------------------

def bfs_neighbours(
    root: int,
    dataset: TKGDataset,
    max_depth: int = 2,
) -> Set[int]:
    """
    Breadth-first traversal starting from `root`.
    Returns all entity ids reachable within `max_depth` hops
    (both head-to-tail and tail-to-head directions).
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

    visited.discard(root)   # the target entity itself is not a candidate
    return visited


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def approximate_entity_query(
    tkg: TKGDataset,
    query_triples: List[Tuple[int, int, int]],   # (ev, tr, et)  from TQG RTQ
    gcn_model: ImprovedGCN,
    top_k: int = 20,
    bfs_depth: int = 2,
    device: Optional[torch.device] = None,
) -> List[Tuple[float, RelationTriple]]:
    """
    Parameters
    ----------
    tkg           : the full temporal knowledge graph
    query_triples : entity-query triples (ev, tr, et) from the TQG.
                    ev is the target entity (to be approximated).
    gcn_model     : trained ImprovedGCN instance
    top_k         : number of approximate results to return
    bfs_depth     : BFS hop depth for neighbour collection

    Returns
    -------
    List of (rr_score, RelationTriple) sorted by ascending RR distance.
    Lower RR → more semantically similar to the query.
    """
    if device is None:
        device = next(gcn_model.parameters()).device

    gcn_model.eval()
    with torch.no_grad():
        H = gcn_model().to(device)   # [N, d]  all entity embeddings  (line 04)

    # collect all target entities appearing in the query
    target_entities: Set[int] = set()
    for (ev, tr, et) in query_triples:
        target_entities.add(ev)

    # --- lines 01-03: BFS for each target entity ---
    E_all: Set[int] = set()
    for ev in target_entities:
        nbrs = bfs_neighbours(ev, tkg, bfs_depth)
        E_all |= nbrs

    results: List[Tuple[float, RelationTriple]] = []

    for ev in target_entities:
        # triples in RTQ that contain ev
        ev_triples = [(e, r, t) for (e, r, t) in query_triples if e == ev]

        # --- lines 05-07: attr for each neighbour entity in E ---
        # (attr is defined per query triple, not per neighbour — Eq 6)
        attr_scores = compute_attr(ev, ev_triples, tkg)

        # --- line 08: final embedding of ev (Eq 7) ---
        ev_emb = compute_final_entity_embedding(ev, ev_triples, H, tkg)  # [d]

        # --- lines 09-11: RR distance over all TKG relation triples ---
        # Collect candidate triples: any triple in TKG whose head or tail
        # could replace ev.
        candidate_triples: List[RelationTriple] = []
        candidate_embs: List[torch.Tensor] = []

        for triple in tkg.relation_triples:
            # Candidate: tail entity in a head-side query  <?ev, tr, et>
            candidate_triples.append(triple)
            candidate_embs.append(H[triple.tail])

        if not candidate_embs:
            continue

        E_matrix = torch.stack(candidate_embs, dim=0)          # [M, d]
        rr_scores = relational_reliability(ev_emb, E_matrix)   # [M]  Eq 8

        # --- lines 12-13: sort by RR and collect top-k ---
        k = min(top_k, len(candidate_triples))
        topk_indices = torch.topk(rr_scores, k=k, largest=False).indices.tolist()

        for idx in topk_indices:
            results.append((rr_scores[idx].item(), candidate_triples[idx]))

    # global sort across all target entities and deduplicate
    results.sort(key=lambda x: x[0])
    seen: Set[int] = set()
    unique: List[Tuple[float, RelationTriple]] = []
    for score, triple in results:
        tid = id(triple)
        if tid not in seen:
            seen.add(tid)
            unique.append((score, triple))
        if len(unique) >= top_k:
            break

    return unique[:top_k]
