"""
TLAQ end-to-end pipeline  (Section 3, Figure 2).

Orchestrates the three-part framework:

  Part 1  ImprovedGCN          → entity embeddings + Algorithm 1
  Part 2  HyTEAttributeEmbedding → attribute embeddings + Algorithm 2
  Part 3  TemporalPredicateEncoder + GraphLevelEmbedding
             → time-aware predicate embeddings + graph-similarity ranking

Query types
-----------
  Entity query   : (head, relation, ?, time_start, time_end)
                   "Which entities are connected to head via relation at time t?"
  Attribute query: (entity, attr_pred, ?)
                   "What is the value of attr_pred for entity?"
  Graph query    : full graph pattern → rank candidate result graphs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.data.tkg_dataset import TKGDataset, RelationTriple, AttributeTriple
from src.models.improved_gcn import ImprovedGCN
from src.models.hyte_attribute import HyTEAttributeEmbedding
from src.models.temporal_predicate_encoder import TemporalPredicateEncoder
from src.models.graph_level_embedding import (
    GraphLevelEmbedding,
    ApproximateResultsGenerator,
)
from src.algorithms.algo1_entity_query import approximate_entity_query
from src.algorithms.algo2_attribute_query import approximate_attribute_query
from src.utils.time_encoding import encode_interval, encode_timestamp, TIME_DIM


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TLAQConfig:
    embed_dim:      int   = 128
    gcn_layers:     int   = 2
    lstm_layers:    int   = 1
    lstm_dropout:   float = 0.1
    top_k:          int   = 20
    bfs_depth:      int   = 2
    use_interval:   bool  = True    # encode time as interval (104 bits) vs point (52)
    device:         str   = "cpu"


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class TLAQPipeline:
    """
    Unified TLAQ pipeline.  Accepts pre-trained model components; training
    each component separately is handled by GCNTrainer / HyTETrainer.

    Parameters
    ----------
    dataset   : TKGDataset  (with build_indices() already called)
    gcn       : trained ImprovedGCN
    hyte      : trained HyTEAttributeEmbedding
    encoder   : trained TemporalPredicateEncoder
    embedder  : GraphLevelEmbedding
    config    : TLAQConfig
    """

    def __init__(
        self,
        dataset:  TKGDataset,
        gcn:      ImprovedGCN,
        hyte:     HyTEAttributeEmbedding,
        encoder:  TemporalPredicateEncoder,
        embedder: GraphLevelEmbedding,
        config:   Optional[TLAQConfig] = None,
    ) -> None:
        self.dataset  = dataset
        self.gcn      = gcn
        self.hyte     = hyte
        self.encoder  = encoder
        self.embedder = embedder
        self.cfg      = config or TLAQConfig()
        self.device   = torch.device(self.cfg.device)

        # build time-aware relation embedding table once
        self._rel_emb_table = self._build_relation_emb_table()

        # top-level graph ranker
        self._generator = ApproximateResultsGenerator(
            graph_embedder   = self.embedder,
            entity_emb_table = self.gcn.entity_emb.weight.data,
            attr_emb_table   = self.hyte.attr_emb.weight.data,
            rel_emb_table    = self._rel_emb_table,
            device           = self.device,
        )

    # ------------------------------------------------------------------
    # Build time-aware relation embedding table (Part 3)
    # ------------------------------------------------------------------

    def _build_relation_emb_table(self) -> torch.Tensor:
        """
        For every relation id, compute the time-aware predicate embedding
        by averaging LSTM outputs over all triples that use that relation.
        If a relation has no triples, its embedding is the raw GCN output.
        """
        d       = self.cfg.embed_dim
        n_rels  = self.dataset.num_relations
        time_dim = TIME_DIM * 2 if self.cfg.use_interval else TIME_DIM

        # accumulator: sum of LSTM outputs + count
        sums   = torch.zeros(n_rels, d)
        counts = torch.zeros(n_rels)

        self.encoder.eval()
        with torch.no_grad():
            for triple in self.dataset.relation_triples:
                rid = triple.relation

                # base predicate embedding: use entity difference as proxy
                # (in a fully trained system this would come from a dedicated
                #  relation embedding table; here we derive it from the GCN)
                h_emb = self.gcn.entity_emb.weight.data[triple.head]   # [d]
                t_emb = self.gcn.entity_emb.weight.data[triple.tail]   # [d]
                pred_emb = (h_emb - t_emb).unsqueeze(0)                # [1, d]

                # time encoding
                if self.cfg.use_interval and triple.time_end is not None:
                    time_vec = encode_interval(
                        triple.time_start, triple.time_end
                    ).unsqueeze(0)
                elif triple.time_start is not None:
                    ts = triple.time_start
                    base = encode_timestamp(ts)
                    if self.cfg.use_interval:
                        time_vec = torch.cat([base, base], dim=0).unsqueeze(0)
                    else:
                        time_vec = base.unsqueeze(0)
                else:
                    time_vec = torch.zeros(1, time_dim)

                r_emb = self.encoder(pred_emb, time_vec).squeeze(0)   # [d]
                sums[rid]   += r_emb
                counts[rid] += 1

        # fallback for unseen relations: use zero vector
        mask          = counts > 0
        result        = torch.zeros(n_rels, d)
        result[mask]  = sums[mask] / counts[mask].unsqueeze(1)
        return result

    # ------------------------------------------------------------------
    # Query entry points
    # ------------------------------------------------------------------

    def entity_query(
        self,
        query_triples: list[tuple],   # (head_id, relation_id, -1, time_start, time_end)
        top_k: Optional[int] = None,
    ) -> list[tuple[float, RelationTriple]]:
        """
        Algorithm 1: approximate entity query.
        Returns list of (rr_score, RelationTriple) sorted ascending by score.
        """
        k = top_k or self.cfg.top_k
        # strip time fields — algo1 accepts (head, relation, tail_placeholder)
        triples_3 = [(h, r, t) for h, r, t, *_ in query_triples]
        return approximate_entity_query(
            self.dataset, triples_3, self.gcn,
            top_k=k, bfs_depth=self.cfg.bfs_depth,
        )

    def attribute_query(
        self,
        query_triples: list[tuple],   # (entity_id, attr_pred_id, -1)
        top_k: Optional[int] = None,
    ) -> list[tuple[float, AttributeTriple]]:
        """
        Algorithm 2: approximate attribute query.
        Returns list of (ac_score, AttributeTriple) sorted ascending by score.
        """
        k = top_k or self.cfg.top_k
        return approximate_attribute_query(
            self.dataset, query_triples, self.hyte,
            top_k=k, bfs_depth=self.cfg.bfs_depth,
        )

    def graph_query(
        self,
        query_entity_ids: list[int],
        query_attr_ids:   list[int],
        query_rel_ids:    list[int],
        candidate_graphs: list[tuple[list[int], list[int], list[int]]],
        top_k: Optional[int] = None,
    ) -> list[tuple[float, int]]:
        """
        Part 3: embed query graph, rank candidate graphs by Eq 16 similarity.

        Parameters
        ----------
        query_entity_ids  : entity ids in the query graph
        query_attr_ids    : attribute value ids in the query graph
        query_rel_ids     : relation/predicate ids in the query graph
        candidate_graphs  : list of (entity_ids, attr_ids, rel_ids) per candidate
        top_k             : number of results to return

        Returns
        -------
        list of (similarity_score, candidate_index) ascending by score
        """
        k       = top_k or self.cfg.top_k
        g_query = self._generator.embed_graph(
            query_entity_ids, query_attr_ids, query_rel_ids
        )
        return self._generator.rank_candidates(g_query, candidate_graphs, top_k=k)

    # ------------------------------------------------------------------
    # Convenience: extract ranked ids for evaluation
    # ------------------------------------------------------------------

    def entity_query_ids(
        self,
        query_triples: list[tuple],
        top_k: Optional[int] = None,
    ) -> list[int]:
        """Return unique tail entity ids from entity_query, ranked by RR score."""
        results = self.entity_query(query_triples, top_k)
        seen: set[int] = set()
        ids: list[int] = []
        for _, triple in results:
            if triple.tail not in seen:
                seen.add(triple.tail)
                ids.append(triple.tail)
        return ids

    def attribute_query_ids(
        self,
        query_triples: list[tuple],
        top_k: Optional[int] = None,
    ) -> list[int]:
        """Return unique attribute ids from attribute_query, ranked by AC score."""
        results = self.attribute_query(query_triples, top_k)
        seen: set[int] = set()
        ids: list[int] = []
        for _, triple in results:
            if triple.attribute not in seen:
                seen.add(triple.attribute)
                ids.append(triple.attribute)
        return ids


# ---------------------------------------------------------------------------
# Factory: build pipeline from scratch (random init — for smoke tests)
# ---------------------------------------------------------------------------

def build_pipeline(
    dataset: TKGDataset,
    config:  Optional[TLAQConfig] = None,
) -> TLAQPipeline:
    """
    Construct a TLAQPipeline with randomly initialised weights.
    Useful for integration tests and as a starting point before training.
    """
    cfg = config or TLAQConfig()
    d   = cfg.embed_dim
    dev = torch.device(cfg.device)
    time_dim = TIME_DIM * 2 if cfg.use_interval else TIME_DIM

    gcn = ImprovedGCN(
        dataset   = dataset,
        embed_dim = d,
        num_layers= cfg.gcn_layers,
        device    = dev,
    )
    hyte = HyTEAttributeEmbedding(dataset=dataset, embed_dim=d)
    encoder = TemporalPredicateEncoder(
        embed_dim  = d,
        time_dim   = time_dim,
        num_layers = cfg.lstm_layers,
        dropout    = cfg.lstm_dropout,
    )
    embedder = GraphLevelEmbedding(embed_dim=d)

    return TLAQPipeline(dataset, gcn, hyte, encoder, embedder, cfg)
