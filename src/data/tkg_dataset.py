"""
TKG data structures and loader.
Stores relation triples <eh, tr, et> and attribute triples <e, ta, a>.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class RelationTriple:
    head: int       # entity id
    relation: int   # relation predicate id
    tail: int       # entity id
    time_start: Optional[int] = None
    time_end: Optional[int] = None


@dataclass
class AttributeTriple:
    entity: int         # entity id
    attr_pred: int      # attribute predicate id
    attribute: int      # attribute value id
    time_start: Optional[int] = None
    time_end: Optional[int] = None


class TKGDataset:
    """
    Temporal Knowledge Graph container.

    Maintains forward and inverse indices over relation triples for fast
    per-entity neighbour lookup, and precomputes the statistics needed by
    the improved adjacency matrix (head_count, tail_count, nums_r).
    """

    def __init__(self) -> None:
        # --- vocabularies ---
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        self.attr_pred2id: Dict[str, int] = {}
        self.id2attr_pred: Dict[int, str] = {}
        self.attribute2id: Dict[str, int] = {}
        self.id2attribute: Dict[int, str] = {}

        # --- triple stores ---
        self.relation_triples: List[RelationTriple] = []
        self.attribute_triples: List[AttributeTriple] = []

        # --- adjacency indices (built lazily via build_indices) ---
        # head_triples[e] = list of RelationTriple where head == e
        self.head_triples: Dict[int, List[RelationTriple]] = defaultdict(list)
        # tail_triples[e] = list of RelationTriple where tail == e
        self.tail_triples: Dict[int, List[RelationTriple]] = defaultdict(list)
        # relation_triples_by_r[r] = all triples with relation r
        self.relation_triples_by_r: Dict[int, List[RelationTriple]] = defaultdict(list)

        # --- per-entity statistics ---
        # head_count[e] = number of triples where e is head
        self.head_count: Dict[int, int] = defaultdict(int)
        # tail_count[e] = number of triples where e is tail
        self.tail_count: Dict[int, int] = defaultdict(int)
        # nums_r[r] = total triples with relation r
        self.nums_r: Dict[int, int] = defaultdict(int)
        # entity_in_r[(e,r)] = count of triples containing e with relation r
        self.entity_in_r: Dict[Tuple[int, int], int] = defaultdict(int)

        self._indices_built = False

    # ------------------------------------------------------------------
    # vocabulary helpers
    # ------------------------------------------------------------------

    def _get_or_add(self, vocab: Dict, inv_vocab: Dict, name: str) -> int:
        if name not in vocab:
            idx = len(vocab)
            vocab[name] = idx
            inv_vocab[idx] = name
        return vocab[name]

    def add_entity(self, name: str) -> int:
        return self._get_or_add(self.entity2id, self.id2entity, name)

    def add_relation(self, name: str) -> int:
        return self._get_or_add(self.relation2id, self.id2relation, name)

    def add_attr_pred(self, name: str) -> int:
        return self._get_or_add(self.attr_pred2id, self.id2attr_pred, name)

    def add_attribute(self, name: str) -> int:
        return self._get_or_add(self.attribute2id, self.id2attribute, name)

    # ------------------------------------------------------------------
    # triple addition
    # ------------------------------------------------------------------

    def add_relation_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> None:
        h = self.add_entity(head)
        r = self.add_relation(relation)
        t = self.add_entity(tail)
        self.relation_triples.append(RelationTriple(h, r, t, time_start, time_end))

    def add_attribute_triple(
        self,
        entity: str,
        attr_pred: str,
        attribute: str,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> None:
        e = self.add_entity(entity)
        ta = self.add_attr_pred(attr_pred)
        a = self.add_attribute(attribute)
        self.attribute_triples.append(AttributeTriple(e, ta, a, time_start, time_end))

    # ------------------------------------------------------------------
    # index construction (call once after all triples are loaded)
    # ------------------------------------------------------------------

    def build_indices(self) -> None:
        self.head_triples.clear()
        self.tail_triples.clear()
        self.relation_triples_by_r.clear()
        self.head_count.clear()
        self.tail_count.clear()
        self.nums_r.clear()
        self.entity_in_r.clear()

        for triple in self.relation_triples:
            h, r, t = triple.head, triple.relation, triple.tail
            self.head_triples[h].append(triple)
            self.tail_triples[t].append(triple)
            self.relation_triples_by_r[r].append(triple)
            self.head_count[h] += 1
            self.tail_count[t] += 1
            self.nums_r[r] += 1
            self.entity_in_r[(h, r)] += 1
            self.entity_in_r[(t, r)] += 1

        self._indices_built = True

    # ------------------------------------------------------------------
    # size properties
    # ------------------------------------------------------------------

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    @property
    def num_attr_preds(self) -> int:
        return len(self.attr_pred2id)

    @property
    def num_attributes(self) -> int:
        return len(self.attribute2id)

    def neighbours(self, entity_id: int) -> Set[int]:
        """All entities adjacent to entity_id (head or tail side)."""
        nbrs: Set[int] = set()
        for t in self.head_triples.get(entity_id, []):
            nbrs.add(t.tail)
        for t in self.tail_triples.get(entity_id, []):
            nbrs.add(t.head)
        return nbrs

    def __repr__(self) -> str:
        return (
            f"TKGDataset("
            f"entities={self.num_entities}, "
            f"relations={self.num_relations}, "
            f"rel_triples={len(self.relation_triples)}, "
            f"attr_triples={len(self.attribute_triples)})"
        )
