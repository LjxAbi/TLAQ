"""
Minimal SPARQL pattern extractor for TLAQ evaluation.

Parses QALD-6 / QALD-7 / LC-QuAD SPARQL queries and extracts the
single most useful (anchor_entity, predicate) pair that can be fed
to TLAQPipeline.entity_query().

Supported extraction targets
-----------------------------
forward  : { <S> <P> ?uri }  →  entity_query(head=S, relation=P)
reverse  : { ?uri <P> <O> }  →  entity_query_reverse(tail=O, relation=P)

For multi-triple WHERE clauses the function walks every triple pattern
and returns the first forward/reverse hit (forward preferred).

Returns None for ASK, COUNT, or queries where no DBpedia resource
anchor could be identified.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class SPARQLPattern:
    """
    A single extractable (entity, predicate, direction) pattern.

    Attributes
    ----------
    mode     : "forward" — entity is the subject; ?uri is the object
               "reverse" — entity is the object; ?uri is the subject
    entity   : local name of the DBpedia resource anchor
    relation : local name of the predicate
    """
    __slots__ = ("mode", "entity", "relation")

    def __init__(self, mode: str, entity: str, relation: str) -> None:
        self.mode     = mode
        self.entity   = entity
        self.relation = relation

    def __repr__(self) -> str:  # pragma: no cover
        return f"SPARQLPattern({self.mode!r}, entity={self.entity!r}, relation={self.relation!r})"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_PREFIX_RE  = re.compile(r'PREFIX\s+(\w+):\s*<([^>]+)>', re.IGNORECASE)
_TOKEN_RE   = re.compile(r'(<[^>]+>|[\w][\w./\-()]*:[\w./\-()#%]+|\?[A-Za-z_][A-Za-z_0-9]*)')
_VARNAMES   = {"?uri", "?date", "?num", "?n", "?d", "?answer", "?ans"}


def extract_pattern(sparql: str) -> Optional[SPARQLPattern]:
    """
    Extract the dominant 1-hop pattern from a SPARQL query string.

    Returns a SPARQLPattern or None if the query is unsupported.
    """
    if not sparql or not sparql.strip():
        return None

    # --- expand PREFIX macros ---
    prefixes: dict[str, str] = {}
    for m in _PREFIX_RE.finditer(sparql):
        prefixes[m.group(1)] = m.group(2)

    def expand(tok: str) -> Optional[str]:
        tok = tok.strip()
        if tok.startswith("<") and tok.endswith(">"):
            return tok[1:-1]
        if ":" in tok:
            pfx, local = tok.split(":", 1)
            if pfx in prefixes:
                return prefixes[pfx] + local
        return None

    def is_variable(tok: str) -> bool:
        return tok.startswith("?")

    def is_dbr(uri: str) -> bool:
        return "dbpedia.org/resource/" in uri

    def local(uri: str) -> str:
        if "#" in uri:
            return uri.rsplit("#", 1)[-1]
        return uri.rsplit("/", 1)[-1]

    # --- tokenize and slide a 3-token window ---
    tokens = _TOKEN_RE.findall(sparql)

    forward_hit:  Optional[SPARQLPattern] = None
    reverse_hit:  Optional[SPARQLPattern] = None

    for i in range(len(tokens) - 2):
        s, p, o = tokens[i], tokens[i + 1], tokens[i + 2]

        # skip if any token looks like a keyword
        if any(t.upper() in ("SELECT", "WHERE", "FILTER", "UNION", "OPTIONAL",
                              "DISTINCT", "LIMIT", "PREFIX", "ORDER", "ASK")
               for t in (s, p, o)):
            continue

        # forward: <S> <P> ?uri
        if not is_variable(s) and not is_variable(p) and o.lower() in _VARNAMES:
            se = expand(s)
            pe = expand(p)
            if se and pe and is_dbr(se) and forward_hit is None:
                s_loc = local(se)
                p_loc = local(pe)
                if s_loc and p_loc:   # skip namespace roots (trailing / or #)
                    forward_hit = SPARQLPattern("forward", s_loc, p_loc)

        # reverse: ?uri <P> <O>
        elif s.lower() in _VARNAMES and not is_variable(p) and not is_variable(o):
            pe = expand(p)
            oe = expand(o)
            if pe and oe and is_dbr(oe) and reverse_hit is None:
                o_loc = local(oe)
                p_loc = local(pe)
                if o_loc and p_loc:
                    reverse_hit = SPARQLPattern("reverse", o_loc, p_loc)

    # prefer forward over reverse
    return forward_hit or reverse_hit


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def extract_patterns(sparql_list: list[str]) -> list[Optional[SPARQLPattern]]:
    """Extract patterns for a list of SPARQL strings."""
    return [extract_pattern(s) for s in sparql_list]
