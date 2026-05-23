"""
TKG dataset loaders for TLAQ benchmarks.

Supported formats
-----------------
load_tsv(path)
    Tab-separated triples (3/4/5 columns, optional time fields).

load_turtle(path)
    DBpedia Turtle (.ttl) or compressed (.ttl.bz2) via rdflib.
    Handles @prefix declarations automatically.
    Supports streaming bzip2 decompression — no manual unzip needed.

load_ntriples(path)
    W3C N-Triples (.nt) — lightweight regex parser for simple files.

load_qald(path)
    QALD-6 / QALD-7 multilingual JSON format.

load_lcquad(path)
    LC-QuAD 1.0 JSON format (AskNowQA/LC-QuAD).

All loaders return a TKGDataset (or QALDDataset) with build_indices() called.
"""
from __future__ import annotations

import bz2
import csv
import json
import re
from pathlib import Path
from typing import Optional

from src.data.tkg_dataset import TKGDataset


# ---------------------------------------------------------------------------
# TSV loader  (ICEWS / YAGO / Freebase subset style)
# ---------------------------------------------------------------------------

def load_tsv(
    path: str | Path,
    delimiter: str = "\t",
    encoding: str = "utf-8",
) -> TKGDataset:
    """
    Load a temporal KG from a tab-separated file.

    Column layouts detected automatically:
      3 cols : head  relation  tail               (no time)
      4 cols : head  relation  tail  year
      5 cols : head  relation  tail  year_start  year_end

    Lines starting with '#' are treated as comments.
    """
    ds   = TKGDataset()
    path = Path(path)

    with path.open(encoding=encoding, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(delimiter)
            if len(cols) == 3:
                h, r, t = cols
                ds.add_relation_triple(h, r, t)
            elif len(cols) == 4:
                h, r, t, ys = cols
                ds.add_relation_triple(h, r, t,
                                       time_start=_parse_year(ys))
            elif len(cols) >= 5:
                h, r, t, ys, ye = cols[:5]
                ds.add_relation_triple(
                    h, r, t,
                    time_start=_parse_year(ys),
                    time_end=_parse_year(ye),
                )
            # ignore lines with unexpected column counts

    ds.build_indices()
    return ds


# ---------------------------------------------------------------------------
# N-Triples loader
# ---------------------------------------------------------------------------

_NT_PATTERN = re.compile(
    r'<([^>]+)>\s+<([^>]+)>\s+(.+?)\s+\.\s*$'
)
_LITERAL_YEAR = re.compile(r'"(\d{1,4})"(?:\^\^<[^>]*gYear[^>]*>)?')
_URI_PATTERN  = re.compile(r'<([^>]+)>')


def load_ntriples(
    path: str | Path,
    encoding: str = "utf-8",
    limit: Optional[int] = None,
    progress_every: int = 500_000,
) -> TKGDataset:
    """
    Load a TKG from a W3C N-Triples (.nt) or N-Triples-in-Turtle (.ttl / .ttl.bz2)
    file.  Supports bzip2 compression transparently.

    DBpedia's mappingbased-objects_lang=en.ttl.bz2 and
    mappingbased-literals_lang=en.ttl.bz2 are actually N-Triples format
    despite the .ttl extension, so this loader handles them directly.

    Parameters
    ----------
    limit          : stop after this many triples (useful for testing/subsets)
    progress_every : print a progress line every N triples
    """
    ds    = TKGDataset()
    path  = Path(path)
    count = 0

    opener = (bz2.open(path, "rt", encoding=encoding, errors="replace")
              if str(path).endswith(".bz2")
              else path.open(encoding=encoding, errors="replace"))

    with opener as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _NT_PATTERN.match(line)
            if not m:
                continue
            subj = _local_name(m.group(1))
            pred = _local_name(m.group(2))
            obj  = m.group(3).strip()

            year_m = _LITERAL_YEAR.match(obj)
            if year_m:
                ds.add_relation_triple(subj, pred, subj,
                                       time_start=int(year_m.group(1)))
            elif obj.startswith("<"):
                uri_m = _URI_PATTERN.match(obj)
                if uri_m:
                    ds.add_relation_triple(subj, pred, _local_name(uri_m.group(1)))
            else:
                literal = obj.strip('"').split('"')[0]
                ds.add_attribute_triple(subj, pred, literal)

            count += 1
            if progress_every and count % progress_every == 0:
                print(f"  [load_ntriples] {count:,} triples…")
            if limit and count >= limit:
                break

    if progress_every:
        print(f"  [load_ntriples] done — {count:,} triples, "
              f"{ds.num_entities:,} entities, {ds.num_relations:,} relations")
    ds.build_indices()
    return ds


# ---------------------------------------------------------------------------
# Turtle loader  (DBpedia .ttl / .ttl.bz2)
# ---------------------------------------------------------------------------

def load_turtle(
    path: str | Path,
    limit: Optional[int] = None,
    progress_every: int = 500_000,
) -> TKGDataset:
    """
    Load a DBpedia .ttl or .ttl.bz2 file.

    DBpedia files (mappingbased-objects, mappingbased-literals) are actually
    N-Triples format despite the .ttl extension — one full URI triple per line,
    no @prefix declarations.  This function detects that and delegates to the
    fast streaming load_ntriples loader.

    If the file contains @prefix declarations (true Turtle), rdflib is used.
    For very large true-Turtle files, prefer converting to N-Triples first.
    """
    path = Path(path)

    # Peek at first non-empty line to detect format
    opener = (bz2.open(path, "rt", encoding="utf-8", errors="replace")
              if str(path).endswith(".bz2")
              else path.open("r", encoding="utf-8", errors="replace"))
    first_line = ""
    with opener as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_line = stripped
                break

    # If first line is a full URI triple, it's N-Triples (delegate)
    if first_line.startswith("<"):
        return load_ntriples(path, limit=limit, progress_every=progress_every)

    # True Turtle with @prefix — use rdflib (only practical for small files)
    try:
        import rdflib
        from rdflib import URIRef, Literal, Graph
    except ImportError:
        raise ImportError("rdflib is required for true-Turtle files: pip install rdflib")

    ds    = TKGDataset()
    count = [0]

    class _LimitReached(Exception):
        pass

    def _add(subj, pred, obj):
        if not isinstance(subj, URIRef):
            return
        s = _local_name(str(subj))
        p = _local_name(str(pred))
        if isinstance(obj, URIRef):
            ds.add_relation_triple(s, p, _local_name(str(obj)))
        elif isinstance(obj, Literal):
            dt = str(obj.datatype) if obj.datatype else ""
            val = str(obj)
            if "gYear" in dt or "date" in dt:
                ds.add_relation_triple(s, p, s, time_start=_parse_year(val[:4]))
            else:
                ds.add_attribute_triple(s, p, val[:256])
        else:
            return
        count[0] += 1
        if progress_every and count[0] % progress_every == 0:
            print(f"  [load_turtle] {count[0]:,} triples…")
        if limit and count[0] >= limit:
            raise _LimitReached()

    class _SinkGraph(Graph):
        def add(self, triple):
            _add(*triple)

    fh = (bz2.open(path, "rt", encoding="utf-8", errors="replace")
          if str(path).endswith(".bz2")
          else path.open("r", encoding="utf-8", errors="replace"))
    try:
        _SinkGraph().parse(fh, format="turtle")
    except _LimitReached:
        pass
    finally:
        fh.close()

    print(f"  [load_turtle] done — {count[0]:,} triples, "
          f"{ds.num_entities:,} entities, {ds.num_relations:,} relations")
    ds.build_indices()
    return ds


# ---------------------------------------------------------------------------
# QALD loader  (QALD-6 / QALD-7 JSON format)
# ---------------------------------------------------------------------------

class QALDDataset:
    """
    Holds a TKGDataset (built from QALD question annotations) together
    with the per-question answer sets needed for evaluation.

    Attributes
    ----------
    tkg       : TKGDataset
    questions : list of {"id": str, "question": str, "answers": set[str]}
    """

    def __init__(self, tkg: TKGDataset, questions: list[dict]) -> None:
        self.tkg       = tkg
        self.questions = questions

    def __len__(self) -> int:
        return len(self.questions)

    def answer_entity_ids(self, question_idx: int) -> set[int]:
        """Return the TKG entity ids for the answers to question i."""
        q = self.questions[question_idx]
        return {
            self.tkg.entity2id[ans]
            for ans in q["answers"]
            if ans in self.tkg.entity2id
        }


def load_qald(path: str | Path, encoding: str = "utf-8") -> QALDDataset:
    """
    Parse a real QALD-6 / QALD-7 multilingual JSON file.

    Actual file structure (ag-sc/QALD on GitHub):
    {
      "dataset": {"id": "qald-7-train-multilingual"},
      "questions": [
        {
          "id": "1",
          "answertype": "resource",
          "question": [
            {"language": "en", "string": "Who is the ...", "keywords": "..."}
          ],
          "query": {"sparql": "SELECT DISTINCT ?uri WHERE { ... }"},
          "answers": [
            {
              "head": {"vars": ["uri"]},
              "results": {
                "bindings": [
                  {"uri": {"type": "uri", "value": "http://dbpedia.org/resource/..."}}
                ]
              }
            }
          ]
        }
      ]
    }

    Entities mentioned in answers are registered in the TKGDataset vocabulary.
    Triple-level temporal annotations are not present in QALD files; the SPARQL
    query string is stored for reference but not executed here.
    """
    ds        = TKGDataset()
    questions = []

    with Path(path).open(encoding=encoding) as f:
        data = json.load(f)

    for q in data.get("questions", []):
        qid = str(q.get("id", ""))

        # --- English question string ---
        qtext = ""
        for lang_entry in q.get("question", []):
            if lang_entry.get("language") == "en":
                qtext = lang_entry.get("string", "")
                break

        # --- SPARQL query string (stored for reference) ---
        sparql = q.get("query", {}).get("sparql", "")

        # --- answer URIs ---
        answers: set[str] = set()
        for answer_block in q.get("answers", []):
            for binding in answer_block.get("results", {}).get("bindings", []):
                for val in binding.values():
                    uri = val.get("value", "")
                    if uri and val.get("type") in ("uri", "typed-literal", None):
                        answers.add(_local_name(uri))

        # register answer entities in the vocabulary
        for ans in answers:
            ds.add_entity(ans)

        questions.append({
            "id":       qid,
            "question": qtext,
            "sparql":   sparql,
            "answers":  answers,
        })

    ds.build_indices()
    return QALDDataset(ds, questions)


def load_lcquad(path: str | Path, encoding: str = "utf-8") -> QALDDataset:
    """
    Parse an LC-QuAD 1.0 JSON file (AskNowQA/LC-QuAD on GitHub).

    Actual file structure (train.json / test.json):
    [
      {
        "_id": "1",
        "corrected_question": "What is the ...",
        "sparql_query": "SELECT DISTINCT ?uri WHERE { ... }",
        "sparql_template_id": 1,
        "subgraph": "...",
        "entities": ["http://dbpedia.org/resource/..."],
        "relations": ["http://dbpedia.org/ontology/..."],
        "answer": [
          {"type": "uri", "value": "http://dbpedia.org/resource/..."}
        ]
      },
      ...
    ]

    Answer URIs are resolved to local names and registered as entities.
    Entity/relation URIs from the question annotations are added to the TKG
    vocabulary (without triples — the TKG structure is supplied separately
    from a DBpedia dump loaded via load_ntriples or load_tsv).
    """
    ds        = TKGDataset()
    questions = []

    with Path(path).open(encoding=encoding) as f:
        data = json.load(f)

    for q in data:
        qid   = str(q.get("_id", q.get("id", "")))
        qtext = q.get("corrected_question", q.get("question", ""))
        sparql = q.get("sparql_query", "")

        # --- answer URIs ---
        answers: set[str] = set()
        for ans_entry in q.get("answer", []):
            uri = ans_entry.get("value", "")
            if uri:
                answers.add(_local_name(uri))

        # --- register entities and relations from annotations ---
        for uri in q.get("entities", []):
            ds.add_entity(_local_name(uri))
        for uri in q.get("relations", []):
            ds.add_relation(_local_name(uri))
        for ans in answers:
            ds.add_entity(ans)

        questions.append({
            "id":       qid,
            "question": qtext,
            "sparql":   sparql,
            "answers":  answers,
        })

    ds.build_indices()
    return QALDDataset(ds, questions)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _local_name(uri: str) -> str:
    """Extract the fragment or last path segment from a URI."""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def _parse_year(s: str) -> Optional[int]:
    s = s.strip()
    if not s or s in ("-", "?", "NA", "None", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        return None
