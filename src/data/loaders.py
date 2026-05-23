"""
TKG dataset loaders for TLAQ benchmarks.

Supported formats
-----------------
load_tsv(path)
    Tab-separated triples.  Two variants are auto-detected:

    4-column  head  relation  tail  year
    5-column  head  relation  tail  year_start  year_end

    Example (ICEWS-style):
        Barack_Obama  visited  Germany  2009
        Merkel        leads    CDU      1998  2018

load_ntriples(path)
    W3C N-Triples (.nt) with optional temporal annotations stored as
    xsd:gYear literals on the object position.

    <subject> <predicate> <object> .
    <subject> <predicate> "2004"^^<xsd:gYear> .

load_qald(path)
    QALD-6 / QALD-7 JSON format (questions + SPARQL answers).
    Extracts the answer entities from each question's SPARQL result and
    returns (question_id, answer_entity_list) pairs.

    The returned QALDDataset contains both the raw TKG triples (built
    from the question annotations) and the QA pairs for evaluation.

All loaders return a TKGDataset with build_indices() already called.
"""
from __future__ import annotations

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
) -> TKGDataset:
    """
    Load a TKG from a W3C N-Triples (.nt) file.

    URIs are shortened to their local name (fragment or last path segment)
    to keep entity/relation labels readable.  Literal objects that are
    xsd:gYear values are stored as time_start.  All other literal objects
    are treated as attribute values.
    """
    ds   = TKGDataset()
    path = Path(path)

    with path.open(encoding=encoding) as f:
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
                ds.add_relation_triple(subj, pred, subj,  # self-loop as timestamp stub
                                       time_start=int(year_m.group(1)))
            elif obj.startswith("<"):
                uri_m = _URI_PATTERN.match(obj)
                if uri_m:
                    ds.add_relation_triple(subj, pred, _local_name(uri_m.group(1)))
            else:
                # plain literal → attribute triple
                literal = obj.strip('"').split('"')[0]
                ds.add_attribute_triple(subj, pred, literal)

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
