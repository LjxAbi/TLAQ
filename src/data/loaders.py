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
    Parse a QALD-6/7 JSON file.

    QALD JSON structure (simplified):
    {
      "questions": [
        {
          "id": "1",
          "question": [{"language": "en", "string": "..."}],
          "answers": [{"results": {"bindings": [{"uri": {"value": "..."}}]}}]
          "triples": [  ← optional temporal annotation
            {"subject": "...", "predicate": "...", "object": "...",
             "time_start": 2004, "time_end": 2021}
          ]
        }
      ]
    }

    For questions without explicit triple annotations, the function
    inserts a placeholder triple using the question id as the subject.
    """
    ds        = TKGDataset()
    questions = []

    with Path(path).open(encoding=encoding) as f:
        data = json.load(f)

    for q in data.get("questions", []):
        qid = str(q.get("id", ""))

        # --- extract English question string ---
        qtext = ""
        for lang_entry in q.get("question", []):
            if lang_entry.get("language") == "en":
                qtext = lang_entry.get("string", "")
                break

        # --- extract answer URIs ---
        answers: set[str] = set()
        for answer_block in q.get("answers", []):
            for binding in answer_block.get("results", {}).get("bindings", []):
                for key, val in binding.items():
                    uri = val.get("value", "")
                    if uri:
                        answers.add(_local_name(uri))

        # --- extract triple annotations ---
        for triple in q.get("triples", []):
            subj = triple.get("subject",   qid)
            pred = triple.get("predicate", "relatedTo")
            obj  = triple.get("object",    "unknown")
            ts   = triple.get("time_start")
            te   = triple.get("time_end")
            if triple.get("type") == "attribute":
                ds.add_attribute_triple(subj, pred, obj)
            else:
                ds.add_relation_triple(subj, pred, obj,
                                       time_start=ts, time_end=te)

        # register answer entities so they appear in the vocabulary
        for ans in answers:
            ds.add_entity(ans)

        questions.append({"id": qid, "question": qtext, "answers": answers})

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
