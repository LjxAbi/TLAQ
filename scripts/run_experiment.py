"""
End-to-end TLAQ evaluation on QALD-6 and/or QALD-7.

Pipeline
--------
1. Parse QALD JSON files → extract anchor entities + SPARQL patterns
2. Targeted-scan DBpedia bz2 files → keep only triples involving QALD
   anchor entities (both as subject and as object) plus a configurable
   random-sample budget for background structure
3. Add inverse relations  (r → r_inv) so reverse queries become forward
4. Train TLAQ (GCN → HyTE → pipeline assembly)
5. Evaluate P/R/F1@{20,40,100,200} per dataset
6. Save metrics.json + loss_history.json to results/<run_name>/

Usage
-----
python scripts/run_experiment.py \\
    --qald  src/data/qald7/qald-7-train-multilingual.json \\
            src/data/qald7/qald-7-test-multilingual.json  \\
    --dbpedia-objects   src/data/dbpedia/mappingbased-objects_lang=en.ttl.bz2 \\
    --dbpedia-literals  src/data/dbpedia/mappingbased-literals_lang=en.ttl.bz2 \\
    --background-limit  200000  \\
    --epochs 50 --embed-dim 128 --device cpu
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.tkg_dataset import TKGDataset
from src.data.sparql_parser import extract_pattern
from src.training.trainer import TLAQTrainer, TrainConfig
from src.evaluation.metrics import evaluate_dataset

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NT_PATTERN   = re.compile(r'<([^>]+)>\s+<([^>]+)>\s+(.+?)\s+\.\s*$')
_LITERAL_YEAR = re.compile(r'"(\d{1,4})"(?:\^\^<[^>]*gYear[^>]*>)?')
_URI_PATTERN  = re.compile(r'<([^>]+)>')


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# QALD loading
# ---------------------------------------------------------------------------

def load_qald_eval(paths: list[str]) -> tuple[list[dict], set[str]]:
    """
    Load one or more QALD JSON files and return:
      questions : list of {id, question, sparql, answers (set of local names), pattern}
      anchor_set: set of local entity names appearing as SPARQL anchors
    """
    questions: list[dict] = []
    anchor_set: set[str] = set()

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            qid   = str(q.get("id", ""))
            qtext = next(
                (e.get("string", "") for e in q.get("question", [])
                 if e.get("language") == "en"),
                "",
            )
            sparql = q.get("query", {}).get("sparql", "")
            pattern = extract_pattern(sparql)

            # ground-truth answers (local names of URIs)
            answers: set[str] = set()
            for ab in q.get("answers", []):
                for binding in ab.get("results", {}).get("bindings", []):
                    for val in binding.values():
                        uri = val.get("value", "")
                        if uri:
                            answers.add(_local(uri))

            if pattern:
                anchor_set.add(pattern.entity)

            questions.append({
                "id":       qid,
                "question": qtext,
                "sparql":   sparql,
                "answers":  answers,
                "pattern":  pattern,
            })

    return questions, anchor_set


# ---------------------------------------------------------------------------
# Targeted DBpedia scan
# ---------------------------------------------------------------------------

def targeted_scan(
    bz2_path: str,
    anchor_set: set[str],
    background_limit: int = 200_000,
    progress_every: int = 500_000,
    scan_limit: Optional[int] = None,
) -> list[tuple[str, str, str, Optional[int]]]:
    """
    Stream a DBpedia N-Triples bz2 file.

    Keeps:
    - ALL triples where subject or object local-name is in anchor_set
    - Up to background_limit additional triples for structural context

    Returns list of (subj_local, pred_local, obj_local, year_or_None)
    """
    triples: list[tuple[str, str, str, Optional[int]]] = []
    bg_count  = 0
    line_count = 0

    opener = (bz2.open(bz2_path, "rt", encoding="utf-8", errors="replace")
              if bz2_path.endswith(".bz2")
              else open(bz2_path, encoding="utf-8", errors="replace"))

    with opener as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _NT_PATTERN.match(line)
            if not m:
                continue

            subj = _local(m.group(1))
            pred = _local(m.group(2))
            obj_raw = m.group(3).strip()

            # parse object
            year: Optional[int] = None
            obj: Optional[str] = None

            year_m = _LITERAL_YEAR.match(obj_raw)
            if year_m:
                year = int(year_m.group(1))
                obj  = subj   # self-loop for year annotation
            elif obj_raw.startswith("<"):
                uri_m = _URI_PATTERN.match(obj_raw)
                if uri_m:
                    obj = _local(uri_m.group(1))
            else:
                obj = obj_raw.strip('"').split('"')[0]

            if obj is None:
                line_count += 1
                continue

            is_anchor = subj in anchor_set or obj in anchor_set

            if is_anchor:
                triples.append((subj, pred, obj, year))
            elif bg_count < background_limit:
                triples.append((subj, pred, obj, year))
                bg_count += 1

            line_count += 1
            if progress_every and line_count % progress_every == 0:
                print(f"    scanned {line_count:,} lines, "
                      f"kept {len(triples):,} triples (anchor+bg) …")
            if scan_limit and line_count >= scan_limit:
                break

    print(f"    done — scanned {line_count:,} lines, "
          f"kept {len(triples):,} triples (bg={bg_count:,})")
    return triples


# ---------------------------------------------------------------------------
# DBpedia SPARQL fetch (alternative to local bz2 scan)
# ---------------------------------------------------------------------------

def sparql_fetch(
    anchor_set: set[str],
    endpoint: str = "https://dbpedia.org/sparql",
    per_entity_limit: int = 2000,
    delay: float = 1.0,
) -> list[tuple[str, str, str, Optional[int]]]:
    """
    Query the DBpedia SPARQL endpoint for all triples involving each anchor
    entity (as subject or object).  No local files needed.

    Returns list of (subj_local, pred_local, obj_local, year_or_None).
    """
    import urllib.request
    import urllib.parse

    triples: list[tuple[str, str, str, Optional[int]]] = []
    seen: set[tuple[str, str, str]] = set()
    failed = 0

    anchors = sorted(anchor_set)
    for i, entity in enumerate(anchors):
        uri = f"http://dbpedia.org/resource/{entity}"
        query = (
            f"SELECT ?s ?p ?o WHERE {{"
            f"  {{ <{uri}> ?p ?o }} UNION {{ ?s ?p <{uri}> }}"
            f"}} LIMIT {per_entity_limit}"
        )
        params = urllib.parse.urlencode({
            "query":  query,
            "format": "application/sparql-results+json",
        })
        url = f"{endpoint}?{params}"
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/sparql-results+json",
                              "User-Agent": "TLAQ-research/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            for binding in data.get("results", {}).get("bindings", []):
                p_b = binding.get("p", {})
                o_b = binding.get("o", {})
                s_b = binding.get("s", {})
                if not p_b or not o_b:
                    continue

                subj = _local(s_b.get("value", "")) if s_b else entity
                pred = _local(p_b.get("value", ""))
                o_type   = o_b.get("type", "")
                o_val    = o_b.get("value", "")
                o_dtype  = o_b.get("datatype", "")

                year: Optional[int] = None
                if o_type == "uri":
                    obj: Optional[str] = _local(o_val)
                elif "gYear" in o_dtype:
                    try:
                        year = int(o_val[:4])
                    except ValueError:
                        year = None
                    obj = subj
                else:
                    obj = o_val.strip()

                if not obj or not pred or not subj:
                    continue

                key = (subj, pred, obj)
                if key not in seen:
                    seen.add(key)
                    triples.append((subj, pred, obj, year))

        except Exception as exc:
            failed += 1
            print(f"  [warn] {entity}: {exc}")
            time.sleep(delay * 2)
            continue

        if (i + 1) % 20 == 0 or (i + 1) == len(anchors):
            print(f"  {i+1}/{len(anchors)} entities fetched — "
                  f"{len(triples):,} triples so far")
        time.sleep(delay)

    print(f"  Done. {len(triples):,} triples, {failed} entities failed.")
    return triples


# ---------------------------------------------------------------------------
# TKG builder
# ---------------------------------------------------------------------------

def build_tkg(raw_triples: list[tuple[str, str, str, Optional[int]]],
              add_inverse: bool = True) -> TKGDataset:
    """
    Build a TKGDataset from (subj, pred, obj, year?) tuples.
    If add_inverse, also insert (obj, pred+'_inv', subj) for each triple.
    """
    ds = TKGDataset()
    for subj, pred, obj, year in raw_triples:
        if not obj:
            continue
        if year is not None:
            ds.add_relation_triple(subj, pred, obj, time_start=year)
            if add_inverse:
                ds.add_relation_triple(obj, pred + "_inv", subj, time_start=year)
        else:
            # decide relation vs attribute heuristic:
            # entity local-names contain upper-case or underscores; literals don't
            if "/" in obj or "_" in obj or (obj and obj[0].isupper()):
                ds.add_relation_triple(subj, pred, obj)
                if add_inverse:
                    ds.add_relation_triple(obj, pred + "_inv", subj)
            else:
                ds.add_attribute_triple(subj, pred, obj)

    ds.build_indices()
    return ds


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_qald_eval(
    questions: list[dict],
    tkg: TKGDataset,
    pipeline,
    top_k: int,
    eval_ks: list[int],
) -> dict:
    """
    Evaluate pipeline on QALD questions that have extractable patterns.

    Returns per-dataset metrics dict.
    """
    ranked_all: list[list[int]] = []
    gt_all:     list[set[int]]  = []
    skipped = 0

    for q in questions:
        pat = q["pattern"]
        if pat is None:
            skipped += 1
            continue

        # resolve entity and relation
        if pat.mode == "forward":
            head_name = pat.entity
            rel_name  = pat.relation
        else:  # reverse → use inverse relation
            head_name = pat.entity
            rel_name  = pat.relation + "_inv"

        head_id = tkg.entity2id.get(head_name)
        rel_id  = tkg.relation2id.get(rel_name)

        if head_id is None or rel_id is None:
            skipped += 1
            continue

        # ground truth: answer entities that exist in TKG
        gt_ids = {tkg.entity2id[a] for a in q["answers"] if a in tkg.entity2id}
        if not gt_ids:
            skipped += 1
            continue

        # TLAQ entity query
        query = (head_id, rel_id, -1, None, None)
        ranked = pipeline.entity_query_ids([query], top_k=top_k)

        ranked_all.append(ranked)
        gt_all.append(gt_ids)

    print(f"  Evaluated {len(ranked_all)} questions  "
          f"(skipped {skipped} — no pattern / OOV / no GT)")

    if not ranked_all:
        return {}

    return evaluate_dataset(ranked_all, gt_all, ks=eval_ks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TLAQ QALD experiment.")
    p.add_argument("--qald", nargs="+", required=True,
                   help="QALD JSON files (train and/or test).")
    p.add_argument("--dbpedia-objects",
                   default="src/data/dbpedia/mappingbased-objects_lang=en.ttl.bz2")
    p.add_argument("--dbpedia-literals",
                   default="src/data/dbpedia/mappingbased-literals_lang=en.ttl.bz2")
    p.add_argument("--background-limit", type=int, default=200_000,
                   help="Max background (non-anchor) triples per DBpedia file.")
    p.add_argument("--scan-limit", type=int, default=None,
                   help="Stop reading each DBpedia file after this many lines "
                        "(for quick testing; None = scan entire file).")
    p.add_argument("--embed-dim",   type=int,   default=128)
    p.add_argument("--epochs",      type=int,   default=50,
                   help="GCN and HyTE epochs each.")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--top-k",       type=int,   default=200)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--out",         default="results")
    p.add_argument("--run-name",    default=None)
    p.add_argument("--no-inverse",  action="store_true",
                   help="Do not add inverse relations (reverse queries become no-ops).")
    p.add_argument("--use-sparql", action="store_true",
                   help="Fetch triples from DBpedia SPARQL instead of local bz2 files.")
    p.add_argument("--sparql-endpoint", default="https://dbpedia.org/sparql",
                   help="SPARQL endpoint URL (default: DBpedia public endpoint).")
    p.add_argument("--sparql-limit", type=int, default=2000,
                   help="Max triples per entity from SPARQL (default 2000).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── 1. Load QALD questions ────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Loading QALD questions")
    print("=" * 60)
    questions, anchor_set = load_qald_eval(args.qald)
    print(f"  {len(questions)} questions, {len(anchor_set)} unique anchor entities")

    # ── 2. Load DBpedia triples ───────────────────────────────────────────
    raw_triples: list[tuple] = []

    if args.use_sparql:
        print("\nStep 2 — Fetching triples from DBpedia SPARQL endpoint")
        print(f"  Endpoint: {args.sparql_endpoint}")
        t0 = time.time()
        raw_triples = sparql_fetch(
            anchor_set,
            endpoint=args.sparql_endpoint,
            per_entity_limit=args.sparql_limit,
        )
        print(f"  {len(raw_triples):,} triples fetched in {time.time()-t0:.1f}s")
    else:
        for fpath in (args.dbpedia_objects, args.dbpedia_literals):
            if not Path(fpath).exists():
                print(f"  [skip] {fpath} not found")
                continue
            print(f"\nStep 2 — Scanning {Path(fpath).name}")
            t0 = time.time()
            triples = targeted_scan(
                fpath,
                anchor_set,
                background_limit=args.background_limit,
                scan_limit=args.scan_limit,
            )
            raw_triples.extend(triples)
            print(f"  {len(triples):,} triples in {time.time()-t0:.1f}s")

    if not raw_triples:
        print("ERROR: No triples loaded. Check DBpedia file paths or SPARQL endpoint.")
        sys.exit(1)

    # deduplicate
    raw_triples = list(dict.fromkeys(raw_triples))
    print(f"\n  Total after dedup: {len(raw_triples):,} triples")

    # ── 3. Build TKG ──────────────────────────────────────────────────────
    print("\nStep 3 — Building TKG")
    add_inv = not args.no_inverse
    tkg = build_tkg(raw_triples, add_inverse=add_inv)
    print(f"  {tkg}")

    # coverage stats
    covered = sum(1 for e in anchor_set if e in tkg.entity2id)
    print(f"  Anchor coverage: {covered}/{len(anchor_set)} "
          f"({100*covered/max(len(anchor_set),1):.1f}%)")

    # ── 4. Train TLAQ ─────────────────────────────────────────────────────
    print("\nStep 4 — Training TLAQ")
    cfg = TrainConfig(
        embed_dim      = args.embed_dim,
        device         = args.device,
        gcn_epochs     = args.epochs,
        gcn_lr         = args.lr,
        gcn_batch_size = args.batch_size,
        hyte_epochs    = args.epochs,
        hyte_lr        = args.lr,
        hyte_batch_size= args.batch_size,
        patience       = args.patience,
        top_k          = args.top_k,
    )
    trainer  = TLAQTrainer(tkg, cfg)
    t0 = time.time()
    pipeline, history = trainer.train()
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\nStep 5 — Evaluating on QALD")
    all_metrics: dict[str, dict] = {}
    for fpath in args.qald:
        label = Path(fpath).stem
        with open(fpath, encoding="utf-8") as f:
            data_f = json.load(f)
        ids_in_file = {str(q2.get("id", "")) for q2 in data_f.get("questions", [])}
        q_subset = [q for q in questions if q["id"] in ids_in_file]
        metrics = run_qald_eval(q_subset, tkg, pipeline, args.top_k, cfg.eval_ks)
        all_metrics[label] = metrics
        if metrics:
            print(f"\n  [{label}]")
            for k in sorted(metrics,
                             key=lambda s: (int(s.split("@")[1]), s.split("@")[0])):
                print(f"    {k} = {metrics[k]:.4f}")

    # ── 6. Save ────────────────────────────────────────────────────────────
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir  = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pipeline.gcn.state_dict(),     out_dir / "pipeline_gcn.pt")
    torch.save(pipeline.hyte.state_dict(),    out_dir / "pipeline_hyte.pt")
    torch.save(pipeline.encoder.state_dict(), out_dir / "pipeline_encoder.pt")
    torch.save(pipeline.embedder.state_dict(),out_dir / "pipeline_embedder.pt")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    loss_record = {
        "gcn":  history.gcn_losses,
        "hyte": history.hyte_losses,
        "gcn_time_s":   history.gcn_time_s,
        "hyte_time_s":  history.hyte_time_s,
        "total_time_s": train_time,
    }
    with open(out_dir / "loss_history.json", "w") as f:
        json.dump(loss_record, f, indent=2)

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
