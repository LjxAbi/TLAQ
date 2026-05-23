"""
TLAQ training + evaluation CLI.

Usage examples
--------------
# Train on a TSV dataset, evaluate, save pipeline
python scripts/train.py --data data/train.tsv --format tsv --epochs 100

# Train on QALD-7 JSON
python scripts/train.py --data data/qald7_train.json --format qald --epochs 200

# Use GPU, larger embedding, custom top-k
python scripts/train.py --data data/train.tsv --format tsv \\
    --embed-dim 256 --device cuda --top-k 40 --epochs 150

Output
------
  results/<run_name>/pipeline_gcn.pt      GCN state dict
  results/<run_name>/pipeline_hyte.pt     HyTE state dict
  results/<run_name>/metrics.json         P/R/F1@k scores
  results/<run_name>/loss_history.json    loss curves
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# make sure project root is on path when invoked from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.loaders import load_tsv, load_turtle, load_qald, load_lcquad
from src.training.trainer import TLAQTrainer, TrainConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate the TLAQ pipeline."
    )
    p.add_argument("--data",      required=True,
                   help="Path to the training dataset file.")
    p.add_argument("--test",      default=None,
                   help="Path to a separate test split (TSV only). "
                        "If omitted, evaluation is skipped.")
    p.add_argument("--format",
                   choices=["tsv", "turtle", "ttl", "qald", "lcquad"],
                   default="tsv",
                   help="Dataset file format (default: tsv). "
                        "Use 'turtle'/'ttl' for DBpedia .ttl/.ttl.bz2 files.")
    p.add_argument("--embed-dim", type=int,   default=128)
    p.add_argument("--gcn-epochs",type=int,   default=100)
    p.add_argument("--hyte-epochs",type=int,  default=100)
    p.add_argument("--gcn-lr",    type=float, default=1e-3)
    p.add_argument("--hyte-lr",   type=float, default=1e-3)
    p.add_argument("--batch-size",type=int,   default=64)
    p.add_argument("--margin",    type=float, default=1.0)
    p.add_argument("--top-k",     type=int,   default=20)
    p.add_argument("--bfs-depth", type=int,   default=2)
    p.add_argument("--device",    default="cpu",
                   help="PyTorch device string (cpu / cuda / cuda:0).")
    p.add_argument("--patience",  type=int,   default=10)
    p.add_argument("--out",       default="results",
                   help="Output directory (default: results/).")
    p.add_argument("--run-name",  default=None,
                   help="Sub-directory name inside --out. "
                        "Defaults to a timestamp.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── load dataset ──────────────────────────────────────────────────
    print(f"Loading dataset: {args.data}  (format={args.format})")
    if args.format == "tsv":
        dataset = load_tsv(args.data)
    elif args.format in ("turtle", "ttl"):
        dataset = load_turtle(args.data)
    elif args.format == "lcquad":
        dataset = load_lcquad(args.data).tkg
    else:  # qald
        dataset = load_qald(args.data).tkg

    print(f"  entities={dataset.num_entities}  "
          f"relations={dataset.num_relations}  "
          f"rel_triples={len(dataset.relation_triples)}  "
          f"attr_triples={len(dataset.attribute_triples)}")

    # ── build config ──────────────────────────────────────────────────
    cfg = TrainConfig(
        embed_dim      = args.embed_dim,
        device         = args.device,
        gcn_epochs     = args.gcn_epochs,
        gcn_lr         = args.gcn_lr,
        gcn_batch_size = args.batch_size,
        gcn_margin     = args.margin,
        hyte_epochs    = args.hyte_epochs,
        hyte_lr        = args.hyte_lr,
        hyte_batch_size= args.batch_size,
        hyte_margin    = args.margin,
        top_k          = args.top_k,
        bfs_depth      = args.bfs_depth,
        patience       = args.patience,
    )

    # ── train ─────────────────────────────────────────────────────────
    trainer = TLAQTrainer(dataset, cfg)
    t0 = time.time()
    pipeline, history = trainer.train()
    total_time = time.time() - t0
    print(f"Total training time: {total_time:.1f}s")

    # ── save ──────────────────────────────────────────────────────────
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir  = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pipeline.gcn.state_dict(),
               out_dir / "pipeline_gcn.pt")
    torch.save(pipeline.hyte.state_dict(),
               out_dir / "pipeline_hyte.pt")
    torch.save(pipeline.encoder.state_dict(),
               out_dir / "pipeline_encoder.pt")
    torch.save(pipeline.embedder.state_dict(),
               out_dir / "pipeline_embedder.pt")

    loss_record = {
        "gcn":  history.gcn_losses,
        "hyte": history.hyte_losses,
        "gcn_time_s":  history.gcn_time_s,
        "hyte_time_s": history.hyte_time_s,
        "total_time_s": total_time,
    }
    with open(out_dir / "loss_history.json", "w") as f:
        json.dump(loss_record, f, indent=2)

    print(f"\nCheckpoints saved to: {out_dir}")

    # ── evaluate (optional) ───────────────────────────────────────────
    if args.test:
        print(f"\nEvaluating on: {args.test}")
        test_ds = load_tsv(args.test)

        # build entity queries from the test set:
        # query = (head, relation, -1, time_start, time_end)
        entity_queries = [
            (t.head, t.relation, -1,
             t.time_start, t.time_end)
            for t in test_ds.relation_triples
        ]
        entity_gt = [
            {t.tail} for t in test_ds.relation_triples
        ]

        scores = trainer.evaluate(
            pipeline,
            entity_queries=entity_queries,
            entity_gt=entity_gt,
            top_k=args.top_k,
        )

        print("\nResults:")
        for qtype, metrics in scores.items():
            print(f"  [{qtype}]")
            for name, val in sorted(metrics.items()):
                print(f"    {name:8s} = {val:.4f}")

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nMetrics saved to: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
