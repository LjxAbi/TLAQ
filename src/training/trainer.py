"""
Full TLAQ training orchestrator.

Runs the three-stage training pipeline:
  Stage 1 : ImprovedGCN          (entity embeddings)
  Stage 2 : HyTEAttributeEmbedding (attribute embeddings)
  Stage 3 : assemble TLAQPipeline  (no separate training step;
            the LSTM and graph embedder are trained end-to-end in
            future work — for now they are used in inference mode)

After each stage the trained weights are passed into the next, so
the final pipeline uses embeddings from all three components.

Usage
-----
    config  = TrainConfig(embed_dim=128, gcn_epochs=100, hyte_epochs=100)
    trainer = TLAQTrainer(dataset, config)
    pipeline, history = trainer.train()
    scores = trainer.evaluate(pipeline, test_queries, ground_truth)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.data.tkg_dataset import TKGDataset
from src.models.improved_gcn import ImprovedGCN
from src.models.gcn_trainer import GCNTrainer
from src.models.hyte_attribute import HyTEAttributeEmbedding
from src.models.hyte_trainer import HyTETrainer
from src.models.temporal_predicate_encoder import TemporalPredicateEncoder
from src.models.graph_level_embedding import GraphLevelEmbedding
from src.pipeline.tlaq_pipeline import TLAQPipeline, TLAQConfig
from src.evaluation.metrics import evaluate_dataset
from src.utils.time_encoding import TIME_DIM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # shared
    embed_dim:       int   = 128
    device:          str   = "cpu"

    # GCN (Stage 1)
    gcn_layers:      int   = 2
    gcn_epochs:      int   = 100
    gcn_lr:          float = 1e-3
    gcn_margin:      float = 1.0
    gcn_batch_size:  int   = 64
    gcn_log_every:   int   = 10

    # HyTE (Stage 2)
    hyte_epochs:     int   = 100
    hyte_lr:         float = 1e-3
    hyte_margin:     float = 1.0
    hyte_batch_size: int   = 64
    hyte_neg_mode:   str   = "tans"   # "tans" or "tdns"
    hyte_log_every:  int   = 10

    # pipeline / Part 3
    lstm_layers:     int   = 1
    lstm_dropout:    float = 0.1
    use_interval:    bool  = True
    top_k:           int   = 20
    bfs_depth:       int   = 2

    # early stopping (applied within each stage)
    patience:        int   = 10    # stop if no improvement for this many log steps
    min_delta:       float = 1e-4  # minimum improvement to reset patience counter

    # evaluation k values
    eval_ks:         list[int] = field(default_factory=lambda: [20, 40, 100, 200])


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

@dataclass
class TrainHistory:
    gcn_losses:  list[float] = field(default_factory=list)
    hyte_losses: list[float] = field(default_factory=list)
    gcn_time_s:  float = 0.0
    hyte_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TLAQTrainer:
    """
    Orchestrates all training stages for TLAQ.

    Parameters
    ----------
    dataset : TKGDataset  (build_indices() already called)
    config  : TrainConfig
    """

    def __init__(
        self,
        dataset: TKGDataset,
        config:  Optional[TrainConfig] = None,
    ) -> None:
        self.dataset = dataset
        self.cfg     = config or TrainConfig()
        self.device  = torch.device(self.cfg.device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> tuple[TLAQPipeline, TrainHistory]:
        """
        Run all training stages and return the assembled pipeline.

        Returns
        -------
        pipeline : TLAQPipeline  (ready for inference)
        history  : TrainHistory  (loss curves and wall-clock times)
        """
        history = TrainHistory()
        cfg     = self.cfg

        # ── Stage 1: GCN ──────────────────────────────────────────────
        print("=" * 60)
        print("Stage 1 — Training ImprovedGCN")
        print("=" * 60)
        t0  = time.time()
        gcn = ImprovedGCN(
            dataset    = self.dataset,
            embed_dim  = cfg.embed_dim,
            num_layers = cfg.gcn_layers,
            device     = self.device,
        )
        gcn_trainer = GCNTrainer(
            model   = gcn,
            dataset = self.dataset,
            margin  = cfg.gcn_margin,
            lr      = cfg.gcn_lr,
            device  = self.device,
        )
        history.gcn_losses = self._fit_with_patience(
            trainer    = gcn_trainer,
            epochs     = cfg.gcn_epochs,
            batch_size = cfg.gcn_batch_size,
            log_every  = cfg.gcn_log_every,
            label      = "GCN",
        )
        history.gcn_time_s = time.time() - t0
        print(f"  GCN done in {history.gcn_time_s:.1f}s  "
              f"final loss={history.gcn_losses[-1]:.4f}\n")

        # ── Stage 2: HyTE ─────────────────────────────────────────────
        if self.dataset.num_attr_preds > 0:
            print("=" * 60)
            print("Stage 2 — Training HyTEAttributeEmbedding")
            print("=" * 60)
            t0   = time.time()
            hyte = HyTEAttributeEmbedding(
                dataset   = self.dataset,
                embed_dim = cfg.embed_dim,
            )
            hyte_trainer = HyTETrainer(
                model             = hyte,
                dataset           = self.dataset,
                margin            = cfg.hyte_margin,
                lr                = cfg.hyte_lr,
                negative_sampling = cfg.hyte_neg_mode,
                device            = self.device,
            )
            history.hyte_losses = self._fit_with_patience(
                trainer    = hyte_trainer,
                epochs     = cfg.hyte_epochs,
                batch_size = cfg.hyte_batch_size,
                log_every  = cfg.hyte_log_every,
                label      = "HyTE",
            )
            history.hyte_time_s = time.time() - t0
            print(f"  HyTE done in {history.hyte_time_s:.1f}s  "
                  f"final loss={history.hyte_losses[-1]:.4f}\n")
        else:
            print("Stage 2 — Skipped (no attribute triples in dataset)\n")
            hyte = HyTEAttributeEmbedding(
                dataset=self.dataset, embed_dim=cfg.embed_dim
            )

        # ── Stage 3: assemble pipeline ────────────────────────────────
        print("=" * 60)
        print("Stage 3 — Assembling TLAQPipeline")
        print("=" * 60)
        time_dim = TIME_DIM * 2 if cfg.use_interval else TIME_DIM
        encoder  = TemporalPredicateEncoder(
            embed_dim  = cfg.embed_dim,
            time_dim   = time_dim,
            num_layers = cfg.lstm_layers,
            dropout    = cfg.lstm_dropout,
        )
        embedder = GraphLevelEmbedding(embed_dim=cfg.embed_dim)

        pipe_cfg = TLAQConfig(
            embed_dim    = cfg.embed_dim,
            gcn_layers   = cfg.gcn_layers,
            lstm_layers  = cfg.lstm_layers,
            lstm_dropout = cfg.lstm_dropout,
            top_k        = cfg.top_k,
            bfs_depth    = cfg.bfs_depth,
            use_interval = cfg.use_interval,
            device       = cfg.device,
        )
        pipeline = TLAQPipeline(
            dataset  = self.dataset,
            gcn      = gcn,
            hyte     = hyte,
            encoder  = encoder,
            embedder = embedder,
            config   = pipe_cfg,
        )
        print("  Pipeline ready.\n")
        return pipeline, history

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pipeline:      TLAQPipeline,
        entity_queries:    list[tuple],   # (head_id, rel_id, -1, ts, te)
        entity_gt:         list[set[int]],
        attribute_queries: Optional[list[tuple]] = None,  # (eid, apid, -1)
        attribute_gt:      Optional[list[set[int]]] = None,
        top_k:             Optional[int] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Run the full evaluation loop and return P/R/F1@k for both query types.

        Returns
        -------
        {
          "entity":    {"P@20": …, "R@20": …, "F1@20": …, …},
          "attribute": {"P@20": …, …}   ← only if attribute_queries provided
        }
        """
        k = top_k or self.cfg.top_k
        results: dict[str, dict[str, float]] = {}

        if entity_queries:
            all_ranked = [
                pipeline.entity_query_ids([q], top_k=k)
                for q in entity_queries
            ]
            results["entity"] = evaluate_dataset(
                all_ranked, entity_gt, ks=self.cfg.eval_ks
            )

        if attribute_queries and attribute_gt:
            all_ranked = [
                pipeline.attribute_query_ids([q], top_k=k)
                for q in attribute_queries
            ]
            results["attribute"] = evaluate_dataset(
                all_ranked, attribute_gt, ks=self.cfg.eval_ks
            )

        return results

    # ------------------------------------------------------------------
    # Internal: fit with patience-based early stopping
    # ------------------------------------------------------------------

    def _fit_with_patience(
        self,
        trainer,
        epochs:     int,
        batch_size: int,
        log_every:  int,
        label:      str,
    ) -> list[float]:
        """
        Call trainer.fit() in chunks of `log_every` epochs.
        Stop early if loss does not improve by min_delta for `patience` chunks.
        Returns the full loss list.
        """
        patience  = self.cfg.patience
        min_delta = self.cfg.min_delta

        all_losses: list[float] = []
        best_loss  = float("inf")
        no_improve = 0

        chunks = max(1, epochs // log_every)
        for chunk in range(chunks):
            losses = trainer.fit(
                epochs     = log_every,
                batch_size = batch_size,
                log_every  = log_every,
            )
            all_losses.extend(losses)
            cur_loss = losses[-1]

            done = (chunk + 1) * log_every
            print(f"  [{label}] epoch {done:4d}/{epochs}  loss={cur_loss:.4f}")

            if best_loss - cur_loss > min_delta:
                best_loss  = cur_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{label}] Early stop at epoch {done} "
                          f"(no improvement for {patience} checks)")
                    break

        return all_losses
