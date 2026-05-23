"""
LSTM-based joint encoding of predicate + time information  (TLAQ Section 3.5).

"The encoding of time information and predicate vectors trained by Sections 3.3
and 3.4 are put into LSTM.  The predicate encoding with time information is
obtained by using LSTM to jointly encode predicates and time."

Architecture
────────────
The LSTM receives a 2-step sequence per predicate:

  step 0 : predicate embedding     [d]     (from GCN / HyTE training)
  step 1 : projected time encoding [d]     (Linear(time_dim → d) applied to
                                            the 52-bit or 104-bit time vector)

Final hidden state h_n [d] is the time-aware predicate representation fed
into the graph-level embedding module.

For temporal intervals [τs, τe], both the start and end encodings are
projected separately and concatenated before the projection layer, giving
a 2*time_dim input to the time projection.  This choice encodes directionality
(start vs end) that a single timestamp cannot express.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.time_encoding import TIME_DIM


# ---------------------------------------------------------------------------
# Temporal Predicate Encoder
# ---------------------------------------------------------------------------

class TemporalPredicateEncoder(nn.Module):
    """
    Jointly encodes a predicate embedding and a time encoding via LSTM.

    Parameters
    ----------
    embed_dim  : dimension of predicate embeddings (d)
    time_dim   : raw time encoding size (52 for point, 104 for interval)
    num_layers : LSTM depth
    dropout    : applied between LSTM layers (only effective when num_layers > 1)
    """

    def __init__(
        self,
        embed_dim:  int = 128,
        time_dim:   int = TIME_DIM,           # 52 (point) or 2*TIME_DIM (interval)
        num_layers: int = 1,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.time_dim   = time_dim
        self.num_layers = num_layers

        # project the raw binary time vector into predicate embedding space
        self.time_proj = nn.Linear(time_dim, embed_dim)

        # 2-step LSTM: step-0 = predicate, step-1 = time
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = embed_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        pred_emb:  torch.Tensor,   # [B, d]   predicate embedding
        time_enc:  torch.Tensor,   # [B, T]   T = time_dim (52 or 104)
    ) -> torch.Tensor:             # [B, d]   time-aware predicate embedding
        """
        Returns the LSTM final hidden state as the time-aware predicate
        embedding.  The sequence is [pred_emb, proj(time_enc)].
        """
        B = pred_emb.size(0)

        # project time encoding into embed_dim space
        time_feat = F.relu(self.time_proj(time_enc))   # [B, d]

        # 2-step sequence: (step0=predicate, step1=time)
        seq = torch.stack([pred_emb, time_feat], dim=1)   # [B, 2, d]

        _, (h_n, _) = self.lstm(seq)   # h_n: [num_layers, B, d]
        return h_n[-1]                 # [B, d]  last layer's hidden state

    def encode_single(
        self,
        pred_emb: torch.Tensor,   # [d]
        time_enc: torch.Tensor,   # [T]
    ) -> torch.Tensor:            # [d]
        """Convenience wrapper for unbatched single-triple encoding."""
        return self.forward(
            pred_emb.unsqueeze(0),
            time_enc.unsqueeze(0),
        ).squeeze(0)


# ---------------------------------------------------------------------------
# Relation / attribute predicate index builder
# ---------------------------------------------------------------------------

class PredicateTimeIndex:
    """
    Stores the time-encoded predicate representations for all predicates
    in the TKG, built once from the trained embeddings + time annotations.

    Usage:
        index = PredicateTimeIndex(encoder, pred_emb_table, triple_list)
        r_emb = index.get_relation_emb(relation_id)   # [d], time-aware
    """

    def __init__(
        self,
        encoder:        TemporalPredicateEncoder,
        pred_emb_table: torch.Tensor,             # [num_preds, d]
        triples,                                  # list of RelationTriple/AttributeTriple
        use_interval:   bool = True,
        device:         Optional[torch.device] = None,
    ) -> None:
        from src.utils.time_encoding import encode_interval, encode_timestamp

        self.device  = device or torch.device("cpu")
        self._cache: dict[int, torch.Tensor] = {}

        encoder.eval()
        with torch.no_grad():
            for triple in triples:
                pid = triple.relation if hasattr(triple, "relation") else triple.attr_pred
                if pid in self._cache:
                    continue

                pred_emb = pred_emb_table[pid].to(self.device)   # [d]

                ts = triple.time_start
                te = triple.time_end

                if ts is None:
                    # no time info → use a zero time encoding
                    time_vec = torch.zeros(
                        encoder.time_dim, dtype=torch.float32, device=self.device
                    )
                elif use_interval and te is not None:
                    time_vec = encode_interval(ts, te).to(self.device)
                else:
                    time_vec = encode_timestamp(ts).to(self.device)

                self._cache[pid] = encoder.encode_single(pred_emb, time_vec)

    def get(self, pred_id: int) -> torch.Tensor:
        """Return the time-aware predicate embedding [d] for pred_id."""
        if pred_id not in self._cache:
            raise KeyError(f"Predicate id {pred_id} not in index.")
        return self._cache[pred_id]

    def all_embeddings(self) -> Tuple[list[int], torch.Tensor]:
        """Return (ids, embeddings [N, d]) for all cached predicates."""
        ids   = sorted(self._cache.keys())
        embs  = torch.stack([self._cache[i] for i in ids])
        return ids, embs
