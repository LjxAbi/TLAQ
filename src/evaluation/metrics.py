"""
Evaluation metrics for TLAQ: Precision, Recall, F1 at top-k.

Usage
-----
    results  = [triple_id_0, triple_id_1, ...]   # ranked list (ascending score)
    relevant = {triple_id_2, triple_id_5, ...}   # ground-truth set

    scores = evaluate_at_k(results, relevant, ks=[20, 40, 100, 200])
    # → {"P@20": 0.35, "R@20": 0.12, "F1@20": 0.18, ...}

The functions operate on arbitrary hashable ids so they are independent
of the specific triple representation used elsewhere in the project.
"""
from __future__ import annotations

from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def precision_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def f1_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_at_k(
    ranked:   Sequence,
    relevant: set,
    ks:       Iterable[int] = (20, 40, 100, 200),
) -> dict[str, float]:
    """
    Compute P/R/F1 at every k in `ks` for a single query.

    Returns a flat dict like {"P@20": …, "R@20": …, "F1@20": …, "P@40": …}.
    """
    out: dict[str, float] = {}
    for k in ks:
        out[f"P@{k}"]  = precision_at_k(ranked, relevant, k)
        out[f"R@{k}"]  = recall_at_k(ranked, relevant, k)
        out[f"F1@{k}"] = f1_at_k(ranked, relevant, k)
    return out


# ---------------------------------------------------------------------------
# Macro-averaged metrics over a query set
# ---------------------------------------------------------------------------

def macro_average(
    per_query_scores: list[dict[str, float]],
) -> dict[str, float]:
    """
    Average per-query metric dicts element-wise.

    Parameters
    ----------
    per_query_scores : list of dicts returned by evaluate_at_k

    Returns
    -------
    dict with the same keys, values averaged over all queries.
    """
    if not per_query_scores:
        return {}
    keys = per_query_scores[0].keys()
    n    = len(per_query_scores)
    return {k: sum(d[k] for d in per_query_scores) / n for k in keys}


def evaluate_dataset(
    all_ranked:   list[Sequence],
    all_relevant: list[set],
    ks:           Iterable[int] = (20, 40, 100, 200),
) -> dict[str, float]:
    """
    Macro-average P/R/F1@k over all queries.

    Parameters
    ----------
    all_ranked   : list of ranked result sequences (one per query)
    all_relevant : list of ground-truth id sets (one per query)
    ks           : cutoff values

    Returns
    -------
    {"P@20": …, "R@20": …, "F1@20": …, …}  macro-averaged
    """
    per_query = [
        evaluate_at_k(ranked, relevant, ks)
        for ranked, relevant in zip(all_ranked, all_relevant)
    ]
    return macro_average(per_query)
