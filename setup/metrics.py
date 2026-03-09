"""
Evaluation metrics for CLERC retrieval: recall@k, MRR, nDCG.
Used by the evaluation pipeline and as reward signals in Phase 2 RL.
"""
from __future__ import annotations

from typing import Any


def recall_at_k(
    run: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int,
) -> float:
    """
    run: qid -> ordered list of retrieved doc/passage ids (up to k or more).
    qrels: qid -> set of relevant ids.
    """
    if not qrels:
        return 0.0
    scores = []
    for qid, rel_set in qrels.items():
        if qid not in run or not rel_set:
            scores.append(0.0)
            continue
        retrieved = run[qid][:k]
        hits = len(rel_set & set(retrieved))
        scores.append(hits / len(rel_set) if rel_set else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mrr(
    run: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 100,
) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item (0 if none in top-k)."""
    if not qrels:
        return 0.0
    rr = []
    for qid, rel_set in qrels.items():
        if qid not in run or not rel_set:
            rr.append(0.0)
            continue
        retrieved = run[qid][:k]
        rank = 0
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in rel_set:
                rank = i
                break
        rr.append(1.0 / rank if rank else 0.0)
    return sum(rr) / len(rr) if rr else 0.0


def dcg_at_k(retrieved: list[str], rel_set: set[str], k: int) -> float:
    """DCG@k with binary relevance."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1.0 if doc_id in rel_set else 0.0
        dcg += rel / (__import__("math").log2(i + 1))
    return dcg


def ndcg_at_k(
    run: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> float:
    """nDCG@k (binary relevance). Average over queries."""
    if not qrels:
        return 0.0
    import math
    ndcg_list = []
    for qid, rel_set in qrels.items():
        if qid not in run or not rel_set:
            ndcg_list.append(0.0)
            continue
        retrieved = run[qid][:k]
        dcg = dcg_at_k(retrieved, rel_set, k)
        # Ideal DCG: all relevant docs first
        ideal_order = list(rel_set)[:k]
        idcg = dcg_at_k(ideal_order, rel_set, k)
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0


def evaluate(
    run: dict[str, list[str]],
    qrels: dict[str, set[str]],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute recall@k, MRR, nDCG for a run.
    run: qid -> list of retrieved doc/passage ids.
    qrels: qid -> set of relevant ids.
    """
    if ks is None:
        ks = [1, 5, 10, 100]
    out = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(run, qrels, k)
    out["mrr"] = mrr(run, qrels, k=max(ks))
    out["ndcg@10"] = ndcg_at_k(run, qrels, k=10)
    return out
