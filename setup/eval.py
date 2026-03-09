"""
Evaluation API for CLERC: evaluate(retriever_or_run, split, qrels_level).
Returns dict of recall@k, MRR, nDCG for Phase 2 RL reward.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

_TOP = Path(__file__).resolve().parent.parent
if _TOP not in sys.path:
    sys.path.insert(0, str(_TOP))
from setup.metrics import evaluate as _compute_metrics


def evaluate(
    retriever_or_run: dict[str, list[str]] | str | Path | Callable[..., dict[str, list[str]]],
    split: str = "test",
    query_type: str = "direct",
    qrels_level: str = "doc",
    ks: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Compute recall@k, MRR, nDCG for CLERC.

    retriever_or_run: either
      - A run dict (qid -> list of retrieved doc/passage ids), or
      - Path to a JSON run file (same structure), or
      - A callable(queries_df, qrels, **kwargs) that returns a run dict
    """
    from setup.clerc_data import load_queries_and_qrels_hf

    queries_df, qrels = load_queries_and_qrels_hf(query_type=query_type, qrels_level=qrels_level)

    if ks is None:
        ks = [1, 5, 10, 100]

    if callable(retriever_or_run):
        run = retriever_or_run(queries_df, qrels, **kwargs)
    elif isinstance(retriever_or_run, (str, Path)):
        import json
        with open(retriever_or_run) as f:
            run = json.load(f)
        run = {str(k): v for k, v in run.items()}
    else:
        run = {str(k): v for k, v in retriever_or_run.items()}

    return _compute_metrics(run, qrels, ks=ks)
