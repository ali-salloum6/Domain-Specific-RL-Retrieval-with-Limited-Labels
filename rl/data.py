import json
import random
from pathlib import Path
import sys

# Ensure repo root is on path
_TOP = Path(__file__).resolve().parent.parent
if str(_TOP) not in sys.path:
    sys.path.insert(0, str(_TOP))

from setup.legalbench_data import load_queries_and_qrels, load_corpus

class RLDataset:
    def __init__(self, queries, qrels, bm25_run, doc_texts):
        """
        queries: list of dicts with 'qid' and 'text'
        qrels: dict mapping qid to set of relevant doc_ids
        bm25_run: dict mapping qid to list of candidate doc_ids (already truncated if needed)
        doc_texts: dict mapping doc_id to doc text
        """
        self.queries = queries
        self.qrels = qrels
        self.bm25_run = bm25_run
        self.doc_texts = doc_texts

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        qid = q["qid"]
        query_text = q["text"]

        candidates = self.bm25_run.get(qid, [])
        candidate_texts = [self.doc_texts.get(doc_id, "") for doc_id in candidates]
        relevant_docs = self.qrels.get(qid, set())

        return {
            "qid": qid,
            "query_text": query_text,
            "candidate_ids": candidates,
            "candidate_texts": candidate_texts,
            "relevant_docs": relevant_docs,
        }

def prepare_rl_data(
    train_ratio=0.8,
    seed=42,
    bm25_top_k: int = 50,
    max_train_queries: int | None = None,
    max_test_queries: int | None = None,
):
    """
    Loads LegalBench queries, qrels, corpus, and BM25 run.
    Truncates each query's BM25 list to bm25_top_k (e.g. 30).
    Splits into train / test; optionally caps train (and test) size.
    Returns (train_dataset, test_dataset, qrels)
    """
    queries_df, qrels = load_queries_and_qrels()
    corpus_ids, corpus_texts = load_corpus()
    doc_texts = dict(zip(corpus_ids, corpus_texts))

    run_path = _TOP / "setup" / "data" / "runs" / "run_bm25.json"
    with open(run_path, "r") as f:
        bm25_run = json.load(f)
        bm25_run = {str(k): v for k, v in bm25_run.items()}

    if bm25_top_k and bm25_top_k > 0:
        bm25_run = {qid: docs[:bm25_top_k] for qid, docs in bm25_run.items()}

    valid_queries = []
    for _, row in queries_df.iterrows():
        qid = str(row["qid"])
        if qid in bm25_run and len(bm25_run[qid]) > 0 and qid in qrels:
            valid_queries.append({"qid": qid, "text": row["text"]})

    random.seed(seed)
    random.shuffle(valid_queries)
    split_idx = int(len(valid_queries) * train_ratio)

    train_queries = valid_queries[:split_idx]
    test_queries = valid_queries[split_idx:]

    if max_train_queries is not None and max_train_queries > 0:
        train_queries = train_queries[: max_train_queries]
    if max_test_queries is not None and max_test_queries > 0:
        test_queries = test_queries[: max_test_queries]

    train_dataset = RLDataset(train_queries, qrels, bm25_run, doc_texts)
    test_dataset = RLDataset(test_queries, qrels, bm25_run, doc_texts)

    return train_dataset, test_dataset, qrels
