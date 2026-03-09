# CLERC Dev-Set Baseline Results (Phase 1)

**Split:** CLERC test set (direct queries, doc-level qrels)  
**Source:** `queries/test.single-removed.direct.tsv`, `qrels/qrels-doc.test.direct.tsv`  
**Metrics:** recall@k (k=1,5,10,100), MRR, nDCG@10  

## Pipeline run (mock retrievers, 50 queries)

Used for pipeline verification. Replace with full run when BM25 index and DPR model are used.

| Model | recall@1 | recall@5 | recall@10 | recall@100 | MRR | nDCG@10 |
|-------|----------|----------|-----------|------------|-----|---------|
| BM25 (mock) | 0.02 | 0.10 | 0.20 | 1.00 | 0.090 | 0.091 |
| DPR (mock) | 0.02 | 0.10 | 0.20 | 1.00 | 0.090 | 0.091 |

## Reference (from CLERC paper / prior work)

- **Zero-shot IR on CLERC:** ~48.3% recall@1000 (proposal, citing CLERC).
- **Supervised:** Use `jhu-clsp/LegalBERT-DPR-CLERC-ft` or `jhu-clsp/BERT-DPR-CLERC-ft` and run full `python -m setup.run_baselines --dpr` to reproduce paper-style numbers.

## How to reproduce

1. `pip install -r setup/requirements.txt`
2. **DPR (real):** `python -m setup.run_baselines --dpr --out-dir setup/data/runs`
3. **BM25:** Build Pyserini index from CLERC collection (see setup/README.md), then `python -m setup.run_baselines --bm25 --bm25-index <path> --out-dir setup/data/runs`
4. Metrics written to `setup/data/runs/baseline_metrics.json`
