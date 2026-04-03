# Milestone 1 Report: First Baseline — Domain-Specific RL Retrieval

**Project:** Domain-Specific RL Retrieval with Limited Labels (Joint RL + IR)  
**Team:** Ali Salloum, Yazan Kbaili  
**Milestone:** Phase 1 — Setup and first retrieval baseline  
**Date:** April 2026  
**Repository:** [github.com/ali-salloum6/Domain-Specific-RL-Retrieval-with-Limited-Labels](https://github.com/ali-salloum6/Domain-Specific-RL-Retrieval-with-Limited-Labels)

---

## 1. What is Implemented and Validated

We have implemented and verified a full pipeline for evaluating retrieval models on **LegalBench-RAG-mini**. Initially, we considered CLERC, but passage embedding times and subsets evaluation complications led us to switch to LegalBench-RAG-mini for faster iteration and highly valid document-level evaluation metrics.

### 1.1 Environment and Data
- **Environment:** Python 3.10+ virtual environment (`.venv`) with all dependencies properly configured.
- **Data loading:** Implemented in `setup/legalbench_data.py`. We download and parse LegalBench-RAG, specifically configuring it for the "mini" subset (max 194 tests per benchmark, sorted by document) to accelerate baselines.
- **Evaluation Granularity:** We ensured valid evaluation by natively aligning document-level qrels with LegalBench-RAG-mini's retrieval corpus. The passages/documents retrieved correctly match the granularity of the ground truth labels.

### 1.2 Evaluation Pipeline
- **Metrics:** In `setup/metrics.py`, we implemented recall@k (k = 1, 5, 10, 100), MRR, and nDCG@10. For recall@k, each query contributes the fraction of its relevant documents that appear in the top-k ranked list (`|rel ∩ top-k| / |rel|`), and we average over all queries; when a benchmark lists multiple source documents for one question, this differs from a binary “any hit” notion.
- **Eval API:** `setup/eval.py` provides an evaluation entry point that accurately scores any retrieval run against dataset qrels. Metrics written to `setup/data/runs/baseline_metrics.json` are **merged** with any existing file so partial runs (for example cross-encoder only) do not erase BM25 or DPR entries.

### 1.3 Real Baselines
We have completely removed all mock fallback behaviors from the repository to ensure placeholder runs cannot be confused with real experiments. The following real baselines are implemented and running:

1. **BM25:** A true sparse baseline using Pyserini. We implemented a script (`setup/build_bm25_index.py`) to parse the LegalBench corpus into JSONL and build a complete Lucene index for fast searching.
2. **DPR with Document Chunking:** We run Dense Passage Retrieval using `jhu-clsp/LegalBERT-DPR-CLERC-ft` as an off-the-shelf domain-specific dense retriever. **Critically, we identified that Transformer context limits (512 tokens) were causing naive DPR to truncate long legal documents, severely hurting recall.** We implemented a robust chunking strategy (250 words per chunk with 50-word overlap) and MaxP aggregation to score full documents accurately. We also implemented a `.npy` caching mechanism to speed up iterative embedding runs.
3. **Cross-Encoder (Stronger Baseline):** We implemented a neural reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that reranks the top-100 documents from BM25. Like DPR, it uses the same 250-word / 50-word overlap chunks and MaxP aggregation so long contracts are not represented by a single truncated window. The implementation in `setup/run_baselines.py` includes **operational controls** we added for laptop-scale runs: optional **`--bm25-run path/to/run_bm25.json`** to reuse a saved BM25 run and skip Pyserini; **`--cross-encoder-max-chunks-per-doc N`** (with `0` meaning unlimited) to cap how many chunks per retrieved document are scored; **`--cross-encoder-device`** (`auto`, `cpu`, `mps`, `cuda`), **`--cross-encoder-batch-size`**, and optional **`--torch-threads`** for CPU runs; `CrossEncoder` uses **`max_length=512`**, and `model.predict(..., show_progress_bar=True)` surfaces per-query micro-batch progress (each query can show one tqdm “Batches: a/b” bar for its predict call).

**Cross-encoder: compute vs accuracy.** A chunk cap is a deliberate compromise: we take chunks **in order from the start of each document**, score only the first N, then apply MaxP over that **prefix**. That cuts latency and memory when a single contract yields hundreds of chunks. **Accuracy can suffer** relative to an uncapped run because the query’s best-matching span might lie in a later chunk (never scored). Setting N to `0` removes the cap and restores full MaxP over all chunks of that document, at higher resource cost.

**Baseline Results on LegalBench-RAG-mini** (776 queries; BM25 and DPR from `run_bm25.json` / `run_dpr.json`, cross-encoder from `run_cross_encoder.json`; all scored with `setup/metrics.evaluate`. The same cross-encoder figures are merged in `setup/data/runs/baseline_metrics.json`.)

| Metric     | BM25   | DPR (Chunked) | Cross-Encoder (over BM25) |
| ---------- | ------ | ------------- | ------------------------- |
| recall@1   | 0.7874 | 0.0284        | 0.8698                    |
| recall@5   | 0.8673 | 0.0966        | 0.9137                    |
| recall@10  | 0.8930 | 0.1211        | 0.9149                    |
| recall@100 | 0.9149 | 0.4820        | 0.9149                    |
| MRR        | 0.8245 | 0.0657        | 0.8883                    |
| nDCG@10    | 0.8403 | 0.0703        | 0.8951                    |

**Cross-encoder:** These numbers are from a **complete** rerank (non-empty lists for all 776 queries). Reranking uses BM25’s top-100 candidates per query; at k = 100, recall cannot exceed BM25’s first-stage recall on this pool, which is why recall@100 matches BM25 here.

---

## 2. What is Only Setup

- **CLERC Dataset Loaders:** We have data loading logic in `setup/clerc_data.py` ready for future use if we want to scale up to the massive 9GB CLERC dataset later in the project.
- **Baseline Scripts Architecture:** The unified baseline script (`setup/run_baselines.py`) is fully modular, allowing easy addition of new datasets and future RL retrievers.

---

## 3. What is Still Planned

- **Reinforcement Learning Implementation:** There is currently no RL implemented. The project has set up the necessary IR baselines and evaluation infrastructure to *support* RL in the next phase. The project makes no RL claims at this stage.
- **Reward Formulation:** We will formulate reward signals using the implemented metrics (recall@k, MRR, nDCG) to fine-tune the dense retriever via policy-gradient RL.
- **Sample Efficiency Ablations:** Once the RL pipeline is working, we will experiment with limited relevance labels to demonstrate sample efficiency.

---

## 4. Division of Work (Ali Salloum & Yazan Kbaili)

- **Theory & Metric Formulation:** Yazan Kbaili 
- **Data Loaders (LegalBench & CLERC):** Ali Salloum 
- **Baseline Implementations (BM25, DPR, Cross-Encoder):** Ali Salloum 
- **Evaluation Engine:** Yazan Kbaili
- **Documentation & Reporting:** Both
