# Final Project Report: Domain-Specific RL Retrieval with Limited Labels

**Team:** Ali Salloum, Yazan Kbaili  
**Date:** May 2026  
**Repository:** [github.com/ali-salloum6/Domain-Specific-RL-Retrieval-with-Limited-Labels](https://github.com/ali-salloum6/Domain-Specific-RL-Retrieval-with-Limited-Labels)

---

## 1. Introduction and Project Pivot

The original proposal for this joint RL + IR project aimed to explore sample-efficient reinforcement learning for information retrieval in high-stakes domains (legal, medical) where labeled relevance is scarce. The initial plan included evaluating on both LegalBench-RAG and CLERC, performing label-fraction ablations, and potentially incorporating downstream QA rewards.

As the project progressed, we executed a **slight pivot** to focus deeply on the **algorithmic stability and optimization of the RL-IR pipeline** rather than broad scaling. Specifically:

1. **Document-Level Surrogate Task:** Instead of span-level snippet retrieval (which requires complex passage-level qrels), we standardized on a document-level surrogate task using LegalBench-RAG-mini. This allowed us to iterate much faster on the RL algorithms while maintaining valid evaluation metrics.
2. **Algorithmic Depth over Breadth:** We encountered significant variance and collapse issues with vanilla REINFORCE (Milestone 2). Consequently, Milestone 3 was dedicated to implementing advanced ranking algorithms (Neural PG-RANK, Plackett-Luce log-probabilities, Pairwise Policy Gradient) to stabilize training, rather than scaling to the 9GB CLERC dataset or performing the label-efficiency ablations.
3. **Detached Document Cache:** To make training feasible on consumer hardware (Mac MPS), we implemented a query-centric RL gradient approach where document embeddings are cached and detached, trading symmetric tower updates for memory/speed efficiency.

This pivot allowed us to successfully demonstrate that advanced RL techniques (like PPG) can significantly outperform zero-shot dense retrievers on a domain-specific legal benchmark.

---

## 2. Milestone 1: Baselines and Infrastructure

In the first phase, we established a robust evaluation pipeline and strong IR baselines on LegalBench-RAG-mini. 

- **Data & Evaluation:** We implemented custom data loaders and an evaluation engine computing recall@k, MRR, and nDCG@10. We aligned document-level qrels with the retrieval corpus.
- **BM25:** A true sparse baseline using Pyserini with a full Lucene index.
- **Zero-Shot DPR:** We used `jhu-clsp/LegalBERT-DPR-CLERC-ft`. To handle long legal documents (which exceed the 512-token limit), we implemented a chunking strategy (250 words/chunk, 50-word overlap) and MaxP aggregation.
- **Cross-Encoder:** A neural reranker (`ms-marco-MiniLM-L-6-v2`) over BM25 top-100 candidates, serving as our upper-bound baseline.

*Baseline Results (MRR):* BM25 (0.824), Zero-Shot DPR (0.066), Cross-Encoder (0.888). The zero-shot DPR performance highlighted the need for domain-specific adaptation via RL.

---

## 3. Milestone 2: Basic RL Implementation

In Milestone 2, we introduced a basic policy-gradient RL framework to fine-tune the dense retriever using non-differentiable IR metrics (MRR) as the reward.

- **Algorithm:** We used a standard REINFORCE algorithm over a softmax policy on BM25 top-50 candidates.
- **Baseline:** A global moving-average baseline was used for variance reduction.
- **Results:** The basic REINFORCE approach suffered from high variance and mode collapse. After 3 epochs, the model degraded significantly, dropping to an MRR of 0.004 (below the zero-shot DPR baseline of 0.066). This necessitated the algorithmic improvements in Milestone 3.

---

## 4. Milestone 3: Advanced RL and Pairwise Policy Gradient

To address the instability of Milestone 2, we overhauled the RL algorithm, moving towards listwise and pairwise approaches aligned with Neural PG-RANK.

- **Gumbel-Top-k Sampling & Plackett-Luce:** We replaced the single multinomial draw with Gumbel noise addition to sample multiple independent rankings per query. We implemented Plackett-Luce sequence log-probabilities for without-replacement list actions.
- **Intra-Query Leave-One-Out Baseline (`pg_rank`):** We replaced the global moving average with an intra-query baseline, computing advantages across the $N$ sampled rankings for a single query.
- **Pairwise Policy Gradient (`ppg`):** We implemented a pairwise loss that explicitly compares pairs of sampled document lists within the same query, computing the gradient as:
$$L = - \frac{1}{2 N(N-1)} \sum_{i \neq j} (R_i - R_j) (\log P_i - \log P_j)$$
- **Query-Centric Optimization:** To manage compute, we cached and detached document embeddings. The RL gradients only updated the query encoder, which proved highly effective and memory-efficient.

---

## 5. Results and Discussion

We evaluated the models on the LegalBench-RAG-mini document-level surrogate task. For full instructions on how to reproduce these runs, see the `README.md`.


| Metric         | Zero-Shot DPR | M2 REINFORCE | M3 Local (`pg_rank`) | M3 Local (PPG) |
| -------------- | ------------- | ------------ | -------------------- | -------------- |
| **recall@1**   | 0.028         | 0.000        | 0.253                | **0.268**      |
| **recall@10**  | 0.121         | 0.013        | 0.459                | **0.501**      |
| **recall@100** | 0.482         | 0.089        | 0.762                | **0.807**      |
| **MRR**        | 0.066         | 0.004        | 0.325                | **0.352**      |
| **nDCG@10**    | 0.070         | 0.004        | 0.350                | **0.379**      |


**Discussion:**

1. **Recovery from M2 Collapse:** The advanced algorithms in Milestone 3 successfully recovered from the REINFORCE collapse, dramatically outperforming the M2 model.
2. **Beating Zero-Shot:** Both `pg_rank` and `ppg` significantly outperformed the zero-shot DPR baseline (e.g., PPG achieved 0.352 MRR vs. 0.066 MRR). This validates our core hypothesis: policy-gradient RL can successfully adapt a dense retriever to a domain-specific task using non-differentiable rewards.
3. **PPG Superiority:** The Pairwise Policy Gradient (`ppg`) algorithm outperformed the leave-one-out baseline (`pg_rank`) across all metrics. By explicitly contrasting pairs of rankings, PPG provided a more stable and informative gradient signal.

---

## 6. Further Work

While we successfully demonstrated RL-driven retrieval adaptation, several avenues remain for future exploration:

1. **Span-Level Retrieval:** Our current pipeline operates on a document-level surrogate task. Future work should adapt the qrels, chunking strategy, and reward formulation to evaluate precise span/snippet retrieval, fully aligning with the original LegalBench-RAG vision.
2. **Downstream QA Rewards:** We optimized directly for IR metrics (MRR, recall). Incorporating end-to-end downstream rewards (e.g., LLM generation accuracy, citation correctness) would test if RL can optimize retrieval specifically for RAG generation quality.
3. **Scaling to CLERC:** With the algorithmic foundation stabilized, the pipeline can be scaled to the full 9GB CLERC dataset to test robustness on a much larger corpus.
4. **Label-Efficiency Ablations:** As proposed originally, running systematic ablations over the fraction of training labels (10%, 25%, 50%) would quantify the sample efficiency of the PPG algorithm compared to supervised contrastive learning.
5. **Symmetric Tower Updates:** Removing the detached document cache and updating both the query and document encoders (requiring more VRAM) could yield further performance gains.

---

## 7. Student Contributions

Both students contributed equally to the success of this project, collaborating closely on the theoretical formulation and practical implementation.

- **Yazan Kbaili:** Led the data engineering and baseline infrastructure (Milestone 1). Developed the LegalBench and CLERC data loaders, implemented the BM25/DPR/Cross-Encoder pipelines, and managed the Kaggle deployment and sanitization scripts for cloud training.
- **Ali Salloum:** Led the reinforcement learning theory and algorithmic implementation (Milestones 2 & 3). Formulated the evaluation metrics, implemented the initial REINFORCE pipeline, and developed the advanced Plackett-Luce, Neural PG-RANK, and Pairwise Policy Gradient (PPG) algorithms.

*Both students collaborated on debugging, hyperparameter tuning, running experiments, and writing the final reports.*