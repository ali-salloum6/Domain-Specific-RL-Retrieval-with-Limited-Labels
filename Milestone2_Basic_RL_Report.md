# Milestone 2 Report: Basic RL for Domain-Specific Retrieval

**Project:** Domain-Specific RL Retrieval with Limited Labels (Joint RL + IR)  
**Team:** Ali Salloum, Yazan Kbaili  
**Milestone:** Phase 2 — Policy Gradient (REINFORCE) on DPR  
**Date:** April 2026  
**Repository:** [github.com/ali-salloum6/Domain-Specific-RL-Retrieval-with-Limited-Labels](https://github.com/ali-salloum6/Domain-Specific-RL-Retrieval-with-Limited-Labels)

---

## 1. What was Implemented

We have implemented a **sample-efficient Policy Gradient (REINFORCE) algorithm** to adapt our Dense Passage Retriever (DPR) to a non-differentiable task-specific metric (e.g., Mean Reciprocal Rank - MRR). 

### 1.1 Methodology

Following the methodology outlined in our proposal (based on ICLR 2024 work on adapting retrievers via RL), our formulation acts as a neural reranker over a restricted action space:
- **State**: The user query $q$.
- **Action Space**: CLI **`--bm25-top-k`** (default **50**; `0` = full BM25 list). Smaller *K* is faster; larger *K* is a harder reranking task.
- **Policy ($\pi$)**: Our `DPRPolicy` (in `rl/policy.py`) wraps the `jhu-clsp/LegalBERT-DPR-CLERC-ft` bi-encoder. It computes dot-product similarities between the query and chunked documents, applying MaxP aggregation over the chunks. The similarities are scaled by a temperature parameter and passed through a Softmax over that candidate set.
- **Action**: We sample $k$ documents (e.g., $k=10$) from this probability distribution.
- **Reward ($R$)**: We compute a non-differentiable IR metric (MRR, or Recall@k) on the sampled ranking using our existing `setup/metrics.py` evaluation pipeline.
- **Update**: We optimize the policy using the REINFORCE loss: $\mathcal{L} = - (R - b) \sum \log \pi(d_{sampled}|q)$, where $b$ is a moving-average baseline to reduce variance.

### 1.2 Data and Compute Optimizations

To ensure training is practical on a laptop without sacrificing scale:
- **Train query cap**: **`--max-train-queries`** — default **`0`** uses the **full** post-split training set (~620 queries on LegalBench-RAG-mini). Set e.g. **`100`** for a quick pilot.
- **Chunk embedding cache**: Document chunks are encoded **once per `doc_id`** and reused (in-memory). **Disk** path defaults to **`auto`**: e.g. `setup/data/rl_cache/doc_chunks_jhu-clsp_LegalBERT-DPR-CLERC-ft_mc10_w250_o50.pt` (from **`rl_disk_chunk_cache_path`** in `rl/policy.py`). Changing **`--model-name`** or **`--max-chunks`** selects a different file. Override with **`--embedding-cache /path/to/file.pt`**, or **`--embedding-cache ''`** to disable disk I/O.
- **Chunk capping**: **`--max-chunks`** (default 10) limits chunks per document during RL forwards.

### 1.3 Clean re-run from pretrained weights

To avoid mixing an old disk cache or RL checkpoints with a new training trajectory, remove artifacts and train from the **Hugging Face** initializer again:

```bash
rm -f setup/data/rl_cache/*.pt setup/data/rl_checkpoints/rl_epoch_*.pt
```

Then (from repo root, with the venv activated):

```bash
cd "/Users/ali/dev/uni/RL & IR"
.venv/bin/python -m rl.train
```

Defaults: **~620** train queries, **BM25 top-50**, **3** epochs, **`--embedding-cache auto`**. Wall times for a **clean** default run are in **§2.2**; **§2.1** is a smaller pilot (100 queries, top-30).

**After training**, evaluate (see §2.3 — **`--rl-checkpoint`** must **re-encode** corpus chunks; `setup/run_baselines.py` skips the zero-shot `dpr_embeddings_*.npy` cache in that mode so query and passage vectors match):

```bash
.venv/bin/python -m setup.run_baselines --dpr --rl-checkpoint setup/data/rl_checkpoints/rl_epoch_3.pt --out-dir setup/data/runs
```

---

## 2. Evaluation & Results

### 2.1 Pilot RL training run (completed)

Smaller configuration used first for pipeline validation: **100** training queries (after the 80/20 split), **BM25 top-30** candidates per query, **3** epochs, reward **MRR** (with **`--k` 10** sampled docs per query), **`--max-chunks` 10**, AdamW **`lr=1e-5`**, **`--batch-size` 4** (optimizer steps every 4 queries), chunk embedding cache enabled.

| Epoch | Wall time (100 steps) | Avg reward | Avg loss | tqdm snapshot (last window) |
| ----- | --------------------- | ---------- | -------- | ---------------------------- |
| 1 | ~3 m 13 s (~1.9 s/it) | 0.0003 | 0.0049 | rew≈0.0006, loss≈0.0404, cache hit ~85%, 438 unique docs |
| 2 | ~11 s (~8.7 it/s) | 0.0002 | −0.0043 | rew≈0.0002, cache hit ~93% |
| 3 | ~11 s (~8.9 it/s) | 0.0001 | −0.0032 | rew≈0.0000, cache hit ~95% |

**Chunk cache (end of run):** **438** unique `doc_id` tensors on disk/in-memory; **8 562** cumulative cache hits vs **438** misses over training (mostly first-touch encodes per doc). Checkpoints: `setup/data/rl_checkpoints/rl_epoch_{1,2,3}.pt`. *Early runs used a single legacy file `doc_chunk_embeddings.pt`; current code uses the **`auto`** filename pattern under §1.2.*

*Interpretation:* Average reward stays low because the metric is **MRR on only 10 sampled documents** (not the full 30), and **LegalBench qrels can list multiple relevant docs**, so sparse sampled lists often get near-zero credit. The pilot still shows the **cache warming** (epoch 1 vs 2–3) and stable optimization (loss magnitude small by epoch 3).

### 2.2 Full training split (default CLI, clean run)

**`python -m rl.train`** with defaults after **`rm`** of `setup/data/rl_cache/*.pt` and **`rl_epoch_*.pt`**: **~620** train queries, **BM25 top-50**, **3** epochs, **`--embedding-cache auto`**, **`--max-chunks` 10**, etc.

| Epoch | Wall time (620 steps) | Avg reward | Avg loss | Notes |
| ----- | --------------------- | ---------- | -------- | ----- |
| 1 | **5 m 08 s** (~2.0 it/s) | 0.0001 | −0.0006 | Cold chunk cache; **558** unique `doc_id`s cached |
| 2 | **1 m 28 s** (~7.0 it/s) | 0.0001 | −0.0009 | Near-total cache hits |
| 3 | **1 m 26 s** (~7.1 it/s) | 0.0001 | −0.0014 | Same |

**Total** ~**8 minutes** wall time on one MPS laptop for this run. Checkpoints: **`setup/data/rl_checkpoints/rl_epoch_{1,2,3}.pt`**.

### 2.3 Full-corpus retrieval evaluation (776 queries, post-fix only)

**Checkpoint:** **`rl_epoch_3.pt`** from the clean full-split run (§2.2). **Evaluation protocol:** chunked DPR + MaxP; **queries and passages** both encoded with weights loaded from the RL checkpoint. **`setup/run_baselines.py`** does **not** reuse **`dpr_embeddings_*.npy`** when **`--rl-checkpoint`** is set (that cache is zero-shot-only; mixing it with RL query vectors would invalidate metrics).

Merged metrics: **`setup/data/runs/baseline_metrics.json`** → **`rl_dpr`**. Run file: **`setup/data/runs/run_rl_dpr.json`**.

```text
python -m setup.run_baselines --dpr --rl-checkpoint setup/data/rl_checkpoints/rl_epoch_3.pt --out-dir setup/data/runs
```

| Metric     | BM25   | DPR (Zero-Shot) | DPR (RL, full split, epoch 3) |
| ---------- | ------ | --------------- | ------------------------------- |
| recall@1   | 0.7874 | 0.0284          | 0.0000                          |
| recall@5   | 0.8673 | 0.0966          | 0.0026                          |
| recall@10  | 0.8930 | 0.1211          | 0.0129                          |
| recall@100 | 0.9149 | 0.4820          | 0.0889                          |
| MRR        | 0.8245 | 0.0657          | 0.0045                          |
| nDCG@10    | 0.8403 | 0.0703          | 0.0043                          |

*BM25 / zero-shot DPR: Milestone 1. **RL** row: post-fix **`rl_dpr`** in `baseline_metrics.json` (same `evaluate` definition as Milestone 1).*

### 2.4 Key observations

The **RL-fine-tuned** retriever **lags zero-shot DPR** on full-corpus LegalBench-RAG-mini (table above). Likely factors: **REINFORCE** noise and low per-step reward, training on **BM25 top-50** with **sampled** short lists vs **corpus-wide** eval, **query-centric** updates with document-side chunk encodes **frozen** during RL, and only **3** epochs. The **pilot** (§2.1) was for **pipeline** validation only; **§2.2–2.3** are the numbers we report for Milestone 2 retrieval quality.

---

## 3. Next Steps (Milestone 3)

- **Reward Formulation**: We can experiment with blending downstream task accuracy (like citation correctness) into the reward signal, instead of just using standard IR metrics.
- **Scale Up**: If time and compute permit, we can extend the evaluation to the larger CLERC dataset.

---

## 4. Division of Work

- **RL Problem Formulation & Algorithm Design:** Yazan Kbaili
- **PyTorch Policy Implementation & OOM Optimizations:** Ali Salloum
- **Training Loop & Reward Integration:** Ali Salloum
- **Evaluation Engine Alignment:** Yazan Kbaili