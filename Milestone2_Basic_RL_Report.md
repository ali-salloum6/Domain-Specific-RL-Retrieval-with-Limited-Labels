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

Defaults: **~620** train queries, **BM25 top-50**, **3** epochs, **`--embedding-cache auto`**. **Do not confuse with the pilot:** the **~3 min / ~11 s per epoch** timings in **§2.1** are for **100** queries and **top-30** BM25 only.

For the **default** full split with an **empty** chunk cache (clean `rm` above), epoch 1 does most first-time doc encodes. In one MPS trace, **~4 minutes** had elapsed at **259/620** steps (~**42%** of epoch 1), i.e. on the order of **~1 s/step** early on—so **epoch 1 alone is typically ~9–12+ minutes**, not 2–3 minutes. **Epochs 2–3** are usually **much faster** once per-`doc_id` chunk tensors are warm (often on the order of **~1.5–3 minutes** per epoch for 620 steps, depending on hit rate and thermals). **Ballpark ~15–40 minutes** total for three epochs on a laptop GPU, or longer if the machine throttles—use tqdm as ground truth for your run.

**After training**, evaluate and refresh `rl_dpr` in `baseline_metrics.json`:

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

### 2.2 Full training split (default CLI)

**`python -m rl.train`** uses **`--max-train-queries 0`** (~**620** train queries) and **`--bm25-top-k 50`**. **`--max-train-queries 100 --bm25-top-k 30`** reproduces the pilot footprint.

A **short** full-split wall time (~6–7 min for 3 epochs) only showed up when **~438** pilot docs were already on disk in a **legacy** cache file—i.e. not a true “from scratch” encode budget. A **clean** default run (§1.3) should be budgeted as **~15–40 minutes** (see §1.3). Treat metrics from checkpoints produced **after** that clean run as the canonical **full-split RL** line; re-run §1.3’s `run_baselines` command afterward.

### 2.3 Full-corpus retrieval evaluation (776 queries)

**Pilot checkpoint.** After training the pilot (§2.1), we evaluated the **epoch-3** weights with the standard DPR pipeline and merged metrics into **`setup/data/runs/baseline_metrics.json`** under **`rl_dpr`**. Retrieval run file: **`setup/data/runs/run_rl_dpr.json`**.

```text
python -m setup.run_baselines --dpr --rl-checkpoint setup/data/rl_checkpoints/rl_epoch_3.pt --out-dir setup/data/runs
```

| Metric     | BM25   | DPR (Zero-Shot) | DPR (RL pilot, epoch 3) |
| ---------- | ------ | --------------- | ----------------------- |
| recall@1   | 0.7874 | 0.0284          | 0.0000                  |
| recall@5   | 0.8673 | 0.0966          | 0.0477                  |
| recall@10  | 0.8930 | 0.1211          | 0.0619                  |
| recall@100 | 0.9149 | 0.4820          | 0.1649                  |
| MRR        | 0.8245 | 0.0657          | 0.0182                  |
| nDCG@10    | 0.8403 | 0.0703          | 0.0270                  |

*BM25 / zero-shot DPR: Milestone 1 baselines. **RL pilot** row: `setup/metrics.evaluate` on the merged **`rl_dpr`** block in `baseline_metrics.json` (same protocol: chunked DPR, MaxP, 776 queries).*

### 2.4 Key observations

The **pilot** RL checkpoint **underperforms zero-shot DPR** on full-corpus retrieval (table above). Plausible drivers: **only 100** training queries, **REINFORCE** noise, updates that mostly affect the **query** side while chunk encodes are reused, and **train/eval mismatch** (policy on **top-30** + **sampled** lists vs **corpus-wide** DPR eval).

**Full-split** `rl_dpr` numbers in `baseline_metrics.json` should be refreshed after the **clean** training in §1.3 (old checkpoints on disk were deleted to match that protocol).

---

## 3. Next Steps (Milestone 3)

- **Sample Efficiency Ablations**: We will run training sweeps using varying fractions of the training labels (e.g., 10%, 25%, 50%, 100%) to plot sample-efficiency curves.
- **Reward Formulation**: We can experiment with blending downstream task accuracy (like citation correctness) into the reward signal, instead of just using standard IR metrics.
- **Scale Up**: If time and compute permit, we can extend the evaluation to the larger CLERC dataset.

---

## 4. Division of Work

- **RL Problem Formulation & Algorithm Design:** Yazan Kbaili
- **PyTorch Policy Implementation & OOM Optimizations:** Ali Salloum
- **Training Loop & Reward Integration:** Ali Salloum
- **Evaluation Engine Alignment:** Yazan Kbaili