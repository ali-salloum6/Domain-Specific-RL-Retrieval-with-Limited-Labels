# Milestone 3 — Running notes (progress log)

**Project:** Domain-specific RL retrieval (LegalBench-RAG-mini)  
**Purpose:** Single living document for what we implemented, ran, and measured during Milestone 3, so the final report and next steps stay aligned.

---

## 1. Where we started (Milestone 2 baseline)

Milestone 2 used **REINFORCE** over a softmax policy on BM25 top-50 candidates, with a **moving-average baseline**, sampling a short list per query, optimising toward **MRR** (and related metrics via `setup/metrics.py`). Full-corpus evaluation (776 queries) showed RL **below** zero-shot DPR:


| Metric     | BM25      | DPR (zero-shot) | DPR + RL (REINFORCE, epoch 3) |
| ---------- | --------- | --------------- | ----------------------------- |
| recall@1   | 0.787     | 0.028           | 0.000                         |
| recall@10  | 0.893     | 0.121           | 0.013                         |
| recall@100 | 0.915     | 0.482           | 0.089                         |
| **MRR**    | **0.824** | **0.066**       | **0.004**                     |
| nDCG@10    | 0.840     | 0.070           | 0.004                         |


*Source: `Milestone2_Basic_RL_Report.md` §2.3; eval on original local corpus layout.*

---

## 2. What we changed in Milestone 3 (algorithm & code)

**Goal:** Move from high-variance listwise REINFORCE toward ideas aligned with **Neural PG-RANK** (Plackett–Luce style policy, differentiable sampling, query-level variance reduction).

**Implemented in `rl/train.py` (defaults unchanged unless overridden on CLI):**

1. **Gumbel–top-k style sampling** — For each query, sample `N = --n_rankings` (default **4**) independent rankings by adding Gumbel noise to scaled document scores and sorting (instead of a single `multinomial` draw).
2. **Plackett–Luce log-probability** — For each sampled top-`k` document sequence, accumulate log-probabilities of successive choices without replacement (log-sum-exp over remaining candidates). Fixes autograd issues with in-place mask updates (clone before mutating).
3. **Intra-query leave-one-out baseline (Neural PG-RANK)** — Default `--algo pg_rank`. For each query, advantage `A_i = R_i - (sum_j R_j - R_i)/(N-1)` when `N > 1`, replacing the old global moving-average baseline for policy gradient variance reduction.
4. **Pairwise Policy Gradient (PPG)** — Added `--algo ppg`. Instead of using a baseline, explicitly computes the gradient by comparing pairs of document lists sampled within the same query: `L = - 1/(2*N*(N-1)) * sum_{i!=j} (R_i - R_j) * (log P_i - log P_j)`.

**Training CLI additions:** `--n_rankings` (default 4), `--algo` (choices: `pg_rank`, `ppg`).

**Evaluation improvements (`setup/run_baselines.py`):** `--dpr-device` (default `auto`, prefers CUDA then MPS) and `--dpr-batch-size` so RL-DPR eval actually runs on GPU when available after loading weights from disk.

**Kaggle packaging:** `scripts/build_kaggle_rl_bundle.py` builds `kaggle_rl_minimal.zip` (train) and `kaggle_eval_rl.zip` (`--eval`) with **sanitized** corpus pathnames (Kaggle rejects `|`, `&`, `:`, `#`, odd spaces in zip members). Benchmark JSONs and `run_bm25.json` are rewritten so `file_path` / doc ids stay consistent with on-disk names.

### 2.1 Precision: what the numbers mean (addressing feedback + claim scope)

This subsection records **limitations and fixes** so we do not over-state results in the final report.

**A. Training reward vs full qrels (still true in Milestone 3)**  
During training, `evaluate(...)` is called on a **run that contains only the top‑`k` sampled document ids** for that query, while **qrels** still list **all** relevant document ids for the query (`rl/train.py`). Metrics such as recall are therefore computed on that **truncated** retrieved list against the **full** relevant set. That can **cap** recall and depress scalars even when the slate is reasonable; it also means **training reward trajectories are not directly comparable in magnitude** to full-corpus evaluation MRR/recall reported from `setup/run_baselines.py`. Milestone 3 improves **how the reward enters the gradient** (PPG / PL-style terms); it does **not** redefine the reward as full-corpus retrieval.

**B. Policy gradient for top‑`k` without replacement (fixed in Milestone 3)**  
Milestone 2’s REINFORCE-style setup risked a **mismatch** between sampling and log-probability for **without-replacement** list actions. Milestone 3 uses a **sequence** log-probability over successive choices from the remaining candidates (Plackett–Luce–style), consistent with Gumbel–rank sampling, for both `--algo pg_rank` and `--algo ppg`.

**C. Cached, detached document embeddings (not changed in Milestone 3)**  
`rl/policy.py` encodes passage chunks under `torch.no_grad()`, caches them (CPU), and applies `detach()` on chunk embeddings before scoring (`forward`). RL gradients from the training loss therefore **do not update passage-encoder weights** for those cached representations; optimization is **query-centric** (scores still depend on `q_emb`). That is a deliberate **memory/speed** tradeoff. Eval-time full-corpus runs still encode queries and passages through the loaded checkpoint for ranking; claims should **not** imply symmetric “both towers receive RL gradients” unless we change this path.

**D. Document-level surrogate vs LegalBench-RAG span retrieval (claim boundary)**  
We resolved an earlier **granularity mismatch** by standardising on a **document-level** task: qrels and corpus paths refer to **whole documents** (see `setup/legalbench_data.py`: relevance from benchmark `file_path` strings; `load_corpus` loads `.txt` by document). Chunking inside `DPRPolicy` / `run_baselines` is an **internal encoding** choice (MaxP over chunks), not the same problem as **retrieving precise relevant spans or snippets** as in the original LegalBench-RAG-style formulation.

Therefore:

- Reported **baselines and RL numbers are valid for this surrogate: document-level retrieval on LegalBench-RAG-mini** under our pipeline (BM25 slate, metrics in `setup/metrics.py`, etc.).
- They are **not** interchangeable, without extra work, with claims about **span-level legal RAG** or “retrieving exact relevant passages” as the primary evaluation object.
- **Future work** (if we want those claims): passage- or span-level qrels, retrieval targets, and metrics aligned to **snippet** relevance—not only document ids.

---

## 3. Experiments we actually ran

### 3.1 Local training (Mac, MPS)

Command used (lighter settings for laptop stability):

```bash
source .venv/bin/activate && python -m rl.train --device mps --epochs 3 \
  --max_chunks 5 --batch_size 2 --chunk_batch_size 2
```

**Checkpoints:** `setup/data/rl_checkpoints/rl_epoch_{1,2,3}.pt` (same directory convention as `rl/train.py`; ~433 MB per epoch file in our environment).

### 3.2 Kaggle training (CUDA)

Notebook-style command (defaults from `rl/train.py`):

```bash
python -m rl.train --device cuda --epochs 3
```

So Kaggle used `--max_chunks 10`, `--batch_size 4`, `--chunk_batch_size 4` — not the same training recipe as the local run above.

**Checkpoint saved locally after download:** `setup/data/rl_checkpoints/kaggle_checkpoints/rl_epoch_3.pt`

### 3.3 Evaluation (both models on Kaggle)

Both checkpoints were evaluated with the **sanitized** eval bundle (`kaggle_eval_rl.zip` contents: `rl/`, `setup/`, `setup/data/legalbench_data/`, etc.) and `python -m setup.run_baselines --dpr --rl-checkpoint …` (full query set unless you overrode `--max-queries`).

**Exported metric files in this repo:**

- `local_model_eval/baseline_metrics.json` — RL-DPR from **local** training checkpoint  
- `kaggle_model_eval/baseline_metrics.json` — RL-DPR from **Kaggle** training checkpoint

### 3.4 Pairwise Policy Gradient (PPG) experiment

We added a `--algo ppg` flag to `rl/train.py` to implement the Pairwise Policy Gradient algorithm.
This computes the loss by explicitly comparing pairs of sampled rankings:
$L = - \frac{1}{2 N(N-1)} \sum_{i \neq j} (R_i - R_j) (\log P_i - \log P_j)$

Command to run locally (MPS):

```bash
source .venv/bin/activate && python -m rl.train --algo ppg --device mps --epochs 3 \
  --max_chunks 5 --batch_size 2 --chunk_batch_size 2
```

Checkpoints will be saved as `setup/data/rl_checkpoints/rl_epoch_{1,2,3}_ppg.pt`.

### 3.5 PPG — training & evaluation (local)

Same laptop recipe as §3.1 (`--max_chunks 5`, `--batch_size 2`, `--chunk_batch_size 2`, MPS), with `--algo ppg`. Full training completed; evaluation:

```bash
python -m setup.run_baselines --dpr \
  --rl-checkpoint setup/data/rl_checkpoints/rl_epoch_3_ppg.pt \
  --out-dir setup/data/runs_ppg_eval \
  --dpr-device mps
```

**Metrics file:** `setup/data/runs_ppg_eval/baseline_metrics.json` (`rl_dpr` block).

---

## 4. Results snapshot (RL-DPR only)


| Metric     | **M2 REINFORCE** (published) | **M3 local (`pg_rank`)** (`local_model_eval`) | **M3 Kaggle ckpt** (`kaggle_model_eval`) | **M3 PPG local** (`runs_ppg_eval`) |
| ---------- | ---------------------------- | --------------------------------------------- | ---------------------------------------- | ---------------------------------- |
| recall@1   | 0.000                        | 0.253                                         | 0.032                                    | **0.268**                          |
| recall@5   | 0.003                        | 0.398                                         | 0.107                                    | **0.436**                          |
| recall@10  | 0.013                        | 0.459                                         | 0.133                                    | **0.501**                          |
| recall@100 | 0.089                        | 0.762                                         | 0.486                                    | **0.807**                          |
| **MRR**    | **0.004**                    | 0.325                                         | 0.072                                    | **0.352**                          |
| nDCG@10    | 0.004                        | 0.350                                         | 0.078                                    | **0.379**                          |


**M2 zero-shot DPR (same published table):** MRR **0.066**, recall@10 **0.121** — useful as a second reference line.

### 4.1 How to read this table

- **Versus Milestone 2 REINFORCE:** The **local Milestone 3 checkpoint** is dramatically better on every listed metric. So yes — we already have **clear progress** relative to the pure REINFORCE setup from Milestone 2, *under the Milestone 3 eval export you saved* (`local_model_eval`).
- **Versus published zero-shot DPR:** The **local** M3 model is also **much stronger** than the M2 table’s zero-shot row (MRR 0.33 vs 0.07).  
- **Kaggle-trained M3 model:** Metrics are **roughly in the same band as zero-shot DPR** from the old table — so that run did **not** reproduce the local improvement; likely causes include **different training hyperparameters** (see §3) and/or run quality, not “CUDA vs MPS” alone.
- **`pg_rank` vs PPG (both local, same §3.1 training recipe):** The **PPG** eval (`runs_ppg_eval`) is **stronger on every metric** in the table than the earlier **`pg_rank`** export (`local_model_eval`) — e.g. MRR **0.352** vs **0.325**, recall@10 **0.501** vs **0.459**. Same eval script and (local) corpus; different checkpoints — so this is a fair **algorithm A/B** at the training settings you used.
- **Caveat on strict apples-to-apples:** Published M2 numbers used the **original** corpus filenames. The **Kaggle** M3 column (`kaggle_model_eval`) used the **sanitized** Kaggle-safe tree. **`local_model_eval`**, **`runs_ppg_eval`**, and M2 published rows are on the **original** local corpus layout — so compare **`pg_rank` vs PPG** directly; compare **local vs Kaggle M3** only with the sanitization caveat in mind.
- **Claim scope:** See **§2.1** — table numbers support **document-level** surrogate retrieval on this setup, not span-level LegalBench-RAG claims without further work.

---

## 5. Anchors for moving forward

1. **Lock a single “reporting” training config** for any fair comparison (same `--max_chunks`, `--batch_size`, `--chunk_batch_size`, epochs, seed, `n_rankings`) on Kaggle *or* local — then re-evaluate both checkpoints with the same bundle.
2. **Final report (`Milestone3_Final_Report.md`)** should cite: algorithm changes (§2), **§2.1 (scope / feedback / surrogate task)**, commands (§3), metrics (§4), and the sanitized-eval vs M2-published-eval distinction where relevant.
4. **Span-level LegalBench-RAG (optional):** If the project must support **snippet/span** claims, add passage-level qrels + eval aligned to retrieved spans (see §2.1D); not in scope for current numbers.

---

## 6. Changelog (this file)


| Date       | Note                                                                                                                               |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 2026-05-01 | Initial running log: M3 algorithm summary, train/eval commands, `local_model_eval` / `kaggle_model_eval` metrics vs M2 table.      |
| 2026-05-01 | Added Pairwise Policy Gradient (`--algo ppg`) implementation and experiment details.                                               |
| 2026-05-01 | Recorded PPG full train + eval: `setup/data/runs_ppg_eval/baseline_metrics.json`; beats local `pg_rank` row on all listed metrics. |
| 2026-05-01 | Added §2.1: precise scope on training reward vs qrels, PL fix, detached doc cache, document-level surrogate vs span-level LegalBench-RAG. |

