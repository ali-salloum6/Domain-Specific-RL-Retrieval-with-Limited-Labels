# Phase 1 Setup: CLERC Retrieval Baseline

Reproducible setup for the **Domain-Specific RL Retrieval** project (Phase 1): CLERC data, evaluation pipeline (recall@k, MRR, nDCG), and BM25/DPR baselines.

## Environment

- **Python:** 3.10+ (tested on 3.11)
- **Clone CLERC:** `git clone https://github.com/bohanhou14/CLERC.git` (optional; data is loaded from HuggingFace)

From the **repository root** (parent of `setup/`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r setup/requirements.txt
```

Optional for BM25 index build: Java 11+ (for Pyserini). Optional for faster DPR: `faiss-gpu` instead of `faiss-cpu`.

## Data (HuggingFace)

CLERC is loaded from [jhu-clsp/CLERC](https://huggingface.co/datasets/jhu-clsp/CLERC). No manual download needed.

**Splits and qrels (official):**

| Split / resource | File |
|------------------|------|
| Queries (direct) | `queries/test.single-removed.direct.tsv` |
| Queries (indirect) | `queries/test.single-removed.indirect.tsv` |
| Qrels (doc-level) | `qrels/qrels-doc.test.direct.tsv`, `qrels-doc.test.indirect.tsv` |
| Qrels (passage-level) | `qrels/qrels-passage.test.direct.tsv`, `qrels-passage.test.indirect.tsv` |
| Collection (doc) | `collection/collection.doc.tsv.gz` |
| Collection (passage) | `collection/collection.passage.tsv.gz` |

**Load in Python:**

```python
from setup.clerc_data import load_queries_and_qrels_hf, load_clerc_from_hf

# Retrieval eval: queries + qrels (doc-level, direct)
queries_df, qrels = load_queries_and_qrels_hf("direct", "doc")
# queries_df: columns qid, text
# qrels: dict qid -> set of relevant doc ids
```

## Evaluation metrics

Recall@k (k=1,5,10,100), MRR, nDCG@10. Same interface used for Phase 2 RL rewards.

```python
from setup.metrics import evaluate

# run: dict qid -> list of retrieved doc/passage ids
metrics = evaluate(run, qrels, ks=[1, 5, 10, 100])
# -> {"recall@1": ..., "recall@5": ..., "mrr": ..., "ndcg@10": ...}
```

Or use the eval API with a run file or callable retriever:

```python
from setup.eval import evaluate
metrics = evaluate("setup/data/runs/run_dpr.json", query_type="direct", qrels_level="doc")
```

## Baselines

### DPR (zero-shot / fine-tuned)

Uses [jhu-clsp/LegalBERT-DPR-CLERC-ft](https://huggingface.co/jhu-clsp/LegalBERT-DPR-CLERC-ft) or [jhu-clsp/BERT-DPR-CLERC-ft](https://huggingface.co/jhu-clsp/BERT-DPR-CLERC-ft). Encodes queries and passage collection, then retrieves by similarity.

```bash
# Quick run: minimal corpus from qrels (placeholder text; fast, low metrics)
python -m setup.run_baselines --dpr --out-dir setup/data/runs

# Full baseline: first N passages from CLERC collection (real retrieval)
# Requires the 9GB collection cached; streams first N lines, no full load
python -m setup.run_baselines --dpr --dpr-corpus 10000 --out-dir setup/data/runs
```

Output: `setup/data/runs/run_dpr.json`, `baseline_metrics.json`.

### BM25 (Pyserini)

Requires a pre-built Pyserini index from the CLERC document (or passage) collection.

1. **Get collection:** Either download from HuggingFace (e.g. use `load_collection_hf` and write TSV) or build from the [CLERC repo pipeline](https://github.com/bohanhou14/CLERC) (process_raw → build_collections → build_queries → build_qrels).
2. **Build index:** See [Pyserini docs](https://github.com/castorini/pyserini). Example (JSONL corpus with `id` and `contents`):

   ```bash
   python -m pyserini.index -collection JsonCollection \
     -generator DefaultLuceneDocumentGenerator \
     -input <path_to_corpus_jsonl> -index <path_to_index>
   ```

3. **Run baseline:**

   ```bash
   python -m setup.run_baselines --bm25 --bm25-index <path_to_index> --out-dir setup/data/runs
   ```

## Collection cache and download progress

The passage collection (`collection/collection.passage.tsv.gz`) is ~9 GB. It is **not** downloaded when you only load queries or qrels.

- **Check if the collection is cached:**  
  ```bash
  python -c "from setup.clerc_data import check_clerc_collection_cached; ok, path, size = check_clerc_collection_cached('passage'); print('cached:', ok, 'path:', path, 'size_GB:', size/1e9 if size else None)"
  ```
- **First time you run with `--dpr-corpus N`:** the script will download the file. HuggingFace’s `hf_hub_download` shows a **progress bar** by default (disable with `HF_HUB_DISABLE_PROGRESS_BARS=1` if needed). The script prints “collection not in cache. Downloading (~9 GB); progress below.” and then “using cached collection” on later runs.

## Lessons learned

- **Don’t load the full 9GB collection in one go.** `load_dataset(..., data_files={"col": "collection/collection.passage.tsv.gz"})` downloads and parses the entire file before you can slice; the process can sit with no output for 10+ minutes. Use `--dpr-corpus 0` for a quick run (minimal corpus from qrels) or `--dpr-corpus N` so the script uses `load_collection_hf_streaming()`, which reads only the first N lines from the cached `.tsv.gz`.
- **FAISS can segfault on ARM macOS (M1/M2).** Exit code 139 at “building index” is often FAISS. For corpora under 100k passages we use NumPy similarity only; for larger runs you may need `faiss-cpu` from conda or a different build.
- **LegalBERT-DPR-CLERC-ft is not a native sentence-transformers model.** You’ll see “Creating a new one with mean pooling.” The library falls back to mean pooling over token outputs; retrieval still works, but numbers may differ slightly from the paper.
- **Load queries and qrels in separate `load_dataset` calls.** Passing both in one `data_files` dict fails because the TSV schemas differ (queries: qid, text; qrels: qid, Q0, doc_id, rel).

## Reproducibility

- **Random seeds:** No sampling in the provided scripts; data order is fixed by HuggingFace. For any subsampling (e.g. `--max-queries`), set `PYTHONHASHSEED=0` and document the value.
- **Pinned deps:** `setup/requirements.txt`. For full CLERC repo pipeline, use `CLERC/requirements.txt` in the cloned repo.

## Dev-set baseline table (template)

| Model | recall@1 | recall@5 | recall@10 | recall@100 | MRR | nDCG@10 |
|-------|----------|----------|-----------|------------|-----|---------|
| BM25  | —        | —        | —         | —          | —   | —       |
| DPR (LegalBERT-CLERC-ft) | — | — | — | — | — | — |

Fill after running `run_baselines` and copying from `setup/data/runs/baseline_metrics.json`.

## References

- CLERC: [arXiv:2406.17186](https://arxiv.org/abs/2406.17186), [GitHub](https://github.com/bohanhou14/CLERC), [HuggingFace](https://huggingface.co/datasets/jhu-clsp/CLERC)
- Proposal: `Proposal_Idea4_Domain_Specific_RL_Retrieval.md`
