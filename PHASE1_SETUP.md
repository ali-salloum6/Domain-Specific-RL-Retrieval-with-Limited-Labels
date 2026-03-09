# Phase 1 Setup (CLERC) — Quick Start

Phase 1 of the **Domain-Specific RL Retrieval** project: CLERC data, baselines, and evaluation.

- **Full instructions:** [setup/README.md](setup/README.md)
- **Data:** HuggingFace [jhu-clsp/CLERC](https://huggingface.co/datasets/jhu-clsp/CLERC) (loaded automatically)
- **Baselines:** BM25 (Pyserini), DPR (LegalBERT-DPR-CLERC-ft)
- **Metrics:** recall@k, MRR, nDCG (see `setup/metrics.py`, `setup/eval.py`)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r setup/requirements.txt
python -m setup.run_baselines --dpr --max-queries 50 --out-dir setup/data/runs   # or --mock for quick test
```

Baseline results: `setup/data/runs/baseline_metrics.json`, `setup/data/baseline_results_dev.md`.
