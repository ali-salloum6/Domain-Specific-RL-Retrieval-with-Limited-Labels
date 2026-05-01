# Reproducing the Runs

This guide explains how to reproduce the experiments and results detailed in the `Milestone3_Final_Report.md`.

## 1. Environment Setup

Ensure you have Python 3.10+ installed. Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data Preparation

Download and prepare the LegalBench-RAG-mini dataset and build the BM25 index.

```bash
python -m setup.legalbench_data
python -m setup.build_bm25_index
```

## 3. Running Baselines (Milestone 1)

Run the BM25, Zero-Shot DPR, and Cross-Encoder baselines. This will populate `setup/data/runs/baseline_metrics.json`.

```bash
# Run BM25 and DPR
python -m setup.run_baselines --bm25 --dpr

# Run Cross-Encoder (using the BM25 run as a starting point)
python -m setup.run_baselines --cross-encoder --bm25-run setup/data/runs/run_bm25.json
```

## 4. Training the RL Models (Milestone 3)

We provide commands for both the Neural PG-RANK (`pg_rank`) and Pairwise Policy Gradient (`ppg`) algorithms.

### Local Training (Mac / MPS)

To train locally on a Mac with Apple Silicon (MPS), use lighter settings to ensure stability:

```bash
# Train using Neural PG-RANK (leave-one-out baseline)
python -m rl.train --algo pg_rank --device mps --epochs 3 --max_chunks 5 --batch_size 2 --chunk_batch_size 2

# Train using Pairwise Policy Gradient (PPG)
python -m rl.train --algo ppg --device mps --epochs 3 --max_chunks 5 --batch_size 2 --chunk_batch_size 2
```

Checkpoints will be saved in `setup/data/rl_checkpoints/`.

### Kaggle Training (CUDA)

To train on Kaggle (or any CUDA-enabled machine), you can use larger batch sizes and chunk limits. First, build the Kaggle bundle:

```bash
python scripts/build_kaggle_rl_bundle.py
```

Upload `kaggle_rl_minimal.zip` to Kaggle and run:

```bash
python -m rl.train --algo ppg --device cuda --epochs 3 --max_chunks 10 --batch_size 4 --chunk_batch_size 4
```

## 5. Evaluating the RL Models

Once training is complete, evaluate the checkpoints using the baseline script.

```bash
# Evaluate the local PG-RANK model
python -m setup.run_baselines --dpr --rl-checkpoint setup/data/rl_checkpoints/rl_epoch_3.pt --out-dir local_model_eval --dpr-device auto

# Evaluate the local PPG model
python -m setup.run_baselines --dpr --rl-checkpoint setup/data/rl_checkpoints/rl_epoch_3_ppg.pt --out-dir setup/data/runs_ppg_eval --dpr-device auto
```

The results will be saved in `local_model_eval/baseline_metrics.json` and `setup/data/runs_ppg_eval/baseline_metrics.json` respectively, matching the tables in the final report.
