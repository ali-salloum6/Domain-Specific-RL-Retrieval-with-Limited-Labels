import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
import random

_TOP = Path(__file__).resolve().parent.parent
if str(_TOP) not in sys.path:
    sys.path.insert(0, str(_TOP))

from rl.data import prepare_rl_data
from rl.policy import DPRPolicy, rl_disk_chunk_cache_path
from setup.metrics import evaluate

def train(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    
    print("Loading data...")
    train_dataset, test_dataset, qrels = prepare_rl_data(
        train_ratio=0.8,
        seed=args.seed,
        bm25_top_k=args.bm25_top_k,
        max_train_queries=None if args.max_train_queries == 0 else args.max_train_queries,
        max_test_queries=None if args.max_test_queries == 0 else args.max_test_queries,
    )
    print(
        f"Train queries: {len(train_dataset)}, Test queries: {len(test_dataset)} "
        f"(BM25 top-{args.bm25_top_k} per query)"
    )
    
    print(f"Initializing policy model {args.model_name}...")
    model = DPRPolicy(model_name=args.model_name, device=device)
    model.train()
    n_loaded = model.load_chunk_cache(args.embedding_cache, args.max_chunks)
    if n_loaded:
        print(f"Loaded {n_loaded} cached doc chunk tensors from {args.embedding_cache}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    baseline = 0.0
    beta = 0.9 # moving average parameter
    
    out_dir = Path("setup/data/rl_checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        epoch_rewards = []
        epoch_losses = []
        
        optimizer.zero_grad()
        
        # Shuffle train set
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        from tqdm import tqdm
        pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, idx in enumerate(pbar):
            item = train_dataset[idx]
            qid = item["qid"]
            query_text = item["query_text"]
            candidate_ids = item["candidate_ids"]
            candidate_texts = item["candidate_texts"]
            
            # Forward pass
            probs, scores = model(
                query_text=query_text,
                candidate_ids=candidate_ids,
                candidate_texts=candidate_texts,
                max_chunks_per_doc=args.max_chunks,
                temperature=args.temperature,
                batch_size=args.chunk_batch_size,
                use_chunk_cache=not args.no_chunk_cache,
            )
            
            num_candidates = len(candidate_ids)
            k = min(args.k, num_candidates)
            if k == 0:
                continue
                
            # Gumbel-Softmax Sampling (O(|C| log |C|)) for N independent rankings
            N = args.n_rankings
            # Generate Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand((N, num_candidates), device=device) + 1e-10) + 1e-10)
            # Add noise to the raw document scores (logits)
            noisy_scores = (scores / args.temperature).unsqueeze(0) + gumbel_noise
            # Sort to get the sampled rankings
            sampled_rankings = torch.argsort(noisy_scores, dim=-1, descending=True)
            
            # Extract top-k for each ranking
            top_k_rankings = sampled_rankings[:, :k]
            
            # Evaluate reward for each ranking
            rewards = []
            for i in range(N):
                sampled_docs = [candidate_ids[idx.item()] for idx in top_k_rankings[i]]
                run = {qid: sampled_docs}
                reward_dict = evaluate(run, qrels, ks=[args.k])
                rewards.append(reward_dict[args.reward_metric])
            
            rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            
            # Compute Intra-Query Leave-One-Out Baseline
            if N > 1:
                # b_i = (sum(R) - R_i) / (N - 1)
                sum_rewards = rewards.sum()
                baselines = (sum_rewards - rewards) / (N - 1)
            else:
                baselines = torch.zeros_like(rewards)
            
            # Compute Plackett-Luce Log-Probability for the top-k sequence
            # log P(d_1, ..., d_k) = sum_{j=1}^k ( s_{d_j} - log sum_{m in remaining} exp(s_m) )
            log_probs = []
            for i in range(N):
                ranking_indices = top_k_rankings[i]
                
                # Create a mask for available items at each step
                # Initially all items are available
                available_mask = torch.ones(num_candidates, device=device, dtype=torch.bool)
                
                seq_log_prob = 0.0
                for step in range(k):
                    chosen_idx = ranking_indices[step]
                    
                    # Log prob of choosing this item from available ones
                    # log( exp(s_chosen) / sum_{available} exp(s) )
                    # = s_chosen - logsumexp(s_available)
                    
                    # We use the scaled scores (logits)
                    logits = scores / args.temperature
                    
                    # Get logits of available items
                    available_logits = logits[available_mask]
                    
                    # Compute logsumexp
                    log_sum_exp = torch.logsumexp(available_logits, dim=0)
                    
                    # Add to sequence log prob
                    step_log_prob = logits[chosen_idx] - log_sum_exp
                    seq_log_prob = seq_log_prob + step_log_prob
                    
                    # Mark chosen item as unavailable for next steps (avoid inplace modification of tensor used in backward)
                    new_mask = available_mask.clone()
                    new_mask[chosen_idx] = False
                    available_mask = new_mask
                    
                log_probs.append(seq_log_prob)
                
            log_probs = torch.stack(log_probs)
            
            # Policy Gradient Loss
            if args.algo == "pg_rank":
                # Neural PG-RANK: - 1/N * sum_{i=1}^N (R_i - b_i) * log pi(r_i)
                advantages = rewards - baselines
                loss = - (advantages * log_probs).mean()
            elif args.algo == "ppg":
                # Pairwise Policy Gradient: - 1/(2*N*(N-1)) * sum_{i!=j} (R_i - R_j) * (log P_i - log P_j)
                # Note: This is mathematically proportional to the mean-baseline REINFORCE, 
                # but directly computes the pairwise differences as proposed in the PPG paper.
                loss = 0.0
                if N > 1:
                    for i in range(N):
                        for j in range(N):
                            if i != j:
                                loss = loss - (rewards[i] - rewards[j]) * (log_probs[i] - log_probs[j])
                    loss = loss / (2 * N * (N - 1))
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            loss = loss / args.batch_size
            loss.backward()
            
            epoch_rewards.append(rewards.mean().item())
            epoch_losses.append(loss.item() * args.batch_size)
            
            if (step + 1) % args.batch_size == 0 or (step + 1) == len(indices):
                optimizer.step()
                optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                
            if (step + 1) % 5 == 0 or (step + 1) == len(indices):
                hitr = model._chunk_cache_hits
                miss = model._chunk_cache_misses
                tot = max(1, hitr + miss)
                pbar.set_postfix({
                    "rew": f"{np.mean(epoch_rewards[-10:]):.4f}",
                    "loss": f"{np.mean(epoch_losses[-10:]):.4f}",
                    "chit": f"{100.0 * hitr / tot:.0f}%",
                    "cdocs": len(model._chunk_cache),
                })

        print(f"Epoch {epoch+1} finished. Avg Reward: {np.mean(epoch_rewards):.4f} | Avg Loss: {np.mean(epoch_losses):.4f}")
        print(
            f"Chunk cache: {len(model._chunk_cache)} unique docs, "
            f"hits={model._chunk_cache_hits}, misses={model._chunk_cache_misses}"
        )
        if args.embedding_cache:
            model.save_chunk_cache(args.embedding_cache, args.max_chunks)
            print(f"Saved chunk embedding cache to {args.embedding_cache}")

        # Save checkpoint
        ckpt_path = out_dir / f"rl_epoch_{epoch+1}_{args.algo}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="pg_rank", choices=["pg_rank", "ppg"], help="RL algorithm to use: pg_rank (Neural PG-RANK) or ppg (Pairwise Policy Gradient)")
    parser.add_argument("--model_name", type=str, default="jhu-clsp/LegalBERT-DPR-CLERC-ft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of queries before optimizer step")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_chunks", type=int, default=10, help="Cap chunks per doc to save compute")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10, help="Number of documents to sample per query")
    parser.add_argument("--n_rankings", type=int, default=4, help="Number of independent rankings to sample per query for intra-query baseline")
    parser.add_argument("--reward_metric", type=str, default="mrr", choices=["mrr", "recall@1", "recall@5", "recall@10"])
    parser.add_argument("--chunk_batch_size", type=int, default=4, help="Batch size for chunk processing to avoid OOM")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=50,
        help="Use only first K BM25 documents per query as RL action space (speed). 0 = use full list.",
    )
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=0,
        help="Cap training queries after train split. 0 = use all training split queries (~620 on LegalBench-RAG-mini).",
    )
    parser.add_argument(
        "--max-test-queries",
        type=int,
        default=0,
        help="Cap test queries after split. 0 = use full test split.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=str,
        default="auto",
        help="'auto' = path from --model-name and --max-chunks (+ chunking constants in rl/policy.py); "
        "else explicit .pt path; empty string disables disk load/save.",
    )
    parser.add_argument(
        "--no-chunk-cache",
        action="store_true",
        help="Disable in-memory chunk reuse (re-encode every forward; slow).",
    )
    args = parser.parse_args()
    
    if args.device == "cpu":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"

    if args.embedding_cache == "":
        args.embedding_cache = None
    elif args.embedding_cache == "auto":
        args.embedding_cache = rl_disk_chunk_cache_path(_TOP, args.model_name, args.max_chunks)
        print(f"Embedding disk cache (auto): {args.embedding_cache}", flush=True)
    if args.embedding_cache:
        Path(args.embedding_cache).parent.mkdir(parents=True, exist_ok=True)

    train(args)
