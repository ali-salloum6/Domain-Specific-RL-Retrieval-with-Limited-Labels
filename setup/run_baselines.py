"""
Run baselines (BM25, DPR, Cross-Encoder) on LegalBench-RAG-mini or CLERC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root on path for "from setup.xxx" when run as script
_TOP = Path(__file__).resolve().parent.parent
if _TOP not in sys.path:
    sys.path.insert(0, str(_TOP))

from setup.metrics import evaluate

def run_bm25(
    queries_df,
    qrels: dict,
    collection_path: str | Path | None = None,
    top_k: int = 100,
    k1: float = 0.9,
    b: float = 0.4,
) -> dict[str, list[str]]:
    """Run BM25 retrieval via Pyserini."""
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as e:
        raise RuntimeError("BM25 requires Pyserini. Please install dependencies.") from e

    if not collection_path or not Path(collection_path).exists():
        raise ValueError("BM25 requires --bm25-index pointing to an existing Pyserini index.")

    searcher = LuceneSearcher(str(collection_path))
    searcher.set_bm25(k1=k1, b=b)
    run = {}
    for _, row in queries_df.iterrows():
        qid = str(row["qid"])
        hits = searcher.search(row["text"], k=top_k)
        run[qid] = [h.docid for h in hits]
    return run

def _resolve_dpr_device(dpr_device: str | None) -> str:
    """Pick a device for DPR encode (RL or zero-shot)."""
    import torch

    if dpr_device is None or dpr_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return dpr_device


def run_dpr(
    queries_df,
    corpus_ids: list[str],
    corpus_texts: list[str],
    model_name: str = "jhu-clsp/LegalBERT-DPR-CLERC-ft",
    top_k: int = 100,
    batch_size: int = 32,
    cache_dir: str | Path | None = None,
    rl_checkpoint: str | None = None,
    dpr_device: str | None = "auto",
) -> dict[str, list[str]]:
    """Run DPR retrieval using HF encoder with passage chunking and caching."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        raise RuntimeError("DPR requires sentence-transformers and numpy.") from e

    q_texts = queries_df["text"].tolist()
    q_ids = queries_df["qid"].astype(str).tolist()

    print("loading model", flush=True)
    if rl_checkpoint:
        import torch
        from rl.policy import DPRPolicy

        print(f"Loading RL checkpoint from {rl_checkpoint}...")
        policy = DPRPolicy(model_name)
        policy.load_state_dict(torch.load(rl_checkpoint, map_location="cpu"))
        model = policy.encoder
        dev = _resolve_dpr_device(dpr_device)
        model.to(dev)
        print(f"moved RL encoder to {dev}", flush=True)
    else:
        model = SentenceTransformer(model_name)
        dev = _resolve_dpr_device(dpr_device)
        if dev != "cpu":
            model.to(dev)
            print(f"moved DPR encoder to {dev}", flush=True)
    _dev = getattr(model, "_target_device", None) or getattr(model, "device", None) or next(model.parameters()).device
    print("using device: %s" % _dev, flush=True)

    print("encoding queries", flush=True)
    q_emb = model.encode(q_texts, batch_size=batch_size, show_progress_bar=True)
    
    print("chunking passages", flush=True)
    chunk_texts = []
    chunk_doc_ids = []
    
    def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    for doc_id, text in zip(corpus_ids, corpus_texts):
        chunks = chunk_text(text)
        chunk_texts.extend(chunks)
        chunk_doc_ids.extend([doc_id] * len(chunks))
        
    p_emb = None
    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_safe_name = model_name.replace("/", "_")
        cache_path = cache_dir / f"dpr_embeddings_{model_safe_name}.npy"
        # Chunk cache is from a pretrained run; RL moves query (and possibly shared) weights —
        # reusing passage vectors would score queries in the wrong space.
        if rl_checkpoint:
            if cache_path.exists():
                print(
                    "Skipping passage embedding cache: RL checkpoint requires corpus re-encoding "
                    f"(not {cache_path.name} from a prior zero-shot run).",
                    flush=True,
                )
            cache_path = None
        elif cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}", flush=True)
            p_emb = np.load(str(cache_path))

    if p_emb is None:
        print(f"encoding {len(chunk_texts)} chunks", flush=True)
        p_emb = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True)
        if cache_path:
            print(f"Saving embeddings to {cache_path}", flush=True)
            np.save(str(cache_path), p_emb)

    print("building index and ranking", flush=True)
    sim = np.dot(q_emb, np.array(p_emb).T)
    
    run = {}
    for i, qid in enumerate(q_ids):
        # MaxP aggregation
        doc_scores = {}
        for j, score in enumerate(sim[i]):
            doc_id = chunk_doc_ids[j]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score
        
        # Sort docs by score
        ranked_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        run[qid] = ranked_docs[:top_k]
        
    return run

def run_cross_encoder(
    queries_df,
    corpus_ids: list[str],
    corpus_texts: list[str],
    first_stage_run: dict[str, list[str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 100,
    batch_size: int = 32,
    max_length: int = 512,
    max_chunks_per_doc: int | None = None,
    device: str | None = None,
) -> dict[str, list[str]]:
    """Rerank a first-stage run (e.g., BM25) with chunking + MaxP per document.

    If max_chunks_per_doc is set, only the first N chunks of each doc are scored (in file
    order), which saves compute vs all chunks but can underestimate a doc if the best
    matching chunk lies beyond that prefix.
    """
    try:
        from sentence_transformers import CrossEncoder
        import numpy as np
        import torch
    except ImportError as e:
        raise RuntimeError("Cross-Encoder requires sentence-transformers and numpy.") from e

    if device is None or device == "auto":
        # Prefer MPS on macOS when device=auto.
        device = "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cross-encoder device=cuda but CUDA is not available.")
    elif device == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("cross-encoder device=mps but MPS is not available.")
    elif device != "cpu":
        raise ValueError(f"Unknown cross-encoder device: {device}")

    print(f"loading cross-encoder model: {model_name}", flush=True)
    model = CrossEncoder(model_name, device=device, max_length=max_length)
    
    doc_text_map = dict(zip(corpus_ids, corpus_texts))
    
    def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks
    
    run = {}
    print("reranking", flush=True)
    total_queries = len(queries_df)
    for i, (_, row) in enumerate(queries_df.iterrows()):
        if i % 10 == 0:
            print(f"Reranking query {i}/{total_queries}", flush=True)
        qid = str(row["qid"])
        q_text = row["text"]
        
        # Get top docs from first stage
        first_stage_docs = first_stage_run.get(qid, [])
        if not first_stage_docs:
            run[qid] = []
            continue
            
        # Prepare pairs
        pairs = []
        chunk_doc_ids = []
        chunk_counts: dict[str, int] = {}
        for doc_id in first_stage_docs:
            if doc_id in doc_text_map:
                chunks = chunk_text(doc_text_map[doc_id])
                if max_chunks_per_doc is not None and max_chunks_per_doc > 0:
                    chunks = chunks[:max_chunks_per_doc]
                for chunk in chunks:
                    pairs.append((q_text, chunk))
                    chunk_doc_ids.append(doc_id)
                    chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
                
        if not pairs:
            run[qid] = []
            continue

        if i < 1:
            max_chunks = max(chunk_counts.values()) if chunk_counts else 0
            avg_chunks = (sum(chunk_counts.values()) / max(1, len(chunk_counts))) if chunk_counts else 0.0
            print(
                f"[cross-encoder] qid={qid}: first_stage_docs={len(first_stage_docs)} "
                f"pairs={len(pairs)} max_chunks/doc={max_chunks} avg_chunks/doc={avg_chunks:.1f} "
                f"device={device}",
                flush=True,
            )
            
        # Score
        # show_progress_bar helps confirm the run is actually progressing (vs hanging silently).
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
        
        # MaxP aggregation
        doc_scores = {}
        for j, score in enumerate(scores):
            doc_id = chunk_doc_ids[j]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score
                
        # Sort docs by score
        ranked_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        run[qid] = ranked_docs[:top_k]
        
    return run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="legalbench_mini", choices=["legalbench_mini", "clerc"])
    ap.add_argument("--query-type", default="direct", choices=["direct", "indirect"])
    ap.add_argument("--qrels-level", default="doc", choices=["doc", "passage"])
    ap.add_argument("--bm25", action="store_true", help="Run BM25 baseline")
    ap.add_argument("--dpr", action="store_true", help="Run DPR baseline")
    ap.add_argument("--cross-encoder", action="store_true", help="Run Cross-Encoder over BM25")
    ap.add_argument("--bm25-index", type=str, default="", help="Path to Pyserini BM25 index")
    ap.add_argument(
        "--bm25-run",
        type=str,
        default="",
        help="JSON file from a prior BM25 run (qid -> doc ids). Use with --cross-encoder to skip BM25/Pyserini.",
    )
    ap.add_argument(
        "--cross-encoder-max-chunks-per-doc",
        type=int,
        default=0,
        help="Cap chunks scored per doc (in order from doc start). 0 = unlimited (full MaxP). "
        "Lower values reduce compute but may miss the best chunk if it appears later in long docs.",
    )
    ap.add_argument(
        "--cross-encoder-device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device for cross-encoder (cpu is gentler on laptops; auto prefers MPS on Apple Silicon).",
    )
    ap.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=8,
        help="Cross-encoder predict batch size (smaller = lower peak memory).",
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="If set, caps PyTorch CPU thread usage before cross-encoder (helps when device=cpu).",
    )
    ap.add_argument("--max-queries", type=int, default=None, help="Subset queries for quick run")
    ap.add_argument("--out-dir", type=str, default=None, help="Write run and metrics here")
    ap.add_argument("--rl-checkpoint", type=str, default=None, help="Path to RL fine-tuned model checkpoint to evaluate")
    ap.add_argument(
        "--dpr-device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device for DPR / RL-DPR encoding (auto prefers CUDA, then MPS, else CPU).",
    )
    ap.add_argument(
        "--dpr-batch-size",
        type=int,
        default=32,
        help="Encode batch size for DPR queries and passages (lower if GPU OOM).",
    )
    args = ap.parse_args()

    if not any([args.bm25, args.dpr, args.cross_encoder]):
        print("No baseline selected. Please specify --bm25, --dpr, or --cross-encoder.")
        return

    if args.bm25 and args.bm25_run:
        print("Use either --bm25 or --bm25-run, not both.")
        return

    if args.cross_encoder and not args.bm25 and not args.bm25_run:
        print("Cross-encoder needs a first-stage run: pass --bm25 (and --bm25-index) or --bm25-run PATH/to/run_bm25.json")
        return

    if args.dataset == "legalbench_mini":
        from setup.legalbench_data import load_queries_and_qrels, load_corpus, get_data_dir
        print("loading LegalBench-RAG-mini")
        queries_df, qrels = load_queries_and_qrels()
        corpus_ids, corpus_texts = load_corpus()
        out_dir = Path(args.out_dir or get_data_dir() / "runs")
    else:
        from setup.clerc_data import load_queries_and_qrels_hf, load_collection_hf_streaming, get_data_dir
        print("loading CLERC")
        queries_df, qrels = load_queries_and_qrels_hf(args.query_type, args.qrels_level)
        # Note: We load a small subset of CLERC by default for pipeline sanity
        # For a full run, see README
        corpus_ids, corpus_texts = load_collection_hf_streaming("passage", max_docs=5000)
        out_dir = Path(args.out_dir or get_data_dir() / "runs")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_queries:
        queries_df = queries_df.head(args.max_queries)
        qids = set(queries_df["qid"])
        qrels = {q: qrels[q] for q in qids if q in qrels}

    ks = [1, 5, 10, 100]
    all_metrics = {}
    bm25_run = None

    if args.bm25_run:
        run_path = Path(args.bm25_run)
        if not run_path.is_file():
            raise SystemExit(f"--bm25-run file not found: {run_path}")
        with open(run_path) as f:
            bm25_run = json.load(f)
        bm25_run = {str(k): v for k, v in bm25_run.items()}
        qids_needed = set(queries_df["qid"].astype(str).tolist())
        bm25_run = {q: bm25_run[q] for q in qids_needed if q in bm25_run}
        missing = qids_needed - set(bm25_run)
        if missing:
            print(f"warning: {len(missing)} qids missing from --bm25-run (cross-encoder will see empty first stage for them)", flush=True)

    if args.bm25:
        print("running BM25")
        bm25_run = run_bm25(
            queries_df,
            qrels,
            collection_path=args.bm25_index or None,
            top_k=max(ks),
        )
        metrics = evaluate(bm25_run, qrels, ks=ks)
        all_metrics["bm25"] = metrics
        with open(out_dir / "run_bm25.json", "w") as f:
            json.dump({k: v for k, v in bm25_run.items()}, f, indent=0)
        print("BM25:", metrics)

    if args.dpr:
        print("running DPR" + (" (RL fine-tuned)" if args.rl_checkpoint else ""))
        run = run_dpr(
            queries_df,
            corpus_ids,
            corpus_texts,
            top_k=max(ks),
            batch_size=max(1, args.dpr_batch_size),
            cache_dir=out_dir / "cache",
            rl_checkpoint=args.rl_checkpoint,
            dpr_device=args.dpr_device,
        )
        metrics = evaluate(run, qrels, ks=ks)
        metric_key = "rl_dpr" if args.rl_checkpoint else "dpr"
        all_metrics[metric_key] = metrics
        out_name = "run_rl_dpr.json" if args.rl_checkpoint else "run_dpr.json"
        with open(out_dir / out_name, "w") as f:
            json.dump({k: v for k, v in run.items()}, f, indent=0)
        print(f"{'RL DPR' if args.rl_checkpoint else 'DPR'}:", metrics)

    if args.cross_encoder and bm25_run:
        if args.torch_threads is not None:
            import torch

            torch.set_num_threads(max(1, args.torch_threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        dev = None if args.cross_encoder_device == "auto" else args.cross_encoder_device
        print("running Cross-Encoder over BM25")
        run = run_cross_encoder(
            queries_df,
            corpus_ids,
            corpus_texts,
            bm25_run,
            top_k=max(ks),
            batch_size=max(1, args.cross_encoder_batch_size),
            max_chunks_per_doc=(args.cross_encoder_max_chunks_per_doc or None),
            device=dev,
        )
        metrics = evaluate(run, qrels, ks=ks)
        all_metrics["cross_encoder"] = metrics
        with open(out_dir / "run_cross_encoder.json", "w") as f:
            json.dump({k: v for k, v in run.items()}, f, indent=0)
        print("Cross-Encoder:", metrics)

    metrics_path = out_dir / "baseline_metrics.json"
    merged: dict = {}
    if metrics_path.is_file():
        try:
            with open(metrics_path) as f:
                merged = json.load(f)
        except json.JSONDecodeError:
            merged = {}
    merged.update(all_metrics)
    with open(metrics_path, "w") as f:
        json.dump(merged, f, indent=2)
    print("wrote", metrics_path)
    return all_metrics

if __name__ == "__main__":
    main()
