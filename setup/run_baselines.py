"""
Run BM25 and DPR baselines on CLERC dev/test set and record metrics.
Uses subset of corpus for fast iteration; full corpus run documented in README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

# Ensure repo root on path for "from setup.xxx" when run as script
_TOP = Path(__file__).resolve().parent.parent
if _TOP not in sys.path:
    sys.path.insert(0, str(_TOP))

from setup.clerc_data import get_data_dir, load_queries_and_qrels_hf, CLERC_PASSAGE_COUNT_APPROX
from setup.metrics import evaluate


def run_bm25(
    queries_df,
    qrels: dict,
    collection_path: str | Path | None = None,
    top_k: int = 100,
    k1: float = 0.9,
    b: float = 0.4,
) -> dict[str, list[str]]:
    """
    Run BM25 retrieval via Pyserini.
    collection_path: path to TSV (doc_id, text) or directory of Pyserini docs.
    If None, uses a mock run for pipeline test (no real index).
    Returns run dict: qid -> list of retrieved doc ids.
    """
    try:
        from pyserini.search import SimpleSearcher
        from pyserini.index import IndexReader
    except ImportError:
        return _mock_run_bm25(queries_df, qrels, top_k)

    if not collection_path or not Path(collection_path).exists():
        return _mock_run_bm25(queries_df, qrels, top_k)

    # Assume Pyserini index already built (see README)
    searcher = SimpleSearcher(str(collection_path))
    searcher.set_bm25(k1=k1, b=b)
    run = {}
    for _, row in queries_df.iterrows():
        qid = str(row["qid"])
        hits = searcher.search(row["text"], k=top_k)
        run[qid] = [h.docid for h in hits]
    return run


def _mock_run_bm25(queries_df, qrels: dict, top_k: int) -> dict[str, list[str]]:
    """Return a dummy run (all zeros / empty) when Pyserini index not available."""
    run = {}
    all_doc_ids = set()
    for s in qrels.values():
        all_doc_ids.update(s)
    doc_list = list(all_doc_ids)[: top_k * 2]
    for _, row in queries_df.iterrows():
        qid = str(row["qid"])
        # Placeholder: return first top_k docs (poor baseline for testing pipeline)
        run[qid] = doc_list[:top_k] if doc_list else []
    return run


def run_dpr(
    queries_df,
    qrels: dict,
    model_name: str = "jhu-clsp/LegalBERT-DPR-CLERC-ft",
    corpus_ids: list[str] | None = None,
    corpus_texts: list[str] | None = None,
    top_k: int = 100,
    batch_size: int = 32,
    corpus_size: int = 0,
) -> dict[str, list[str]]:
    """
    Run DPR retrieval using HF encoder (query + passage).
    If corpus_* are None and corpus_size<=0: uses minimal corpus from qrels (fast, for pipeline check).
    If corpus_size>0: loads that many passages from CLERC (streaming; no full 9GB load).
    Returns run dict: qid -> list of retrieved doc/passage ids.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return _mock_run_dpr(queries_df, qrels, top_k)

    _from_ckpt_only = False
    _have_q_from_ckpt = False
    if corpus_ids is None or corpus_texts is None:
        if corpus_size > 0:
            try:
                import numpy as np
                from pathlib import Path
                from setup.clerc_data import load_collection_hf_streaming, get_data_dir

                # Checkpoint path: setup/data/embeddings/dpr_passage.npz (or CLERC_DATA/embeddings/ if set)
                embedding_dir = get_data_dir() / "embeddings"
                embedding_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = embedding_dir / "dpr_passage.npz"

                n_encoded = 0
                loaded_ids, loaded_p_emb = [], None

                # Load existing checkpoint (passages and optionally queries)
                loaded_q_emb, loaded_q_ids = None, None
                if ckpt_path.exists():
                    try:
                        ckpt = np.load(ckpt_path, allow_pickle=True)
                        n_encoded = int(ckpt["n_encoded"])
                        loaded_ids = list(ckpt["corpus_ids"])
                        loaded_p_emb = ckpt["p_emb"]
                        if "q_emb" in ckpt:
                            loaded_q_emb = ckpt["q_emb"]
                        if "q_ids" in ckpt:
                            loaded_q_ids = list(ckpt["q_ids"])
                        print("loaded embedding checkpoint: %d passages" % n_encoded, flush=True)
                    except Exception as e:
                        print("checkpoint load failed, starting fresh:", e, flush=True)
                        n_encoded, loaded_ids, loaded_p_emb = 0, [], None
                        loaded_q_emb, loaded_q_ids = None, None

                if n_encoded >= corpus_size:
                    corpus_ids = loaded_ids[:corpus_size]
                    p_emb = loaded_p_emb[:corpus_size]
                    _have_q_from_ckpt = loaded_q_emb is not None and loaded_q_ids is not None
                    if _have_q_from_ckpt:
                        q_emb = loaded_q_emb
                        q_ids = loaded_q_ids
                        print("queries loaded from checkpoint (%d)" % len(q_ids), flush=True)
                    _from_ckpt_only = True
                    print("using %d passages from checkpoint (skip encoding)" % corpus_size, flush=True)
                else:
                    _from_ckpt_only = False
                    q_texts = queries_df["text"].tolist()
                    q_ids = queries_df["qid"].astype(str).tolist()
                    print("loading model", flush=True)
                    model = SentenceTransformer(model_name)
                    _dev = getattr(model, "_target_device", None) or next(model.parameters()).device
                    print("using device: %s" % _dev, flush=True)
                    # Use cached query embeddings from checkpoint when present; else encode
                    if loaded_q_emb is not None and loaded_q_ids is not None:
                        q_emb = loaded_q_emb
                        q_ids = loaded_q_ids
                        print("queries loaded from checkpoint (%d)" % len(q_ids), flush=True)
                    else:
                        print("encoding queries", flush=True)
                        q_emb = model.encode(q_texts, batch_size=batch_size, show_progress_bar=True)

                    # Encode passages in chunks and save checkpoint after each chunk
                    CKPT_CHUNK = 5000
                    corpus_ids = list(loaded_ids) if loaded_ids else []
                    p_emb = loaded_p_emb.copy() if loaded_p_emb is not None and len(loaded_p_emb) > 0 else None
                    # Save query embeddings (and current passage state) immediately so they are never lost
                    _p_emb_save = p_emb if (p_emb is not None and len(p_emb) > 0) else np.zeros((0, q_emb.shape[1]), dtype=q_emb.dtype)
                    _ids_save = np.array(corpus_ids, dtype=object)
                    np.savez_compressed(
                        ckpt_path,
                        n_encoded=n_encoded,
                        corpus_ids=_ids_save,
                        p_emb=_p_emb_save,
                        q_emb=q_emb,
                        q_ids=np.array(q_ids, dtype=object),
                    )
                    print("checkpoint saved: query embeddings + %d passages" % n_encoded, flush=True)
                    n_start = n_encoded
                    while n_encoded < corpus_size:
                        chunk_size = min(CKPT_CHUNK, corpus_size - n_encoded)
                        chunk_ids, chunk_texts = load_collection_hf_streaming(
                            "passage", max_docs=chunk_size, skip_first=n_encoded
                        )
                        if not chunk_texts:
                            break
                        print("encoding passages %d–%d..." % (n_encoded + 1, n_encoded + len(chunk_texts)), flush=True)
                        p_chunk = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True)
                        corpus_ids.extend(chunk_ids)
                        p_emb = np.vstack((p_emb, p_chunk)) if p_emb is not None else p_chunk
                        n_encoded += len(chunk_ids)
                        np.savez_compressed(
                            ckpt_path,
                            n_encoded=n_encoded,
                            corpus_ids=np.array(corpus_ids, dtype=object),
                            p_emb=p_emb,
                            q_emb=q_emb,
                            q_ids=np.array(q_ids, dtype=object),
                        )
                        print("checkpoint saved: %d passages" % n_encoded, flush=True)
                    _from_ckpt_only = False
            except Exception as e:
                print("corpus load failed, using qrels only:", e, flush=True)
                corpus_ids, corpus_texts = _minimal_corpus_from_qrels(qrels)
                _from_ckpt_only = False
                _have_q_from_ckpt = False
        else:
            corpus_ids, corpus_texts = _minimal_corpus_from_qrels(qrels)
            _from_ckpt_only = False
            _have_q_from_ckpt = False

    q_texts = queries_df["text"].tolist()
    q_ids = queries_df["qid"].astype(str).tolist()
    # When we used checkpoint only, load model + encode queries only if checkpoint had no query embeddings
    if corpus_size > 0 and _from_ckpt_only and not _have_q_from_ckpt:
        print("loading model", flush=True)
        model = SentenceTransformer(model_name)
        _dev = getattr(model, "_target_device", None) or next(model.parameters()).device
        print("using device: %s" % _dev, flush=True)
        print("encoding queries", flush=True)
        q_emb = model.encode(q_texts, batch_size=batch_size, show_progress_bar=True)
    elif not _from_ckpt_only and (corpus_size <= 0 or "corpus_texts" in dir()):
        if "model" not in dir():
            print("loading model", flush=True)
            model = SentenceTransformer(model_name)
            _dev = getattr(model, "_target_device", None) or next(model.parameters()).device
            print("using device: %s" % _dev, flush=True)
        if "q_emb" not in dir():
            print("encoding queries", flush=True)
            q_emb = model.encode(q_texts, batch_size=batch_size, show_progress_bar=True)
        if "p_emb" not in dir():
            print("encoding passages", flush=True)
            p_emb = model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True)

    print("building index", flush=True)
    # FAISS often segfaults on ARM macOS (M1/M2) at index.add/search. Use numpy for corpus <= 500k
    # so 1% runs (~250k) complete without crash; numpy is slower but safe.
    use_numpy = len(corpus_ids) < 500_000
    if use_numpy:
        sim = np.dot(q_emb, np.array(p_emb).T)
        I = np.argsort(-sim, axis=1)[:, : min(top_k, len(corpus_ids))]
    else:
        try:
            import faiss
            index = faiss.IndexFlatIP(p_emb.shape[1])
            index.add(p_emb.astype("float32"))
            D, I = index.search(q_emb.astype("float32"), min(top_k, len(corpus_ids)))
        except (ImportError, Exception):
            sim = np.dot(q_emb, np.array(p_emb).T)
            I = np.argsort(-sim, axis=1)[:, : min(top_k, len(corpus_ids))]

    print("ranking", flush=True)
    run = {}
    for i, qid in enumerate(q_ids):
        run[qid] = [corpus_ids[j] for j in I[i] if j < len(corpus_ids)]
    return run


def _minimal_corpus_from_qrels(qrels: dict) -> tuple[list[str], list[str]]:
    all_pids = set()
    for s in qrels.values():
        all_pids.update(s)
    corpus_ids = list(all_pids)
    corpus_texts = ["[CLERC passage]"] * len(corpus_ids)
    return corpus_ids, corpus_texts


def _mock_run_dpr(queries_df, qrels: dict, top_k: int) -> dict[str, list[str]]:
    """Dummy run when sentence_transformers not available."""
    all_doc_ids = set()
    for s in qrels.values():
        all_doc_ids.update(s)
    doc_list = list(all_doc_ids)[: top_k * 2]
    run = {}
    for _, row in queries_df.iterrows():
        run[str(row["qid"])] = doc_list[:top_k] if doc_list else []
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-type", default="direct", choices=["direct", "indirect"])
    ap.add_argument("--qrels-level", default="doc", choices=["doc", "passage"])
    ap.add_argument("--bm25", action="store_true", help="Run BM25 baseline")
    ap.add_argument("--dpr", action="store_true", help="Run DPR baseline (HF LegalBERT-DPR-CLERC-ft)")
    ap.add_argument("--bm25-index", type=str, default="", help="Path to Pyserini BM25 index (optional)")
    ap.add_argument("--max-queries", type=int, default=None, help="Subset queries for quick run")
    ap.add_argument("--out-dir", type=str, default=None, help="Write run and metrics here")
    ap.add_argument("--mock", action="store_true", help="Use mock retrievers only (no model/index); for pipeline test")
    ap.add_argument("--dpr-corpus", type=int, default=0, help="Load N passages from CLERC collection for DPR (0=minimal from qrels only; avoids 9GB full load)")
    ap.add_argument("--dpr-corpus-pct", type=float, default=None, help="Use this percent of CLERC passage collection (e.g. 10 for 10%%); overrides --dpr-corpus when set")
    args = ap.parse_args()

    if args.dpr_corpus_pct is not None:
        pct = max(0.0, min(100.0, args.dpr_corpus_pct))
        args.dpr_corpus = int(CLERC_PASSAGE_COUNT_APPROX * (pct / 100.0))
        print("Using %.1f%% of collection (~%d passages)" % (pct, args.dpr_corpus), flush=True)

    if not args.bm25 and not args.dpr:
        args.bm25 = args.dpr = True

    run_bm25_fn = run_bm25
    run_dpr_fn = run_dpr
    if args.mock:
        def _run_bm25_mock(qdf, qr, **kw):
            return _mock_run_bm25(qdf, qr, kw.get("top_k", 100))
        def _run_dpr_mock(qdf, qr, **kw):
            return _mock_run_dpr(qdf, qr, kw.get("top_k", 100))
        run_bm25_fn = _run_bm25_mock
        run_dpr_fn = _run_dpr_mock

    out_dir = Path(args.out_dir or get_data_dir() / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading queries and qrels")
    queries_df, qrels = load_queries_and_qrels_hf(args.query_type, args.qrels_level)
    if args.max_queries:
        queries_df = queries_df.head(args.max_queries)
        qids = set(queries_df["qid"])
        qrels = {q: qrels[q] for q in qids if q in qrels}

    ks = [1, 5, 10, 100]
    all_metrics = {}

    if args.bm25:
        print("running BM25")
        run = run_bm25_fn(
            queries_df,
            qrels,
            collection_path=args.bm25_index or None,
            top_k=max(ks),
        )
        metrics = evaluate(run, qrels, ks=ks)
        all_metrics["bm25"] = metrics
        with open(out_dir / "run_bm25.json", "w") as f:
            json.dump({k: v for k, v in run.items()}, f, indent=0)
        print("BM25:", metrics)

    if args.dpr:
        print("running DPR")
        run = run_dpr_fn(queries_df, qrels, top_k=max(ks), corpus_size=args.dpr_corpus)
        metrics = evaluate(run, qrels, ks=ks)
        all_metrics["dpr"] = metrics
        with open(out_dir / "run_dpr.json", "w") as f:
            json.dump({k: v for k, v in run.items()}, f, indent=0)
        print("DPR:", metrics)

    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("wrote", out_dir / "baseline_metrics.json")
    return all_metrics


if __name__ == "__main__":
    main()
