"""
CLERC data loading for Phase 1.
Load from HuggingFace jhu-clsp/CLERC; document splits and qrels for evaluation.

HF repo layout (as of 2024):
  collection/collection.doc.tsv.gz, collection.passage.tsv.gz
  collection/mapping.did2pid.tsv, mapping.pid2did.tsv
  generation/train.jsonl, generation/test.jsonl, generation/all.jsonl
  qrels/qrels-doc.test.direct.tsv, qrels-doc.test.indirect.tsv
  qrels/qrels-passage.test.direct.tsv, qrels-passage.test.indirect.tsv
  queries/test.single-removed.direct.tsv, test.single-removed.indirect.tsv, etc.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any

def get_data_dir() -> Path:
    """Default directory for cached CLERC data (queries, qrels, corpus)."""
    d = Path(os.environ.get("CLERC_DATA", Path(__file__).resolve().parent / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_hf_datasets_cache_dir() -> str:
    """
    Directory for HuggingFace `datasets` cache when loading CLERC shards.

    Default is under the user home (short path). A deep repo path like Desktop/.../setup/data/hf_cache
    can make `datasets` lock filenames exceed Windows MAX_PATH. Override with env CLERC_HF_CACHE.
    """
    if os.environ.get("CLERC_HF_CACHE"):
        p = Path(os.environ["CLERC_HF_CACHE"])
    else:
        p = Path.home() / ".cache" / "clerc_hf"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# Approximate total passages in collection.passage.tsv.gz (~9 GB); used for --dpr-corpus-pct.
# Source: dataset scale described as millions of instances (HF/paper).
CLERC_PASSAGE_COUNT_APPROX = 25_000_000

# Official CLERC HuggingFace file paths (for splits and qrels)
CLERC_HF_FILES = {
    "generation_train": "generation/train.jsonl",
    "generation_test": "generation/test.jsonl",
    "queries_direct": "queries/test.single-removed.direct.tsv",
    "queries_indirect": "queries/test.single-removed.indirect.tsv",
    "qrels_doc_direct": "qrels/qrels-doc.test.direct.tsv",
    "qrels_doc_indirect": "qrels/qrels-doc.test.indirect.tsv",
    "qrels_passage_direct": "qrels/qrels-passage.test.direct.tsv",
    "qrels_passage_indirect": "qrels/qrels-passage.test.indirect.tsv",
    "collection_doc": "collection/collection.doc.tsv.gz",
    "collection_passage": "collection/collection.passage.tsv.gz",
    "mapping_pid2did": "collection/mapping.pid2did.tsv",
    "mapping_did2pid": "collection/mapping.did2pid.tsv",
}


def load_clerc_from_hf(
    split: str = "test",
    task: str = "generation",
    cache_dir: str | None = None,
) -> Any:
    """
    Load CLERC from HuggingFace.
    task: 'generation' -> generation/train.jsonl or generation/test.jsonl
    """
    from datasets import load_dataset

    key = f"generation_{split}"
    if key not in CLERC_HF_FILES:
        key = "generation_test"
    data_files = {"data": CLERC_HF_FILES[key]}

    ds = load_dataset(
        "jhu-clsp/CLERC",
        data_files=data_files,
        cache_dir=cache_dir,
    )
    return ds["data"]


def list_clerc_hf_files() -> list[str]:
    """List available files in the CLERC HuggingFace repo."""
    try:
        from huggingface_hub import list_repo_files
        return list_repo_files("jhu-clsp/CLERC", repo_type="dataset")
    except Exception:
        return []


def load_queries_and_qrels_hf(
    query_type: str = "direct",
    qrels_level: str = "doc",
    cache_dir: str | None = None,
) -> tuple[Any, dict[str, set[str]]]:
    """
    Load retrieval queries and qrels from HuggingFace.
    query_type: 'direct' or 'indirect'
    qrels_level: 'doc' or 'passage'
    Returns (queries_df with columns qid, text), qrels_dict (qid -> set of relevant ids).
    """
    from datasets import load_dataset
    import pandas as pd

    cache_dir = cache_dir or get_hf_datasets_cache_dir()

    queries_file = f"queries/test.single-removed.{query_type}.tsv"
    qrels_file = f"qrels/qrels-{qrels_level}.test.{query_type}.tsv"

    print("loading queries", flush=True)
    ds_q = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"data": queries_file},
        cache_dir=cache_dir,
    )
    print("loading qrels", flush=True)
    ds_r = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"data": qrels_file},
        cache_dir=cache_dir,
    )

    # Queries TSV: qid, text (no header) -> columns may be 0, 1 or col1, col2
    queries = ds_q["data"]
    if hasattr(queries, "to_pandas"):
        qdf = queries.to_pandas()
    else:
        qdf = pd.DataFrame(queries)
    if qdf.shape[1] >= 2:
        qdf.columns = ["qid", "text"][: qdf.shape[1]]
    queries_df = qdf[["qid", "text"]].copy()
    queries_df["qid"] = queries_df["qid"].astype(str)
    queries_df["text"] = queries_df["text"].astype(str)

    # Qrels TSV: qid, 0, doc_id_or_pid, 1 (TREC format)
    qrels_data = ds_r["data"]
    if hasattr(qrels_data, "to_pandas"):
        rdf = qrels_data.to_pandas()
    else:
        rdf = pd.DataFrame(qrels_data)
    if rdf.shape[1] < 3:
        raise ValueError("Qrels must have at least qid, _, doc_id/pid")
    rdf.columns = ["qid", "Q0", "doc_id", "rel"][: rdf.shape[1]]
    rdf["qid"] = rdf["qid"].astype(str)
    rdf["doc_id"] = rdf["doc_id"].astype(str)
    qrels_dict = rdf.groupby("qid")["doc_id"].apply(set).to_dict()

    return queries_df, qrels_dict


def check_clerc_collection_cached(collection_type: str = "passage") -> tuple[bool, str | None, int | None]:
    """
    Check if the CLERC collection file is already in the HuggingFace cache.
    Returns (is_cached, local_path_or_none, size_bytes_or_none).
    """
    from huggingface_hub import try_to_load_from_cache, scan_cache_dir

    path_in_repo = CLERC_HF_FILES[f"collection_{collection_type}"]
    local_path = try_to_load_from_cache(
        repo_id="jhu-clsp/CLERC",
        filename=path_in_repo,
        repo_type="dataset",
    )
    if local_path is not None:
        # Can be str or (path, blob_path) in some versions
        p = local_path[0] if isinstance(local_path, tuple) else local_path
        p = Path(p)
        if p.exists():
            return True, str(p), p.stat().st_size
    # Scan cache to see what we have for this repo (and report size if partial)
    try:
        cache = scan_cache_dir()
        for repo in getattr(cache, "repos", []) or []:
            if getattr(repo, "repo_id", None) != "jhu-clsp/CLERC" or getattr(repo, "repo_type", None) != "dataset":
                continue
            for rev in getattr(repo, "revisions", []) or []:
                for f in getattr(rev, "files", []) or []:
                    if path_in_repo in getattr(f, "file_path", "") or (path_in_repo.split("/")[-1] == getattr(f, "file_name", "")):
                        p = getattr(f, "file_path", None) or getattr(f, "blob_path", None)
                        if p and Path(p).exists():
                            return True, str(p), getattr(f, "size_on_disk", None) or Path(p).stat().st_size
    except Exception:
        pass
    return False, None, None


def load_collection_hf_streaming(
    collection_type: str = "passage",
    max_docs: int = 10_000,
    skip_first: int = 0,
) -> tuple[list[str], list[str]]:
    """
    Load passages from CLERC collection by streaming the .tsv.gz file.
    skip_first: number of lines to skip from the start (for resuming).
    Then read up to max_docs lines. Avoids loading the full 9GB into memory.
    """
    import gzip
    from huggingface_hub import hf_hub_download

    key = f"collection_{collection_type}"
    path_in_repo = CLERC_HF_FILES[key]
    # Expected size on HF (for progress logging when resuming)
    expected_size_mb = 9216  # ~9 GB for collection.passage.tsv.gz

    is_cached, cached_path, size = check_clerc_collection_cached(collection_type)
    if is_cached and cached_path:
        size_mb = (size or 0) / (1024 * 1024)
        print("[CLERC] Using cached collection (%.1f GB)" % (size_mb / 1024) if size_mb > 1024 else "[CLERC] Using cached collection (%.0f MB)" % size_mb, flush=True)
        local_path = cached_path
    else:
        import sys
        import threading
        import time

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_dir / "datasets--jhu-clsp--CLERC"
        blobs_dir = repo_dir / "blobs"
        # Detect partial download (.incomplete) so we can log "resuming"
        existing_mb = 0.0
        if blobs_dir.exists():
            for f in blobs_dir.iterdir():
                if f.is_file():
                    existing_mb += f.stat().st_size
            existing_mb /= 1024 * 1024
        if existing_mb > 100:
            print("[CLERC] Cache check: no complete file; found partial download (%.0f MB). Resuming (target ~%d MB)..." % (existing_mb, expected_size_mb), flush=True)
            print("[CLERC] Hugging Face supports range requests; hf_hub will resume from existing bytes. If it stays stuck, try: HF_HUB_ENABLE_HF_TRANSFER=1 or check network/VPN.", flush=True)
        else:
            print("[CLERC] Cache check: not cached. Downloading (~%d MB)..." % expected_size_mb, flush=True)

        stop_poll = threading.Event()
        last_mb = [existing_mb]

        def poll_size():
            while not stop_poll.is_set():
                try:
                    if blobs_dir.exists():
                        total = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())
                        mb = total / (1024 * 1024)
                        if mb != last_mb[0]:
                            last_mb[0] = mb
                            print("\r[CLERC] Download progress: %.0f / ~%d MB" % (mb, expected_size_mb), end="", flush=True)
                except Exception:
                    pass
                stop_poll.wait(3)
            print(flush=True)

        # Use faster download backend when available (resume-friendly, often unblocks stuck downloads)
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("[CLERC] Using hf_transfer for download.", flush=True)
        except ImportError:
            pass

        poller = threading.Thread(target=poll_size, daemon=True)
        poller.start()
        try:
            last_error = None
            for attempt in range(2):
                try:
                    import logging
                    if os.environ.get("CLERC_DOWNLOAD_DEBUG"):
                        logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
                    from tqdm.auto import tqdm
                    local_path = hf_hub_download(
                        repo_id="jhu-clsp/CLERC",
                        filename=path_in_repo,
                        repo_type="dataset",
                        tqdm_class=tqdm,
                    )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt == 0:
                        print("[CLERC] Download attempt failed, retrying once: %s" % e, flush=True)
                    else:
                        raise
            if last_error is not None:
                raise last_error
        finally:
            stop_poll.set()
            poller.join(timeout=1)
        print("[CLERC] Download complete.", flush=True)
    if skip_first > 0:
        print("[CLERC] Reading passages (skip %d, then %d)..." % (skip_first, max_docs), flush=True)
    else:
        print("[CLERC] Reading passages (first %d)..." % max_docs, flush=True)
    ids, texts = [], []
    with gzip.open(local_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < skip_first:
                continue
            if len(ids) >= max_docs:
                break
            parts = line.strip().split("\t", 1)
            if len(parts) >= 2:
                ids.append(parts[0])
                texts.append(parts[1])
            elif len(parts) == 1:
                ids.append(parts[0])
                texts.append("")
    return ids, texts


def load_collection_hf(
    collection_type: str = "passage",
    cache_dir: str | None = None,
    max_docs: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    Load CLERC collection (doc or passage) from HuggingFace.
    collection_type: 'doc' or 'passage'
    Returns (ids, texts). For large collections prefer load_collection_hf_streaming.
    """
    from datasets import load_dataset

    cache_dir = cache_dir or get_hf_datasets_cache_dir()
    key = f"collection_{collection_type}"
    path = CLERC_HF_FILES[key]

    ds = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"col": path},
        cache_dir=cache_dir,
    )
    col = ds["col"]
    if hasattr(col, "to_pandas"):
        df = col.to_pandas()
    else:
        import pandas as pd
        df = pd.DataFrame(col)
    if df.shape[1] < 2:
        raise ValueError("Collection must have id and text columns")
    df.columns = ["pid", "text"][: df.shape[1]]
    if max_docs:
        df = df.head(max_docs)
    return df["pid"].astype(str).tolist(), df["text"].astype(str).tolist()


# Documented splits for reproducibility
CLERC_SPLITS = {
    "train": "generation/train.jsonl",
    "test": "generation/test.jsonl",
    "queries_direct": "queries/test.single-removed.direct.tsv",
    "queries_indirect": "queries/test.single-removed.indirect.tsv",
    "qrels_doc_direct": "qrels/qrels-doc.test.direct.tsv",
    "qrels_doc_indirect": "qrels/qrels-doc.test.indirect.tsv",
    "qrels_passage_direct": "qrels/qrels-passage.test.direct.tsv",
    "qrels_passage_indirect": "qrels/qrels-passage.test.indirect.tsv",
}
