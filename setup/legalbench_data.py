"""
LegalBench-RAG data loading for Phase 1.
Loads the dataset from the local `setup/data/legalbench_data` directory.
Provides doc-level queries and qrels for fast baseline iteration.
"""
import json
import os
import random
from pathlib import Path
import pandas as pd

def get_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "legalbench_data"

def ensure_data_downloaded():
    data_dir = get_data_dir()
    if not (data_dir / "benchmarks").exists() or not (data_dir / "corpus").exists():
        raise RuntimeError(
            f"LegalBench data not found at {data_dir}. "
            "Please download the LegalBench-RAG data (corpus and benchmarks folders) "
            "and place them in setup/data/legalbench_data."
        )

def load_queries_and_qrels(
    max_tests_per_benchmark: int = 194, 
    sort_by_document: bool = True
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """
    Load queries and qrels from LegalBench-RAG benchmarks.
    max_tests_per_benchmark=194 matches LegalBench-RAG-mini for fast iteration.
    Returns (queries_df with columns qid, text), qrels_dict (qid -> set of relevant doc ids).
    """
    ensure_data_downloaded()
    
    benchmark_names = ["privacy_qa", "contractnli", "maud", "cuad"]
    all_queries = []
    qrels = {}
    data_dir = get_data_dir()
    
    qid_counter = 1
    
    for b_name in benchmark_names:
        with open(data_dir / "benchmarks" / f"{b_name}.json") as f:
            data = json.load(f)
            tests = data.get("tests", [])
            
            if len(tests) > max_tests_per_benchmark:
                if sort_by_document:
                    # Sort to group by document, minimizing unique docs needed
                    tests = sorted(
                        tests,
                        key=lambda t: (
                            random.seed(t["snippets"][0]["file_path"] if t["snippets"] else ""),
                            random.random()
                        )[1]
                    )
                else:
                    random.seed(b_name)
                    random.shuffle(tests)
                tests = tests[:max_tests_per_benchmark]
                
            for test in tests:
                qid = f"{b_name}_{qid_counter}"
                qid_counter += 1
                query_text = test["query"]
                
                # Extract relevant document file paths
                rel_docs = set()
                for snippet in test["snippets"]:
                    rel_docs.add(snippet["file_path"])
                
                if not rel_docs:
                    continue
                    
                all_queries.append({"qid": qid, "text": query_text})
                qrels[qid] = rel_docs
                
    queries_df = pd.DataFrame(all_queries)
    return queries_df, qrels

def load_corpus(qrels: dict[str, set[str]] = None) -> tuple[list[str], list[str]]:
    """
    Load documents from the LegalBench-RAG corpus.
    If qrels is provided, only loads documents present in qrels to save memory/time.
    Otherwise loads the entire corpus.
    Returns (doc_ids, texts).
    """
    ensure_data_downloaded()
    data_dir = get_data_dir()
    
    doc_ids = []
    texts = []
    
    target_docs = None
    if qrels is not None:
        target_docs = set()
        for docs in qrels.values():
            target_docs.update(docs)
            
    corpus_dir = data_dir / "corpus"
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
                
            full_path = Path(root) / file
            # Relative path matching what's in snippets (e.g., "privacy_qa/Fiverr.txt")
            rel_path = str(full_path.relative_to(corpus_dir))
            
            # If we specified target_docs, skip docs not in the targets
            if target_docs is not None and rel_path not in target_docs:
                continue
                
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                
            doc_ids.append(rel_path)
            texts.append(content)
            
    return doc_ids, texts
