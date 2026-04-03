"""
Build Pyserini BM25 index for LegalBench-RAG-mini.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import shutil

from setup.legalbench_data import load_corpus, get_data_dir

def main():
    data_dir = get_data_dir()
    index_dir = data_dir / "bm25_index"
    jsonl_dir = data_dir / "bm25_corpus_jsonl"
    
    # Create output directories
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    if index_dir.exists():
        shutil.rmtree(index_dir)
        
    print("Loading LegalBench-RAG corpus...")
    corpus_ids, corpus_texts = load_corpus()
    
    jsonl_path = jsonl_dir / "corpus.jsonl"
    print(f"Writing corpus to {jsonl_path}...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc_id, text in zip(corpus_ids, corpus_texts):
            doc = {
                "id": doc_id,
                "contents": text
            }
            f.write(json.dumps(doc) + "\n")
            
    print("Building Pyserini index...")
    # Requires Java 11+ and pyserini installed
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(jsonl_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully built BM25 index at {index_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build index: {e}")
        print("Please ensure you have Java 11+ installed.")
        
if __name__ == "__main__":
    main()
