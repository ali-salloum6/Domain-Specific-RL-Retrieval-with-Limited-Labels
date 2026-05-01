#!/usr/bin/env python3
"""
Build Kaggle-uploadable zips with LegalBench paths sanitized (Kaggle rejects | : & # etc.).

  python scripts/build_kaggle_rl_bundle.py           -> kaggle_rl_minimal.zip (train)
  python scripts/build_kaggle_rl_bundle.py --eval   -> kaggle_eval_rl.zip (evaluate RL-DPR)

Training and eval zips use the same deterministic rename map from your local
setup/data/legalbench_data — keep using the same script version so paths match.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

# Characters Kaggle (and many Windows tools) reject in archive paths.
_SEGMENT_FORBIDDEN = re.compile(r'[<>:"/\\|?*#&:]')
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize_segment(seg: str) -> str:
    s = seg.strip()
    s = _SEGMENT_FORBIDDEN.sub("_", s)
    s = _CONTROL.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s = re.sub(r"\s+", " ", s).strip()
    low = s.lower()
    if low.endswith(".txt") and len(s) > 4:
        base = s[:-4].rstrip()
        s = base + s[-4:]
    return s or "_"


def _sanitize_rel(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    return "/".join(_sanitize_segment(p) for p in parts if p != "")


def _assign_unique_paths(rel_paths: list[str]) -> dict[str, str]:
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for old in sorted(rel_paths):
        base = _sanitize_rel(old)
        candidate = base
        if candidate not in used:
            used.add(candidate)
            mapping[old] = candidate
            continue
        p = Path(base)
        parent = p.parent
        stem = p.stem
        suffix = p.suffix
        n = 2
        while True:
            mid = f"{stem}__dup{n}{suffix}"
            candidate = str(parent / mid) if str(parent) != "." else mid
            if candidate not in used:
                used.add(candidate)
                mapping[old] = candidate
                break
            n += 1
    return mapping


def _rewrite_file_paths(obj: object, mapping: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "file_path" and isinstance(v, str):
                if v in mapping:
                    obj[k] = mapping[v]
            else:
                _rewrite_file_paths(v, mapping)
    elif isinstance(obj, list):
        for item in obj:
            _rewrite_file_paths(item, mapping)


def _rewrite_bm25(data: dict, mapping: dict[str, str]) -> None:
    for qid, docs in list(data.items()):
        if not isinstance(docs, list):
            continue
        data[qid] = [mapping.get(d, d) for d in docs]


def _corpus_path_map(root: Path) -> dict[str, str]:
    src_corpus = root / "setup" / "data" / "legalbench_data" / "corpus"
    rel_paths: list[str] = []
    for full in src_corpus.rglob("*"):
        if full.is_file() and full.suffix.lower() == ".txt":
            rel_paths.append(str(full.relative_to(src_corpus)).replace("\\", "/"))
    return _assign_unique_paths(rel_paths)


def _write_sanitized_legalbench(root: Path, bundle: Path, path_map: dict[str, str]) -> None:
    src_data = root / "setup" / "data" / "legalbench_data"
    src_corpus = src_data / "corpus"
    src_benchmarks = src_data / "benchmarks"
    src_bm25 = root / "setup" / "data" / "runs" / "run_bm25.json"

    dst_data = bundle / "setup" / "data" / "legalbench_data"
    dst_corpus = dst_data / "corpus"
    dst_benchmarks = dst_data / "benchmarks"
    dst_benchmarks.mkdir(parents=True)
    (bundle / "setup" / "data" / "runs").mkdir(parents=True)

    for old_rel, new_rel in path_map.items():
        src_f = src_corpus / old_rel
        dst_f = dst_corpus / new_rel
        dst_f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_f, dst_f)

    for jf in sorted(src_benchmarks.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        _rewrite_file_paths(data, path_map)
        with open(dst_benchmarks / jf.name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

    with open(src_bm25, encoding="utf-8") as f:
        bm25 = json.load(f)
    bm25 = {str(k): v for k, v in bm25.items()}
    _rewrite_bm25(bm25, path_map)
    with open(bundle / "setup" / "data" / "runs" / "run_bm25.json", "w", encoding="utf-8") as f:
        json.dump(bm25, f, ensure_ascii=False)
        f.write("\n")


def _zip_dir(bundle: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(bundle).as_posix())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval",
        action="store_true",
        help="Build kaggle_eval_rl.zip (run_baselines + policy + data) instead of training zip.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src_corpus = root / "setup" / "data" / "legalbench_data" / "corpus"
    src_bm25 = root / "setup" / "data" / "runs" / "run_bm25.json"
    bundle = root / "kaggle_bundle"
    out_zip = root / ("kaggle_eval_rl.zip" if args.eval else "kaggle_rl_minimal.zip")

    if not src_corpus.is_dir():
        print(f"Missing corpus: {src_corpus}", file=sys.stderr)
        return 1
    if not src_bm25.is_file():
        print(f"Missing BM25 run: {src_bm25}", file=sys.stderr)
        return 1

    path_map = _corpus_path_map(root)
    n_changed = sum(1 for o, n in path_map.items() if o != n)
    print(f"Corpus .txt files: {len(path_map)} ({n_changed} paths renamed for Kaggle)")

    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    (bundle / "rl").mkdir()
    (bundle / "setup").mkdir()

    if args.eval:
        shutil.copy2(root / "rl" / "policy.py", bundle / "rl" / "policy.py")
        shutil.copy2(root / "setup" / "run_baselines.py", bundle / "setup" / "run_baselines.py")
        for name in ("__init__.py", "legalbench_data.py", "metrics.py", "requirements.txt"):
            shutil.copy2(root / "setup" / name, bundle / "setup" / name)
    else:
        for name in ("train.py", "policy.py", "data.py"):
            shutil.copy2(root / "rl" / name, bundle / "rl" / name)
        for name in ("__init__.py", "legalbench_data.py", "metrics.py", "requirements.txt"):
            shutil.copy2(root / "setup" / name, bundle / "setup" / name)

    _write_sanitized_legalbench(root, bundle, path_map)
    _zip_dir(bundle, out_zip)

    print(f"Wrote {out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(bundle)
    print("Removed kaggle_bundle/ staging dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
