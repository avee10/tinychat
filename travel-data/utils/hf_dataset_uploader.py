#!/usr/bin/env python3
"""
hf_uploader.py — Upload ANY dataset to a private Hugging Face repo.

Modes
- Structured (default): Convert CSV/JSONL/Parquet into a Hugging Face `DatasetDict`
  and push via `datasets.push_to_hub`.
- Raw (--raw): Upload a folder tree (CSV, JSONL, GTFS zips, README.md, etc.) using
  huggingface_hub.upload_folder, keeping your original layout.

Features
- Auto-detect split files by naming convention: train*, valid*, dev*, test*, *.parquet/*.csv/*.jsonl
- Optional schema projection (select columns) and dtype hints
- Chunked read to avoid high RAM usage (for CSV/JSONL)
- Creates a minimal Dataset Card (README.md) with license & citation
- Private by default

Install:
  pip install datasets pandas pyarrow huggingface_hub tqdm

Auth:
  export HUGGINGFACE_TOKEN=hf_xxx
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo, HfApi, upload_folder
from tqdm import tqdm

SPLIT_HINTS = {
    "train": ["train", "training"],
    "validation": ["valid", "validation", "dev"],
    "test": ["test", "eval", "evaluation"],
}

TEXT_EXTS = (".jsonl", ".csv")
PARQUET_EXTS = (".parquet", ".pq")

def guess_split(fname: str) -> str:
    low = fname.lower()
    for split, hints in SPLIT_HINTS.items():
        if any(h in low for h in hints):
            return split
    return "train"

def find_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (*TEXT_EXTS, *PARQUET_EXTS):
            files.append(p)
    return sorted(files)

def read_to_dataset(path: Path, columns: Optional[List[str]], dtype: Optional[Dict[str,str]], chunksize: int) -> Dataset:
    suf = path.suffix.lower()
    if suf in PARQUET_EXTS:
        df = pd.read_parquet(path, columns=columns)
        return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)

    if suf == ".csv":
        itr = pd.read_csv(path, usecols=columns, dtype=dtype, chunksize=chunksize)
    elif suf == ".jsonl":
        itr = pd.read_json(path, orient="records", lines=True, dtype=dtype, chunksize=chunksize)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    first = True
    out = None
    for chunk in tqdm(itr, desc=f"Reading {path.name}"):
        ds = Dataset.from_pandas(chunk.reset_index(drop=True), preserve_index=False)
        if first:
            out = ds
            first = False
        else:
            out = Dataset.from_dict({k: list(out[k]) + list(ds[k]) for k in out.features.keys()})
    if out is None:
        out = Dataset.from_pandas(pd.DataFrame(), preserve_index=False)
    return out

def ensure_repo(repo_id: str, private: bool):
    try:
        create_repo(repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"[info] create_repo: {e}", file=sys.stderr)

def write_dataset_card(tmpdir: Path, repo_id: str, license_str: str, desc: str, tags: List[str], citation: Optional[str]):
    readme = tmpdir / "README.md"
    parts = []
    parts.append(f"# {repo_id}\n")
    if desc:
        parts.append(desc.strip() + "\n")
    meta = []
    if license_str:
        meta.append(f"- **License**: {license_str}")
    if tags:
        meta.append(f"- **Tags**: {', '.join(tags)}")
    if meta:
        parts.append("\n" + "\n".join(meta) + "\n")
    if citation:
        parts.append("\n## Citation\n\n```\n" + citation.strip() + "\n```\n")
    readme.write_text("\n".join(parts), encoding="utf-8")

def push_raw_folder(repo_id: str, folder: Path, private: bool, license_str: str, desc: str, tags: List[str], citation: Optional[str]):
    # Write/merge README
    tmp = Path(".hf_tmp_card")
    tmp.mkdir(exist_ok=True)
    write_dataset_card(tmp, repo_id, license_str, desc, tags, citation)
    # Prefer user’s README if present; otherwise add ours
    user_readme = folder / "README.md"
    if not user_readme.exists():
        # upload our README first so the repo has a card
        upload_folder(
            repo_id=repo_id, repo_type="dataset",
            folder_path=str(tmp),
            path_in_repo="../../dev",
            private=private
        )
    # Upload the dataset folder as-is
    upload_folder(
        repo_id=repo_id, repo_type="dataset",
        folder_path=str(folder),
        path_in_repo="../../dev",
        private=private
    )
    try:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/my-private-dataset")
    ap.add_argument("--data-path", required=True, help="Folder containing CSV/JSONL/Parquet (or any files if --raw)")
    ap.add_argument("--private", action="store_true", help="Create/push as private dataset")
    ap.add_argument("--raw", action="store_true", help="Upload folder as-is (no DatasetDict build)")
    ap.add_argument("--columns", nargs="*", help="Optional column projection")
    ap.add_argument("--dtype", nargs="*", help='Optional dtype hints, e.g. text=str id=int')
    ap.add_argument("--chunksize", type=int, default=250_000, help="Rows per chunk for CSV/JSONL")
    ap.add_argument("--license", default="cc-by-sa-4.0", help="License string for card")
    ap.add_argument("--desc", default="", help="Short description for dataset card")
    ap.add_argument("--tags", nargs="*", default=[], help="Tags for dataset card")
    ap.add_argument("--citation", default="", help="Citation block for dataset card")
    args = ap.parse_args()

    dtype_map = None
    if args.dtype:
        dtype_map = {}
        for kv in args.dtype:
            k, v = kv.split("=")
            dtype_map[k] = v

    data_root = Path(args.data_path)
    if not data_root.exists():
        raise SystemExit(f"Data path not found: {data_root}")

    ensure_repo(args.repo_id, args.private)

    if args.raw:
        push_raw_folder(
            repo_id=args.repo_id,
            folder=data_root,
            private=args.private,
            license_str=args.license,
            desc=args.desc,
            tags=args.tags,
            citation=args.citation or None
        )
        print(f"[done] Uploaded raw folder to https://huggingface.co/datasets/{args.repo_id}")
        return

    files = find_files(data_root)
    if not files:
        raise SystemExit("No CSV/JSONL/Parquet files found.")

    # Build split-wise datasets
    split_to_parts: Dict[str, List[Dataset]] = {}
    for f in files:
        split = guess_split(f.name)
        ds = read_to_dataset(f, args.columns, dtype_map, args.chunksize)
        split_to_parts.setdefault(split, []).append(ds)

    ddict = {}
    for split, parts in split_to_parts.items():
        if len(parts) == 1:
            ddict[split] = parts[0]
        else:
            # Concatenate
            base = parts[0]
            for nxt in parts[1:]:
                base = Dataset.from_dict({k: list(base[k]) + list(nxt[k]) for k in base.features.keys()})
            ddict[split] = base

    dset = DatasetDict(ddict)
    # Add a README card on first push
    dset.push_to_hub(args.repo_id, private=args.private)
    # After first push, upload/overwrite README with license/tags
    tmp = Path(".hf_tmp_card")
    tmp.mkdir(exist_ok=True)
    write_dataset_card(tmp, args.repo_id, args.license, args.desc, args.tags, args.citation or None)
    upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(tmp), path_in_repo="../../dev", private=args.private)
    try:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    except Exception:
        pass

    print(f"[done] Pushed DatasetDict to https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()
