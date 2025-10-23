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

# at top
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets  # already imported in your file

import json
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
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
            path_in_repo=".."
        )
    # Upload the dataset folder as-is
    upload_folder(
        repo_id=repo_id, repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=".."
    )
    try:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    except Exception:
        pass

# --- imports at top (make sure these exist) ---
import json
import pandas as pd
from datasets import Dataset, DatasetDict

# Consider these as scalar cell types
_SCALAR_TYPES = (str, int, float, bool, type(None))

def _is_nonscalar(x):
    # list/dict/tuple/set/np arrays etc. treated as non-scalar
    if isinstance(x, _SCALAR_TYPES):
        return False
    # pandas/Numpy NaN: treat as scalar-ish
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return False
    except Exception:
        pass
    # everything else that isn't a basic scalar
    return True

def _jsonify_column(series: pd.Series) -> pd.Series:
    # JSON-stringify non-scalars; leave scalars as-is
    def _to_str(x):
        if _is_nonscalar(x):
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)
        return x
    s = series.map(_to_str)
    # force a consistent string dtype so Arrow never infers a list
    try:
        return s.astype("string")
    except Exception:
        return s.astype(str)

def concat_parts_schema_safe(parts):
    """
    1) Find columns that are non-scalar in ANY part.
    2) JSON-stringify those columns in ALL parts.
    3) Outer-concat -> Dataset.
    """
    # First pass: union of columns + detect non-scalar columns globally
    all_cols = set()
    nonscalar_cols = set()
    dfs_first = []
    for p in parts:
        df = p.to_pandas()
        dfs_first.append(df)
        all_cols.update(df.columns)
        for c in df.columns:
            # if ANY row is non-scalar, mark this column
            if df[c].map(_is_nonscalar).any():
                nonscalar_cols.add(c)

    # Second pass: align columns and normalize types
    dfs = []
    for df in dfs_first:
        # add missing columns
        for c in all_cols:
            if c not in df.columns:
                df[c] = None
        # normalize non-scalar columns to JSON strings everywhere
        for c in nonscalar_cols:
            df[c] = _jsonify_column(df[c])
        # (optional) also coerce obvious JSON-ish fields if present
        for c in ("input", "target", "source", "context", "citations"):
            if c in df.columns:
                # if any non-scalar sneaked in, stringify
                if df[c].map(_is_nonscalar).any():
                    df[c] = _jsonify_column(df[c])
        dfs.append(df)

    big = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    # Ensure pyarrow won’t try to convert something as list
    for c in nonscalar_cols:
        big[c] = big[c].astype("string")
    return Dataset.from_pandas(big, preserve_index=False)


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
            ddict[split] = concat_parts_schema_safe(parts)

    dset = DatasetDict(ddict)
    # Add a README card on first push
    dset.push_to_hub(args.repo_id, private=args.private)
    # After first push, upload/overwrite README with license/tags
    tmp = Path(".hf_tmp_card")
    tmp.mkdir(exist_ok=True)
    write_dataset_card(tmp, args.repo_id, args.license, args.desc, args.tags, args.citation or None)
    upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(tmp), path_in_repo="..")
    try:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    except Exception:
        pass

    print(f"[done] Pushed DatasetDict to https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()