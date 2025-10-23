#!/usr/bin/env python3
"""
Wikivoyage -> Hugging Face (private) dataset builder

Features
- Streams the enwikivoyage XML dump (.bz2) without loading into memory.
- Extracts per-section passages (title, section, text).
- Parses coordinates from common templates ({{geo|...}} / {{coord|...}}).
- Sentence-aware chunking of long sections to ~1k chars (configurable).
- Deterministic train/validation split by hashing page title.
- Pushes to a private Hugging Face repo.

Install:
  pip install mwparserfromhell datasets huggingface_hub tqdm regex

Example:
  python build_wikivoyage_hf.py \
    --dump-url https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2 \
    --repo-id your-username/wikivoyage-travel-sections \
    --private \
    --chunk-target-chars 1000 \
    --val-ratio 0.01
"""

import argparse
import bz2
import hashlib
import io
import os
import re
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import mwparserfromhell
import regex as regex_re
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo, upload_folder
from tqdm import tqdm

# --- Utilities ----------------------------------------------------------------

SECTION_HEADING_RE = regex_re.compile(r"^={2,}\s*(.+?)\s*={2,}\s*$")
# Typical Wikivoyage section names we care about (not strictly required)
CANON_SECTIONS = {
    "understand", "get in", "get around", "see", "do", "buy", "eat", "drink",
    "sleep", "learn", "work", "connect", "stay safe", "go next", "respect",
    "cope", "itineraries", "overview", "history", "climate", "festivals",
}

# Capture {{geo|lat|lon}} or {{coord|lat|lon}} with optional params.
# This is a pragmatic matcher; wikitext has many variants.
COORD_RE = regex_re.compile(
    r"\{\{\s*(?:geo|coord)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)"
    r"(?:\|[^}]*)?\}\}",
    flags=regex_re.IGNORECASE
)

WIKILINK_RE = regex_re.compile(r"\[\[(.+?)(\|.+?)?\]\]")

def strip_markup_keep_text(wikitext: str) -> str:
    """
    Use mwparserfromhell to strip templates/markup while keeping readable text.
    """
    wikicode = mwparserfromhell.parse(wikitext)
    text = wikicode.strip_code(normalize=True, collapse=True)
    # Collapse excessive whitespace
    text = regex_re.sub(r"[ \t]+", " ", text)
    text = regex_re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def sentence_chunk(text: str, target_chars: int = 1000, hard_max: int = 1600) -> List[str]:
    """
    Greedy sentence-aware chunking using simple sentence boundaries.
    We intentionally avoid heavy tokenizers to keep deps minimal.
    """
    # Split on ., !, ? followed by space/newline (very rough)
    sents = regex_re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= target_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            # if sentence itself is huge, hard-split
            if not cur:
                chunks.append(s[:hard_max].strip())
                rest = s[hard_max:].strip()
                if rest:
                    chunks.append(rest[:hard_max])
            else:
                chunks.append(cur)
                cur = s
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]

def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def extract_coords(wikitext: str) -> Tuple[Optional[float], Optional[float]]:
    m = COORD_RE.search(wikitext)
    if not m:
        return None, None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        return lat, lon
    except Exception:
        return None, None

def extract_links(wikitext: str) -> List[str]:
    # Return raw link targets (before |text)
    links = []
    for m in WIKILINK_RE.finditer(wikitext):
        target = m.group(1)
        if target:
            links.append(target.strip())
    return links[:200]  # cap to keep examples light

@dataclass
class SectionRecord:
    page_id: str
    title: str
    rev_timestamp: str
    section_title: str
    section_index: int
    chunk_index: int
    text: str
    full_section_text: str
    lat: Optional[float]
    lon: Optional[float]
    links: List[str]
    url: str  # constructed https://en.wikivoyage.org/wiki/Title

# --- XML streaming parser -----------------------------------------------------

def iter_pages_from_dump(dump_path: str) -> Iterator[Dict[str, str]]:
    """
    Stream pages from a Wikivoyage dump (bz2). Yields dicts with keys:
      page_id, title, rev_timestamp, wikitext
    Only namespace 0 (main) pages are yielded.
    """
    # Wikipedia XML is large; use iterparse and clear elements to free memory.
    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        # MediaWiki xml uses many namespaces; we match tags by suffix.
        for event, elem in context:
            if elem.tag.endswith("page"):
                ns = elem.find("./{*}ns")
                if ns is not None and ns.text != "0":
                    elem.clear()
                    continue
                title_el = elem.find("./{*}title")
                id_el = elem.find("./{*}id")
                rev = elem.find("./{*}revision")
                ts_el = rev.find("./{*}timestamp") if rev is not None else None
                text_el = rev.find("./{*}text") if rev is not None else None

                title = title_el.text if title_el is not None else ""
                page_id = id_el.text if id_el is not None else ""
                rev_ts = ts_el.text if ts_el is not None else ""
                wikitext = text_el.text if text_el is not None else ""

                if title and wikitext:
                    yield {
                        "page_id": page_id,
                        "title": title,
                        "rev_timestamp": rev_ts,
                        "wikitext": wikitext,
                    }
                elem.clear()

# --- Per-page -> per-section records -----------------------------------------

def page_to_section_records(page: Dict[str, str],
                            chunk_target_chars: int = 1000) -> List[SectionRecord]:
    """
    Break a page into (section -> chunks). Always includes a synthetic
    'Lead' section for text before the first heading (if any).
    """
    title = page["title"]
    page_id = page["page_id"]
    ts = page["rev_timestamp"]
    wikitext = page["wikitext"]

    # Extract coordinates and links from raw wikitext (before stripping)
    lat, lon = extract_coords(wikitext)
    links = extract_links(wikitext)

    text = strip_markup_keep_text(wikitext)
    if not text:
        return []

    # Split into sections by wikitext headings.
    # We'll use mwparserfromhell’s headings for robustness.
    wikicode = mwparserfromhell.parse(wikitext)
    nodes = list(wikicode.nodes)

    # Collect tuples: (section_title, section_text)
    sections: List[Tuple[str, str]] = []
    cur_title = "Lead"
    cur_buf: List[str] = []

    def flush():
        if cur_buf:
            sec_text_wt = "".join(cur_buf).strip()
            if sec_text_wt:
                clean = strip_markup_keep_text(sec_text_wt)
                if clean:
                    sections.append((cur_title, clean))

    for n in nodes:
        if isinstance(n, mwparserfromhell.nodes.heading.Heading):
            # flush previous
            flush()
            cur_title = str(n.title).strip()
            cur_buf = []
        else:
            cur_buf.append(str(n))

    flush()

    # Fallback: if the heading parse failed, just make a single "Article" section
    if not sections:
        sections = [("Article", text)]

    recs: List[SectionRecord] = []
    for idx, (sec_title, sec_text) in enumerate(sections):
        # Optional: bias to canonical names
        norm_name = sec_title.lower().strip()
        if norm_name in CANON_SECTIONS:
            sec_title = sec_title.title()

        # Chunk long sections
        chunks = sentence_chunk(sec_text, target_chars=chunk_target_chars)
        for cidx, chunk in enumerate(chunks):
            recs.append(
                SectionRecord(
                    page_id=page_id,
                    title=title,
                    rev_timestamp=ts,
                    section_title=sec_title,
                    section_index=idx,
                    chunk_index=cidx,
                    text=chunk,
                    full_section_text=sec_text if cidx == 0 else "",
                    lat=lat,
                    lon=lon,
                    links=links,
                    url=f"https://en.wikivoyage.org/wiki/{title.replace(' ', '_')}"
                )
            )
    return recs

# --- HF push ------------------------------------------------------------------

def ensure_repo(repo_id: str, private: bool):
    api = HfApi()
    try:
        create_repo(repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True)
    except Exception as e:
        # If it already exists and visibility differs, we still proceed.
        print(f"[info] create_repo: {e}", file=sys.stderr)

def build_and_push(dump_path: str,
                   repo_id: str,
                   private: bool,
                   chunk_target_chars: int,
                   val_ratio: float,
                   max_pages: Optional[int] = None):
    ensure_repo(repo_id, private)

    # Stream pages -> section records -> dicts for HF
    def gen_records() -> Iterator[Dict]:
        n = 0
        for page in iter_pages_from_dump(dump_path):
            for rec in page_to_section_records(page, chunk_target_chars):
                yield {
                    "page_id": rec.page_id,
                    "title": rec.title,
                    "rev_timestamp": rec.rev_timestamp,
                    "section_title": rec.section_title,
                    "section_index": rec.section_index,
                    "chunk_index": rec.chunk_index,
                    "text": rec.text,
                    "full_section_text": rec.full_section_text,
                    "lat": rec.lat,
                    "lon": rec.lon,
                    "links": rec.links,
                    "source_url": rec.url,
                    "source": "enwikivoyage",
                }
            n += 1
            if max_pages and n >= max_pages:
                break

    # Materialize in temp folder (arrow cache), then push
    with tempfile.TemporaryDirectory() as tmpdir:
        print("[info] Building HF dataset (this may take a while)...")
        data_iter = gen_records()
        # Stream into memory in batches so we can split deterministically
        # For simplicity, we’ll collect then split; for truly massive builds,
        # switch to disk-backed writers.
        rows: List[Dict] = []
        for row in tqdm(data_iter, desc="Collecting records"):
            rows.append(row)

        if not rows:
            raise RuntimeError("No records parsed. Check dump path.")

        # Deterministic split: hash by title
        train_rows, val_rows = [], []
        for r in rows:
            h = stable_hash(r["title"]) % 10_000
            if (h / 10_000.0) < val_ratio:
                val_rows.append(r)
            else:
                train_rows.append(r)

        train = Dataset.from_list(train_rows)
        val = Dataset.from_list(val_rows) if val_rows else Dataset.from_list(train_rows[:0])
        dset = DatasetDict({"train": train, "validation": val})

        print(f"[info] Train size: {len(train)} | Val size: {len(val)}")
        dset.push_to_hub(repo_id, private=private)

    print(f"[done] Pushed to https://huggingface.co/datasets/{repo_id} (private={private})")

# --- Download helper ----------------------------------------------------------

def download_dump(dump_url: str, out_path: str):
    print(f"[info] Downloading dump from {dump_url}")
    with urllib.request.urlopen(dump_url) as resp, open(out_path, "wb") as f:
        # Stream to file with a progress bar
        total = resp.length or 0
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading dump") as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"[info] Saved to {out_path}")

# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-url", required=True,
                        help="URL to enwikivoyage XML bz2 (e.g., latest).")
    parser.add_argument("--repo-id", required=True,
                        help="Hugging Face dataset repo id, e.g. user/wikivoyage-travel-sections")
    parser.add_argument("--private", action="store_true", help="Create/push as private dataset")
    parser.add_argument("--chunk-target-chars", type=int, default=1000,
                        help="Approx target chars per chunk for section text")
    parser.add_argument("--val-ratio", type=float, default=0.01,
                        help="Validation split ratio (deterministic by title hash)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Optional limit for quick trials")
    parser.add_argument("--dump-path", default=None,
                        help="If provided, skip download and read this local .bz2 path")

    args = parser.parse_args()

    # Auth via env var (recommended)
    if not os.environ.get("HUGGINGFACE_TOKEN"):
        print(
            "[warning] HUGGINGFACE_TOKEN not set in environment. "
            "If you see auth errors, run: 'huggingface-cli login' or set HUGGINGFACE_TOKEN.",
            file=sys.stderr
        )

    dump_path = args.dump_path
    if not dump_path:
        # download to temp folder
        tmpdir = tempfile.mkdtemp(prefix="wikivoyage_dump_")
        dump_path = os.path.join(tmpdir, "enwikivoyage.xml.bz2")
        download_dump(args.dump_url, dump_path)

    build_and_push(
        dump_path=dump_path,
        repo_id=args.repo_id,
        private=args.private,
        chunk_target_chars=args.chunk_target_chars,
        val_ratio=args.val_ratio,
        max_pages=args.max_pages,
    )

if __name__ == "__main__":
    main()