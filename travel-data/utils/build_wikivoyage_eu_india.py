#!/usr/bin/env python3
# build_wikivoyage_eu_india.py
"""
Filter enwikivoyage dump to EU + India, then push as a private HF dataset.

Usage:
  python build_wikivoyage_eu_india.py \
    --dump-path enwikivoyage-latest-pages-articles.xml.bz2 \
    --repo-id <user>/wikivoyage-eu-india-sections \
    --private
"""

import argparse, bz2, hashlib, os, regex as re, xml.etree.ElementTree as ET, tempfile, sys
from typing import Dict, Iterator, List, Optional, Tuple
import mwparserfromhell
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo

# --- Regions ------------------------------------------------------------------
EU_COUNTRIES = {
    # EU-27 (Oct 2025)
    "Austria","Belgium","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark",
    "Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy",
    "Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal",
    "Romania","Slovakia","Slovenia","Spain","Sweden",
}
ALLOW_COUNTRIES = EU_COUNTRIES | {"India"}
ALLOW_TOKENS = {c.lower() for c in ALLOW_COUNTRIES}

# Coarse geo fallback (very loose bounding boxes)
def in_coarse_bbox(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None: return False
    # EU-wide loose bbox
    if 34.0 <= lat <= 72.5 and -25.0 <= lon <= 45.5:
        return True
    # India bbox
    if 6.0 <= lat <= 37.6 and 68.0 <= lon <= 97.6:
        return True
    return False

# --- Parsing helpers -----------------------------------------------------------
COORD_RE = re.compile(
    r"\{\{\s*(?:geo|coord)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)",
    re.I
)
CAT_RE = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)")
SECTION_NAMES = {
    "understand","get in","get around","see","do","buy","eat","drink","sleep",
    "learn","work","connect","stay safe","go next","respect","cope","itineraries",
    "overview","history","climate","festivals",
}

def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def extract_coords(wt: str):
    m = COORD_RE.search(wt)
    if not m: return None, None
    try: return float(m.group(1)), float(m.group(2))
    except: return None, None

def extract_categories(wt: str) -> List[str]:
    return [m.group(1).strip() for m in CAT_RE.finditer(wt)]

def match_region_by_meta(wt: str) -> bool:
    # 1) {{IsPartOf|...}} crumbs
    if "{{ispartof" in wt.lower():
        low = wt.lower()
        for name in ALLOW_TOKENS:
            if name in low:
                return True
    # 2) Category tokens
    for cat in extract_categories(wt):
        if cat.strip().lower() in ALLOW_TOKENS:
            return True
        # also allow "Cities in France", "Tourism in India", etc.
        c = cat.lower()
        for name in ALLOW_TOKENS:
            if name in c:
                return True
    return False

def strip_markup(wt: str) -> str:
    code = mwparserfromhell.parse(wt)
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text).strip()
    return text

def sentence_chunk(text: str, target_chars=1000, hard_max=1600):
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sents:
        if not s: continue
        if len(cur)+len(s)+1 <= target_chars:
            cur = (cur+" "+s).strip() if cur else s
        else:
            if not cur:
                out.append(s[:hard_max].strip())
                rest = s[hard_max:].strip()
                if rest: out.append(rest[:hard_max])
            else:
                out.append(cur); cur = s
    if cur: out.append(cur)
    return [c for c in out if c]

def iter_pages(dump_path: str) -> Iterator[Dict[str,str]]:
    with bz2.open(dump_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if not elem.tag.endswith("page"): continue
            ns = elem.find("./{*}ns")
            if ns is not None and ns.text != "0":
                elem.clear(); continue
            title_el = elem.find("./{*}title")
            id_el = elem.find("./{*}id")
            rev = elem.find("./{*}revision")
            ts_el = rev.find("./{*}timestamp") if rev is not None else None
            txt_el = rev.find("./{*}text") if rev is not None else None
            title = title_el.text if title_el is not None else ""
            page_id = id_el.text if id_el is not None else ""
            ts = ts_el.text if ts_el is not None else ""
            wt = txt_el.text if txt_el is not None else ""
            if title and wt:
                yield {"title": title, "page_id": page_id, "rev_timestamp": ts, "wikitext": wt}
            elem.clear()

def page_to_rows(page: Dict[str,str], chunk_target_chars: int):
    title = page["title"]; wt = page["wikitext"]; ts = page["rev_timestamp"]; pid = page["page_id"]
    lat, lon = extract_coords(wt)
    # Region filter: metadata OR coarse geo
    if not (match_region_by_meta(wt) or in_coarse_bbox(lat, lon)):
        return []

    code = mwparserfromhell.parse(wt)
    nodes = list(code.nodes)
    sections = []
    cur_title, buf = "Lead", []
    def flush():
        if buf:
            raw = "".join(buf).strip()
            txt = strip_markup(raw)
            if txt: sections.append((cur_title, txt))
    for n in nodes:
        if isinstance(n, mwparserfromhell.nodes.heading.Heading):
            flush(); cur_title = str(n.title).strip(); buf = []
        else:
            buf.append(str(n))
    flush()
    if not sections:
        txt = strip_markup(wt)
        if txt: sections = [("Article", txt)]

    url = f"https://en.wikivoyage.org/wiki/{title.replace(' ','_')}"
    out = []
    for sidx, (sec_title, sec_text) in enumerate(sections):
        if sec_title.lower() in SECTION_NAMES:
            sec_title = sec_title.title()
        chunks = sentence_chunk(sec_text, target_chars=chunk_target_chars)
        for cidx, chunk in enumerate(chunks):
            out.append({
                "page_id": pid, "title": title, "rev_timestamp": ts,
                "section_title": sec_title, "section_index": sidx,
                "chunk_index": cidx, "text": chunk,
                "full_section_text": sec_text if cidx==0 else "",
                "lat": lat, "lon": lon, "links": [],
                "source_url": url, "source": "enwikivoyage"
            })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-path", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--chunk-target-chars", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    create_repo(repo_id=args.repo_id, private=args.private, repo_type="dataset", exist_ok=True)

    rows = []
    n = 0
    for page in iter_pages(args.dump_path):
        rows.extend(page_to_rows(page, args.chunk_target_chars))
        n += 1
        if args.max_pages and n >= args.max_pages:
            break

    if not rows:
        print("[error] No rows after filtering. Check dump path and filters.", file=sys.stderr); sys.exit(1)

    # deterministic split by title
    train, val = [], []
    for r in rows:
        h = stable_hash(r["title"]) % 10_000
        (val if (h/10_000.0) < args.val_ratio else train).append(r)

    dset = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val) if val else Dataset.from_list(train[:0])
    })
    print(f"[info] train={len(train)} val={len(val)} (from {n} pages scanned)")
    dset.push_to_hub(args.repo_id, private=args.private)
    print(f"[done] https://huggingface.co/datasets/{args.repo_id} (private)")
if __name__ == "__main__":
    main()