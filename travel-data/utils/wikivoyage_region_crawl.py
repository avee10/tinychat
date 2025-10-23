#!/usr/bin/env python3
"""
wikivoyage_region_crawl.py — Build a private HF dataset with ONLY US/EU pages.

- Crawls en.wikivoyage.org Category trees (United States, Europe) via MediaWiki API
- Recurses into subcategories (depth-limited)
- Collects main-namespace pages (ns=0), downloads latest wikitext
- Converts to per-section chunks (same as build_wikivoyage_hf.py)
- Pushes DatasetDict(train/validation) to a private Hugging Face repo

Install:
  pip install requests mwparserfromhell datasets huggingface_hub tqdm regex backoff
Auth:
  export HUGGINGFACE_TOKEN=hf_xxx
"""
# --- add at top (new) ---
import os, random, time
import requests, backoff

UA = os.environ.get(
    "WVF_USER_AGENT",
    # <project>/<version> (<homepage or repo>; <email>)
    "Aveek-TravelCrawler/0.1 (https://huggingface.co/aveekmukherjee; mailto:YOUR_EMAIL@EXAMPLE.COM)"
)
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": UA,
    "From": os.environ.get("WVF_CONTACT", "YOUR_EMAIL@EXAMPLE.COM"),  # optional but nice
    "Accept": "application/json"
})
API = "https://en.wikivoyage.org/w/api.php"

# polite rate limit ~1 req/sec with jitter
def polite_sleep():
    time.sleep(1.0 + random.random() * 0.5)

# backoff on network + 429/503; for 403, lower rate + retry a few times
class RetryableHTTPError(Exception): pass

@backoff.on_exception(
    backoff.expo,
    (requests.RequestException, RetryableHTTPError),
    max_time=120,
    jitter=backoff.full_jitter,
)
def mw_api(params: dict) -> dict:
    params = {**params, "format": "json"}
    polite_sleep()
    r = SESSION.get(API, params=params, timeout=30)
    # Map some statuses to retries
    if r.status_code in (429, 500, 502, 503, 504):
        raise RetryableHTTPError(f"Transient HTTP {r.status_code}")
    if r.status_code == 403:
        # Often due to missing UA/rate—slow down and try a couple times
        polite_sleep()
        raise RetryableHTTPError("HTTP 403 (likely rate/UA); retrying slowly")
    r.raise_for_status()
    j = r.json()
    # MediaWiki can also signal errors in JSON
    if "error" in j:
        code = j["error"].get("code")
        if code in {"ratelimited", "internal_api_error"}:
            raise RetryableHTTPError(f"API error: {code}")
    return j


import argparse, os, sys, time, hashlib, json
from collections import deque, defaultdict
from typing import Dict, List, Set, Iterator, Tuple, Optional

import requests, backoff
import mwparserfromhell
import regex as re
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo
from tqdm import tqdm

API = "https://en.wikivoyage.org/w/api.php"

SECTION_HEADING_RE = re.compile(r"^={2,}\s*(.+?)\s*={2,}\s*$")
WIKILINK_RE = re.compile(r"\[\[(.+?)(\|.+?)?\]\]")
COORD_RE = re.compile(
    r"\{\{\s*(?:geo|coord)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)"
    r"(?:\|[^}]*)?\}\}",
    flags=re.IGNORECASE
)
CANON_SECTIONS = {
    "understand", "get in", "get around", "see", "do", "buy", "eat", "drink",
    "sleep", "learn", "work", "connect", "stay safe", "go next", "respect",
    "cope", "itineraries", "overview", "history", "climate", "festivals",
}

def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def strip_markup_keep_text(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def sentence_chunk(text: str, target_chars: int = 1000, hard_max: int = 1600):
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= target_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if not cur:
                chunks.append(s[:hard_max].strip())
                rest = s[hard_max:].strip()
                if rest: chunks.append(rest[:hard_max])
            else:
                chunks.append(cur); cur = s
    if cur: chunks.append(cur)
    return [c for c in chunks if c]

def extract_coords(wikitext: str):
    m = COORD_RE.search(wikitext)
    if not m: return None, None
    try: return float(m.group(1)), float(m.group(2))
    except Exception: return None, None

def extract_links(wikitext: str):
    links = []
    for m in WIKILINK_RE.finditer(wikitext):
        target = (m.group(1) or "").strip()
        if target: links.append(target)
    return links[:200]

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=120)
def mw_api(params: Dict):
    params = dict(params)
    params["format"] = "json"
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def crawl_category_tree(root_cats: List[str], max_depth: int = 5) -> Tuple[Set[str], Set[str]]:
    """Return (page_titles, visited_cats) reachable from root categories."""
    visited_cats: Set[str] = set()
    pages: Set[str] = set()
    q = deque([(c, 0) for c in root_cats])

    while q:
        cat, depth = q.popleft()
        if cat in visited_cats or depth > max_depth:
            continue
        visited_cats.add(cat)

        cmtitle = f"Category:{cat}"
        cont = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cmtitle,
                "cmlimit": "500",
                "cmtype": "page|subcat",
            }
            if cont: params["cmcontinue"] = cont
            data = mw_api(params)
            cms = data.get("query", {}).get("categorymembers", [])
            for item in cms:
                if item.get("ns") == 0:  # main namespace page
                    pages.add(item["title"])
                elif item.get("ns") == 14:  # Category
                    subcat = item["title"].split("Category:",1)[-1]
                    q.append((subcat, depth+1))
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont: break
    return pages, visited_cats

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=120)
def fetch_wikitext_batch(titles: List[str]) -> Dict[str, Dict]:
    """Return dict title -> {wikitext, timestamp, pageid}."""
    out = {}
    # MediaWiki limit ~50 titles per query
    for i in range(0, len(titles), 50):
        batch = titles[i:i+50]
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content|timestamp",
            "rvslots": "main",
            "titles": "|".join(batch),
            "formatversion": "2",
        }
        data = mw_api(params)
        pages = data.get("query", {}).get("pages", [])
        for p in pages:
            title = p.get("title")
            revs = p.get("revisions") or []
            if not title or not revs:
                continue
            slot = revs[0].get("slots", {}).get("main", {})
            wikitext = slot.get("content", "")
            ts = revs[0].get("timestamp", "")
            out[title] = {"wikitext": wikitext, "timestamp": ts, "pageid": str(p.get("pageid", ""))}
    return out

def page_to_rows(title: str, pageid: str, ts: str, wikitext: str, chunk_target_chars: int):
    lat, lon = extract_coords(wikitext)
    links = extract_links(wikitext)
    # section split
    code = mwparserfromhell.parse(wikitext)
    nodes = list(code.nodes)

    sections = []
    cur_title, buf = "Lead", []
    def flush():
        if buf:
            wt = "".join(buf).strip()
            txt = strip_markup_keep_text(wt)
            if txt: sections.append((cur_title, txt))
    for n in nodes:
        if isinstance(n, mwparserfromhell.nodes.heading.Heading):
            flush(); cur_title = str(n.title).strip(); buf = []
        else:
            buf.append(str(n))
    flush()
    if not sections:
        text = strip_markup_keep_text(wikitext)
        if text: sections = [("Article", text)]

    url = f"https://en.wikivoyage.org/wiki/{title.replace(' ', '_')}"
    rows = []
    for sidx, (sec_title, sec_text) in enumerate(sections):
        norm = sec_title.lower().strip()
        if norm in CANON_SECTIONS: sec_title = sec_title.title()
        chunks = sentence_chunk(sec_text, target_chars=chunk_target_chars)
        for cidx, chunk in enumerate(chunks):
            rows.append({
                "page_id": pageid,
                "title": title,
                "rev_timestamp": ts,
                "section_title": sec_title,
                "section_index": sidx,
                "chunk_index": cidx,
                "text": chunk,
                "full_section_text": sec_text if cidx == 0 else "",
                "lat": lat, "lon": lon,
                "links": links,
                "source_url": url,
                "source": "enwikivoyage",
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="HF repo e.g. user/wikivoyage-us-eu")
    ap.add_argument("--private", action="store_true", help="Create/push as private")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--chunk-target-chars", type=int, default=1000)
    ap.add_argument("--max-depth", type=int, default=5, help="Category recursion depth")
    ap.add_argument("--roots", nargs="*", default=["United States","Europe"],
                    help="Root categories to crawl (without 'Category:')")
    ap.add_argument("--max-pages", type=int, default=None, help="Optional page cap for quick trials")
    args = ap.parse_args()

    if not os.environ.get("HUGGINGFACE_TOKEN"):
        print("[warn] HUGGINGFACE_TOKEN not set; run `huggingface-cli login` or export the token.", file=sys.stderr)

    # Prepare repo
    try:
        create_repo(repo_id=args.repo_id, private=args.private, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"[info] {e}", file=sys.stderr)

    print(f"[info] Crawling roots: {args.roots} (max_depth={args.max_depth})")
    pages, cats = crawl_category_tree(args.roots, max_depth=args.max_depth)
    print(f"[info] Found pages: {len(pages)} | categories visited: {len(cats)}")

    titles = sorted(pages)
    if args.max_pages:
        titles = titles[:args.max_pages]
        print(f"[info] Limiting to first {len(titles)} pages for trial")

    # Fetch content in batches with retries
    rows = []
    for i in tqdm(range(0, len(titles), 200), desc="Downloading pages"):
        batch_titles = titles[i:i+200]
        info = fetch_wikitext_batch(batch_titles)
        for t, meta in info.items():
            wikitext = meta.get("wikitext", "")
            if not wikitext:
                continue
            rows.extend(page_to_rows(t, meta.get("pageid",""), meta.get("timestamp",""),
                                     wikitext, args.chunk_target_chars))

    if not rows:
        print("[error] No rows parsed.")
        sys.exit(1)

    # Deterministic split by title hash
    train_rows, val_rows = [], []
    for r in rows:
        h = stable_hash(r["title"]) % 10_000
        if (h / 10_000.0) < args.val_ratio:
            val_rows.append(r)
        else:
            train_rows.append(r)

    ds_train = Dataset.from_list(train_rows)
    ds_val = Dataset.from_list(val_rows) if val_rows else Dataset.from_list(train_rows[:0])
    dset = DatasetDict({"train": ds_train, "validation": ds_val})
    print(f"[info] Train size: {len(ds_train)} | Val size: {len(ds_val)}")
    dset.push_to_hub(args.repo_id, private=args.private)
    print(f"[done] Pushed to https://huggingface.co/datasets/{args.repo_id} (private={args.private})")

if __name__ == "__main__":
    main()
