#!/usr/bin/env python3
# build_wikidata_travel_eu_india.py
"""
EU+India travel-relevant pages via Wikidata → English Wikipedia → HF dataset.

Usage:
  export WV_USER_AGENT="Aveek-TravelBuilder/0.1 (mailto:you@example.com)"
  python build_wikidata_travel_eu_india.py \
    --repo-id <user>/wikipedia-travel-eu-india \
    --private
"""

import argparse, os, sys, time, random, hashlib, json, textwrap
from typing import List, Dict, Tuple, Set
import requests, mwparserfromhell, regex as re
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo

WD_SPARQL = "https://query.wikidata.org/sparql"
WP_API = "https://en.wikipedia.org/w/api.php"

EU_QIDS = {
    # EU-27 Wikidata QIDs (stable)
    "Q40","Q31","Q219","Q224","Q229","Q213","Q35","Q191","Q33","Q142","Q183","Q41",
    "Q28","Q27","Q38","Q211","Q37","Q32","Q233","Q55","Q36","Q45","Q218","Q214",
    "Q215","Q29","Q34"
}
INDIA_QID = "Q668"

UA = os.environ.get("WV_USER_AGENT", "Aveek-TravelBuilder/0.1 (mailto:YOUR_EMAIL@example.com)")
SESS = requests.Session()
SESS.headers.update({"User-Agent": UA, "Accept": "application/json"})

def polite_sleep(): time.sleep(1.0 + random.random()*0.5)

def shasum(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def wd_query(q: str) -> List[Dict]:
    polite_sleep()
    r = SESS.get(WD_SPARQL, params={"query": q, "format": "json"}, timeout=45)
    r.raise_for_status()
    return r.json()["results"]["bindings"]

def get_titles_for_entities(qids: List[str]) -> Dict[str, str]:
    titles = {}
    for i in range(0, len(qids), 50):
        polite_sleep()
        batch = qids[i:i+50]
        r = SESS.get("https://www.wikidata.org/w/api.php", params={
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "sitelinks/urls", "sitefilter": "enwiki", "format": "json"
        }, timeout=45)
        r.raise_for_status()
        for qid, ent in r.json().get("entities", {}).items():
            site = ent.get("sitelinks", {}).get("enwiki")
            if site and site.get("title"):
                titles[qid] = site["title"]
    return titles

def fetch_wikitext(titles: List[str]) -> Dict[str, Dict]:
    out = {}
    for i in range(0, len(titles), 50):
        polite_sleep()
        batch = titles[i:i+50]
        r = SESS.get(WP_API, params={
            "action": "query", "prop": "revisions", "rvprop": "content|timestamp",
            "rvslots": "main", "titles": "|".join(batch), "formatversion": "2", "format": "json"
        }, timeout=45)
        r.raise_for_status()
        for p in r.json().get("query", {}).get("pages", []):
            title = p.get("title")
            revs = p.get("revisions") or []
            if title and revs:
                slot = revs[0].get("slots", {}).get("main", {})
                wt = slot.get("content", "")
                ts = revs[0].get("timestamp", "")
                out[title] = {"wikitext": wt, "timestamp": ts, "pageid": str(p.get("pageid",""))}
    return out

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

def page_to_rows(title: str, pid: str, ts: str, wt: str, source_tag: str):
    # split to sections
    code = mwparserfromhell.parse(wt); nodes = list(code.nodes)
    sections = []; cur_title, buf = "Lead", []
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
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ','_')}"
    rows = []
    for sidx, (sec_title, sec_text) in enumerate(sections):
        chunks = sentence_chunk(sec_text, target_chars=1000)
        for cidx, chunk in enumerate(chunks):
            rows.append({
                "page_id": pid, "title": title, "rev_timestamp": ts,
                "section_title": sec_title, "section_index": sidx,
                "chunk_index": cidx, "text": chunk,
                "full_section_text": sec_text if cidx==0 else "",
                "lat": None, "lon": None, "links": [],
                "source_url": url, "source": source_tag
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None, help="Max entities per category (debug)")
    args = ap.parse_args()

    create_repo(repo_id=args.repo_id, private=args.private, repo_type="dataset", exist_ok=True)

    country_filter = "VALUES ?country { " + " ".join(f"wd:{q}" for q in sorted(EU_QIDS|{INDIA_QID})) + " }"

    # Queries (balanced & safe sizes)
    QUERIES = {
        "city": f"""
SELECT DISTINCT ?item WHERE {{
  {country_filter}
  ?item wdt:P31/wdt:P279* wd:Q515 .      # instance of city (or subclass)
  ?item wdt:P17 ?country .
  OPTIONAL {{ ?item wdt:P1082 ?pop . }}
}} LIMIT 20000
""",
        "airport": f"""
SELECT DISTINCT ?item WHERE {{
  {country_filter}
  ?item wdt:P31/wdt:P279* wd:Q1248784 .  # instance of airport
  ?item wdt:P17 ?country .
}} LIMIT 20000
""",
        "unesco": f"""
SELECT DISTINCT ?item WHERE {{
  {country_filter}
  ?item wdt:P1435 wd:Q9259 .             # UNESCO World Heritage Site
  ?item wdt:P17 ?country .
}} LIMIT 20000
""",
        "rail_station": f"""
SELECT DISTINCT ?item WHERE {{
  {country_filter}
  ?item wdt:P31/wdt:P279* wd:Q55488 .    # railway station
  ?item wdt:P17 ?country .
}} LIMIT 20000
"""
    }

    all_qids: Dict[str, List[str]] = {}
    for tag, q in QUERIES.items():
        rows = wd_query(q)
        qids = [r["item"]["value"].split("/")[-1] for r in rows]
        if args.limit: qids = qids[:args.limit]
        all_qids[tag] = qids
        print(f"[info] {tag}: {len(qids)} entities")

    # Resolve to enwiki titles
    qid_to_title = {}
    for tag, qids in all_qids.items():
        qid_to_title.update(get_titles_for_entities(qids))
    titles = sorted(set(qid_to_title.values()))
    print(f"[info] Titles with enwiki: {len(titles)}")

    # Fetch wikitext & build rows
    info = fetch_wikitext(titles)
    rows = []
    for title, meta in info.items():
        rows += page_to_rows(title, meta["pageid"], meta["timestamp"], meta["wikitext"], "enwiki_travel")

    if not rows:
        print("[error] no rows built", file=sys.stderr); sys.exit(1)

    # Split by title
    train, val = [], []
    for r in rows:
        h = shasum(r["title"]) % 10000
        (val if (h/10000.0) < args.val_ratio else train).append(r)

    dset = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val) if val else Dataset.from_list(train[:0])
    })
    print(f"[info] train={len(train)} val={len(val)}")
    dset.push_to_hub(args.repo_id, private=args.private)
    print(f"[done] https://huggingface.co/datasets/{args.repo_id} (private)")
if __name__ == "__main__":
    main()
