#!/usr/bin/env python3
import os, json, uuid, random, argparse
from datasets import load_dataset
from pathlib import Path

RANDOM_SEED = 7
random.seed(RANDOM_SEED)

# Your private corpora (EU + India)
WV = "aveekmukherjee/wikivoyage-eu-india-sections"
WP = "aveekmukherjee/wikipedia-travel-eu-india"

SYSTEM_PROMPT = "You are a concise, factual travel assistant. Cite with [1], [2] when using sources."

ASK_TEMPLATES = [
  "I have one day in {title}. Any quick plan?",
  "Visiting {title} in {month}—top 2–3 must-see spots?",
  "What’s the best time of year to visit {title}?",
  "I’ll be in {title} with kids—any quick tips?",
  "Two neighborhoods to stay in {title}?"
]

CLARIFY_TEMPLATES = [
  "Do you mean {title} in {country}? If not, which one?",
  "{title} has multiple districts. Which area do you prefer—historic center or near the station?"
]

GUIDE_TEMPLATES = [
  "• " + "{b1}\n• " + "{b2}\n[1]",
  "{one_liner} [1]"
]

def bulletize(text: str, max_bullets=2):
    # super simple compressor: take first lines/sentences of decent length
    sents = [s.strip() for s in text.split(". ") if len(s) > 40]
    sents = sents[:max_bullets]
    if not sents:
        return ["Check the main sights and prebook popular attractions."]
    # trim
    return [s[:160].rstrip(".") + "." for s in sents]

def one_line(text: str):
    s = " ".join(text.split())
    return (s[:220] + "…") if len(s) > 220 else s

def make_dialog(rec, clarify=False):
    title = rec["title"]
    url = rec["source_url"]
    month = random.choice(["April","May","June","July","August","September","October"])
    user_q = random.choice(ASK_TEMPLATES).format(title=title, month=month)
    bullets = bulletize(rec["text"])
    answer = random.choice(GUIDE_TEMPLATES).format(b1=bullets[0], b2=bullets[1] if len(bullets)>1 else "Explore the old town streets.", one_liner=one_line(rec["text"]))
    dialog = [{"role":"system","content":SYSTEM_PROMPT}]
    if clarify and random.random() < 0.5:
        dialog += [
            {"role":"user","content": user_q},
            {"role":"assistant","content": random.choice(CLARIFY_TEMPLATES).format(title=title, country=("India" if rec.get("lat") and 6<=rec["lat"]<=37.6 else "the EU"))},
            {"role":"user","content":"Historic center is fine."},
            {"role":"assistant","content": answer}
        ]
    else:
        dialog += [
            {"role":"user","content": user_q},
            {"role":"assistant","content": answer}
        ]
    dialog.append({"role":"assistant","content": f"[1] {url}"})
    return {
        "id": str(uuid.uuid4()),
        "dialog": dialog,
        "meta": {"source_url": url, "seed": RANDOM_SEED}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="sft/out")
    ap.add_argument("--n-dialogs", type=int, default=40000)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # Pull both corpora and sample
    wv = list(load_dataset(WV, split="train"))
    wp = list(load_dataset(WP, split="train"))
    pool = [r for r in wv + wp if len(r["text"]) >= 350]
    random.shuffle(pool)

    items = []
    for i, rec in enumerate(pool[:args.n_dialogs]):
        items.append(make_dialog(rec, clarify=(i % 5 == 0)))  # ~20% with a clarifying turn

    # deterministic split by title hash
    def h(title): return (hash(title) & 0xFFFFFFFF) / 2**32
    train = [d for d in items if h(d["dialog"][1]["content"]) >= args.val_ratio]
    val   = [d for d in items if h(d["dialog"][1]["content"]) <  args.val_ratio]

    with open(os.path.join(args.out_dir, "train.jsonl"), "w") as f:
        for d in train: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "validation.jsonl"), "w") as f:
        for d in val: f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(train)} train, {len(val)} val to {args.out_dir}/")

if __name__ == "__main__":
    main()
