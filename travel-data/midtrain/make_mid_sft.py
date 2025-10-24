#!/usr/bin/env python3
# make_mid_sft.py
import os, json, uuid, math, random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from datasets import load_dataset
from pathlib import Path

# ---------- config ----------
RANDOM_SEED = 7
MID_OUT = Path("out"); SFT_OUT = Path("../sft/out")
MID_OUT.mkdir(parents=True, exist_ok=True); SFT_OUT.mkdir(parents=True, exist_ok=True)

# Point to your private datasets (EU + India)
DS_WV = "aveekmukherjee/wikivoyage-eu-india-sections"
DS_WP = "aveekmukherjee/wikipedia-travel-eu-india"

# Sizes (adjust as needed for Tiny model)
N_TZ = 25000
N_DATE = 25000
N_DIST = 15000
N_COMPRESS = 60000
N_QA = 20000
N_DIALOG = 30000

random.seed(RANDOM_SEED)

# ---------- helpers ----------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return round(R*2*math.asin(math.sqrt(a)))

EU_TZS = ["Europe/Paris","Europe/Rome","Europe/Berlin","Europe/Madrid","Europe/Amsterdam",
          "Europe/Prague","Europe/Lisbon","Europe/Athens","Europe/Stockholm","Europe/Warsaw"]
IN_TZS = ["Asia/Kolkata"]
MIX_TZS = EU_TZS + IN_TZS + ["Europe/London","Europe/Dublin","Europe/Helsinki"]

def sample_tz_pair():
    a = random.choice(MIX_TZS); b = random.choice(MIX_TZS)
    while b == a: b = random.choice(MIX_TZS)
    # sample a date around DST edges too
    base = datetime(2025, random.randint(1,12), random.randint(1,28), random.randint(0,23), random.choice([0,15,30,45]))
    return a, b, base

def tz_math_item():
    a, b, dt = sample_tz_pair()
    dt_a = dt.replace(tzinfo=ZoneInfo(a))
    dt_utc = dt_a.astimezone(ZoneInfo("UTC"))
    dt_b = dt_a.astimezone(ZoneInfo(b)).replace(tzinfo=None)
    return {
      "task":"tz_math",
      "input":{"from_tz":a,"local_time":dt.strftime("%Y-%m-%dT%H:%M:%S"),"to_tz":b},
      "target":{"from_utc":dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to_local":dt_b.strftime("%Y-%m-%dT%H:%M:%S")}
    }

def date_range_item():
    start = datetime(2025, random.randint(1,12), random.randint(1,25))
    nights = random.choice([2,3,4,5,7])
    out = start + timedelta(days=nights)
    return {
      "task":"date_range",
      "input":{"check_in": start.date().isoformat(), "nights": nights},
      "target":{"check_out": out.date().isoformat(),
                "weekday_in": start.strftime("%A"),
                "weekday_out": out.strftime("%A")}
    }

def load_sections():
    wv = load_dataset(DS_WV, split="train")
    wp = load_dataset(DS_WP, split="train")
    # keep items with decent length
    pool = [r for r in list(wv) + list(wp) if len(r["text"]) >= 350]
    random.shuffle(pool)
    return pool

def distance_item(a, b):
    # expects dicts with lat/lon
    km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    bucket = "0-5km" if km<=5 else "5-20km" if km<=20 else "20-100km" if km<=100 else "100-200km" if km<=200 else "200-400km" if km<=400 else "400km+"
    return {
      "task":"distance_km",
      "input":{"a":{"name":a["title"],"lat":a["lat"],"lon":a["lon"]},
               "b":{"name":b["title"],"lat":b["lat"],"lon":b["lon"]}},
      "target":{"km_rounded": km, "bucket": bucket}
    }

def bulletize_target(text):
    # very simple compressor: take first ~3 sentences/lines as bullets.
    sents = [s.strip() for s in text.split(". ") if len(s)>40][:3]
    return [s[:160].rstrip(".") + "." for s in sents]

def make_dialog_from_section(sec):
    # two-turn minimal, grounded by the section/city
    city = sec["title"]
    url = sec["source_url"]
    prompt = random.choice([
        f"I have one day in {city}. What’s a simple plan?",
        f"Visiting {city} in August—any quick tips?",
        f"What are two must-see places in {city}?"
    ])
    answer = "• " + "\n• ".join(bulletize_target(sec["text"])[:2]) + f"\n[1]"
    return {
      "id": str(uuid.uuid4()),
      "dialog": [
        {"role":"system","content":"You are a concise, factual travel assistant. Cite with [1], [2] when using sources."},
        {"role":"user","content": prompt},
        {"role":"assistant","content": answer},
        {"role":"assistant","content": f"[1] {url}"}
      ]
    }

# ---------- build ----------
def main():
    secs = load_sections()

    # 1) TSR
    with open(MID_OUT/"tsr.jsonl","w") as f:
        for _ in range(N_TZ): f.write(json.dumps(tz_math_item())+"\n")
        for _ in range(N_DATE): f.write(json.dumps(date_range_item())+"\n")

        # distances only for items with coords
        geo = [s for s in secs if s.get("lat") is not None and s.get("lon") is not None]
        random.shuffle(geo)
        for i in range(min(N_DIST, len(geo)//2)):
            f.write(json.dumps(distance_item(geo[2*i], geo[2*i+1]))+"\n")

    # 2) Compression
    with open(MID_OUT/"compress.jsonl","w") as f:
        for s in secs[:N_COMPRESS]:
            rec = {"task":"bulletize",
                   "source":{"title":s["title"],"section":s["section_title"],"url":s["source_url"],"text":s["text"]},
                   "target": bulletize_target(s["text"])}
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    # 3) Grounded QA (make a simple wh- question seed)
    def mk_q(s):
        base = s["title"]
        return random.choice([
          f"When is a good time to visit {base}?",
          f"What is one highlight in {base}?",
          f"Any quick safety tip for {base}?"
        ])
    with open(MID_OUT/"qa.jsonl","w") as f:
        for s in secs[:N_QA]:
            ans = " ".join(bulletize_target(s["text"])[:2]) + " [source]"
            rec = {"task":"qa_grounded",
                   "question": mk_q(s),
                   "context":{"title":s["title"],"url":s["source_url"],"text":s["text"]},
                   "answer": ans,
                   "citations":[s["source_url"]]}
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    # 4) Dialog SFT (no tools)
    with open(SFT_OUT/"dialogs.jsonl","w") as f:
        for s in secs[:N_DIALOG]:
            f.write(json.dumps(make_dialog_from_section(s), ensure_ascii=False)+"\n")

    print("Done:",
          MID_OUT/"tsr.jsonl",
          MID_OUT/"compress.jsonl",
          MID_OUT/"qa.jsonl",
          SFT_OUT/"dialogs.jsonl")

if __name__ == "__main__":
    main()
