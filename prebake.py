"""Pre-bake the sample clips' analysis so the DEPLOYED demo never depends on a
live YouTube download (datacenter IPs get 403'd) and never burns credits when a
visitor clicks an example.

Run this LOCALLY, where YouTube downloads work and your Sarvam key is in
.streamlit/secrets.toml:

    python prebake.py

It writes one JSON per example into data/prebaked/<videoid>.json. Commit those
files; the app loads them instantly for the matching example URL. Re-run any time
the sample list or scoring changes.
"""
import json
import os
import re

from sentence_transformers import SentenceTransformer

from engines import (cultural, dubber, extractor, idiomatic, langs,
                     localization, opportunity, prosody, semantic, structural)
import scorer

# The example URLs (must match SAMPLES in app.py).
SAMPLE_URLS = [
    "https://youtube.com/shorts/q9wKARQ8_pg",   # English · Barbie monologue
    "https://youtube.com/shorts/obRs6VrF9FE",   # English · Skincare tips
    "https://youtube.com/shorts/LPI9mLkIIS8",   # Hindi · Antarctica vlog
    "https://youtube.com/shorts/1CBtqHEDUXI",   # Tamil · Assembly speech
    "https://youtube.com/shorts/mRFs19QPAwI",   # Tamil · Shirt seller
    "https://youtube.com/shorts/L5mWKH07TBQ",   # Hindi · Street-food vlog
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "prebaked")


def video_id(url: str) -> str:
    m = re.search(r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-16:]


def _key() -> str:
    path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    for line in open(path, encoding="utf-8"):
        if line.strip().startswith("SARVAM_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def analyze_url(url: str, key: str, model) -> dict:
    """Faithful copy of app.run_analysis, minus the Streamlit session caching."""
    result = extractor.extract_and_transcribe(key, youtube_url=url)
    tx = result.to_dict()
    transcript = tx["transcript"]
    src = langs.detect_source(tx.get("language_code"), transcript)
    tg = langs.targets_for(src)
    res = {"transcript": tx, "source_lang": src, "targets": tg}
    res["structural"] = structural.analyze(transcript)
    res["idiomatic"] = idiomatic.analyze(transcript)
    res["cultural"] = cultural.analyze(transcript, tg)
    res["prosody"] = prosody.analyze(transcript, tx.get("duration_seconds"),
                                     tx.get("word_count"))
    res["opportunity"] = opportunity.analyze(transcript, tg)
    res["semantic"] = semantic.analyze(transcript, model, src, tg, key)
    res["dub"] = dubber.build_excerpts(transcript, src, tg, key)
    res["localization"] = localization.analyze(transcript, tg, key)
    res["scores"] = scorer.compute_scores(res, tg)
    res["_prebaked"] = True
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    key = _key()
    if not key:
        print("No SARVAM_API_KEY in .streamlit/secrets.toml — aborting.")
        return
    print("Loading embedding model…")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    ok, fail = 0, 0
    for url in SAMPLE_URLS:
        vid = video_id(url)
        try:
            res = analyze_url(url, key, model)
            with open(os.path.join(OUT_DIR, f"{vid}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False)
            best = res["scores"]["quality_priority_order"][0]
            score = res["scores"]["by_language"][best]["dub_quality_score"]
            print(f"  OK   {vid}  {res['source_lang']} -> best {score}")
            ok += 1
        except Exception as exc:
            print(f"  FAIL {vid}: {type(exc).__name__}: {str(exc)[:70]}")
            fail += 1
    print(f"\nDone. {ok} baked, {fail} failed. Commit data/prebaked/*.json")


if __name__ == "__main__":
    main()
