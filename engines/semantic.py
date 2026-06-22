"""Engine 3 — Semantic Opacity via Back-Translation.

The honest test of dubbability: take a passage in its SOURCE language, translate
it into the target language, translate it straight back to the source, and
measure how much meaning survived (cosine similarity of multilingual sentence
embeddings). Clean, literal speech round-trips with high fidelity. Idioms,
wordplay, and culturally-loaded phrasing collapse — the back-translation drifts,
similarity drops, and that drop is the single most trustworthy signal we have.

3-way aware: the round trip is source→target→source (e.g. Tamil→Hindi→Tamil),
NOT always via English — so it works for any source language. The comparison
happens in the source language; the multilingual model embeds all three.
"""
from __future__ import annotations

import time
from typing import Callable

from engines.langs import LANG_CODES

CHUNK_WORDS = 150          # smaller chunks => more reliable MT
DEFAULT_SLEEP = 0.5        # gap between Google Translate calls (avoid blocks)
TRANSLATE_RETRIES = 2

ProgressCb = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


def _chunk_text(text: str, size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


def _translate(text: str, source: str, target: str, sleep: float):
    """One Google-Translate hop, with a short retry on transient failures."""
    from deep_translator import GoogleTranslator

    for attempt in range(TRANSLATE_RETRIES + 1):
        try:
            out = GoogleTranslator(source=source, target=target).translate(text)
            time.sleep(sleep)
            return out or ""
        except Exception:
            if attempt < TRANSLATE_RETRIES:
                time.sleep(sleep * (attempt + 2))
            else:
                return None
    return None


def analyze(transcript: str, model, source_lang: str, targets: list[str],
            sleep: float = DEFAULT_SLEEP, progress_cb: ProgressCb = _noop) -> dict:
    """Round-trip each chunk source→target→source for each target; score loss.

    `model` is a loaded SentenceTransformer (passed in so this stays pure and
    Streamlit-agnostic; the app caches it with @st.cache_resource).
    """
    from sentence_transformers import util

    src_code = LANG_CODES.get(source_lang, "en")
    target_items = [(t, LANG_CODES[t]) for t in targets if t in LANG_CODES]

    chunks = [c for c in _chunk_text(transcript) if c.strip()]
    if not chunks:
        return {"by_language": {}, "worst_chunks": [],
                "note": "Transcript was empty; no semantic analysis run."}

    # Encode originals (in the source language) once; reused across targets.
    orig_emb = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)

    by_language: dict[str, dict] = {}
    all_chunk_records: list[dict] = []
    total_steps = max(1, len(target_items) * len(chunks))
    step = 0

    for lang_name, lang_code in target_items:
        sims: list[float] = []
        for ci, chunk in enumerate(chunks):
            step += 1
            progress_cb(
                0.55 + 0.25 * (step / total_steps),
                f"Back-translation check: {source_lang}→{lang_name} ({ci + 1}/{len(chunks)})",
            )

            fwd = _translate(chunk, src_code, lang_code, sleep)
            back = _translate(fwd, lang_code, src_code, sleep) if fwd else None
            if not back:
                continue  # skip chunks where MT failed rather than crash

            back_emb = model.encode(back, convert_to_tensor=True,
                                    normalize_embeddings=True)
            sim = max(0.0, min(1.0, float(util.cos_sim(orig_emb[ci], back_emb).item())))
            sims.append(sim)
            all_chunk_records.append({
                "language": lang_name,
                "similarity": round(sim, 3),
                "loss": round(1 - sim, 3),
                "original": chunk,
                "back_translated": back,
            })

        if sims:
            avg = sum(sims) / len(sims)
            by_language[lang_name] = {
                "similarity": round(avg, 3),
                "loss": round(1 - avg, 3),
                "chunks_scored": len(sims),
            }
        else:
            by_language[lang_name] = {
                "similarity": None, "loss": None, "chunks_scored": 0,
                "note": "Translation unavailable (rate-limited or offline).",
            }

    # Worst chunks overall: lowest similarity, de-duplicated by original text.
    seen, worst = set(), []
    for rec in sorted(all_chunk_records, key=lambda r: r["similarity"]):
        key = rec["original"][:80]
        if key in seen:
            continue
        seen.add(key)
        worst.append(rec)
        if len(worst) >= 3:
            break

    return {
        "source_language": source_lang,
        "by_language": by_language,
        "worst_chunks": worst,
        "chunks_analyzed": len(chunks),
        "note": (f"Back-translation round trip ({source_lang}→target→{source_lang}) "
                 f"via Google Translate (free). Lower similarity = more meaning "
                 f"lost on the round trip."),
    }
