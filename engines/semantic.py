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

# A faithful back-translation never returns a perfect match. Two benign reasons:
#   1. Paraphrase — "I went to the store" round-trips as "I visited the shop".
#   2. Code-mixing — a Hinglish speaker says "इंपॉर्टेंट ऑप्शन चूज करना", and the
#      round trip normalizes the English loanwords to formal Hindi
#      ("महत्वपूर्ण विकल्प चुनना"). Same meaning, different surface form.
# The embedding reads both as lower similarity. A measured clean Hinglish
# monologue round-trips at ~0.72 similarity (0.28 raw "loss") with ZERO real
# meaning loss; a colloquial street-vendor clip, whose slang genuinely doesn't
# survive, lands far lower (~0.58, 0.42 loss). So we subtract a benign-drift
# floor and rescale the remainder to 0..1. The floor sits a little BELOW the
# clean baseline on purpose: a faithful translation then reads as near-clean
# (small penalty, still "travels cleanly"), while a clip with real loss is still
# clearly penalised. Tuned so those two measured clips land on opposite sides.
BENIGN_LOSS_FLOOR = 0.20

ProgressCb = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


def _calibrated_loss(raw_loss: float) -> float:
    """Subtract the benign round-trip floor, then rescale to 0..1."""
    return max(0.0, min(1.0, (raw_loss - BENIGN_LOSS_FLOOR) / (1 - BENIGN_LOSS_FLOOR)))


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
        return {"by_language": {}, "worst_chunks": [], "translations": {},
                "note": "Transcript was empty; no semantic analysis run."}

    # Encode originals (in the source language) once; reused across targets.
    orig_emb = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)

    by_language: dict[str, dict] = {}
    all_chunk_records: list[dict] = []
    translations: dict[str, str] = {}    # full forward translation per target
    total_steps = max(1, len(target_items) * len(chunks))
    step = 0

    for lang_name, lang_code in target_items:
        sims: list[float] = []
        losses: list[float] = []   # calibrated (benign drift removed)
        fwd_texts: list[str] = []
        for ci, chunk in enumerate(chunks):
            step += 1
            progress_cb(
                0.55 + 0.25 * (step / total_steps),
                f"Back-translation check: {source_lang}→{lang_name} ({ci + 1}/{len(chunks)})",
            )

            fwd = _translate(chunk, src_code, lang_code, sleep)
            if fwd:
                fwd_texts.append(fwd)
            back = _translate(fwd, lang_code, src_code, sleep) if fwd else None
            if not back:
                continue  # skip chunks where MT failed rather than crash

            back_emb = model.encode(back, convert_to_tensor=True,
                                    normalize_embeddings=True)
            sim = max(0.0, min(1.0, float(util.cos_sim(orig_emb[ci], back_emb).item())))
            cal = _calibrated_loss(1 - sim)
            sims.append(sim)
            losses.append(cal)
            all_chunk_records.append({
                "language": lang_name,
                "similarity": round(sim, 3),
                "loss": round(cal, 3),          # calibrated meaning loss
                "raw_loss": round(1 - sim, 3),  # before benign-drift floor
                "original": chunk,
                "back_translated": back,
            })

        if sims:
            avg_loss = sum(losses) / len(losses)
            by_language[lang_name] = {
                "similarity": round(sum(sims) / len(sims), 3),
                "loss": round(avg_loss, 3),
                "max_loss": round(max(losses), 3),  # worst single chunk
                "chunks_scored": len(sims),
            }
        else:
            by_language[lang_name] = {
                "similarity": None, "loss": None, "chunks_scored": 0,
                "note": "Translation unavailable (rate-limited or offline).",
            }
        translations[lang_name] = " ".join(fwd_texts)

    # The genuinely worst chunk per target language (only if meaning really
    # dropped). Avoids surfacing a high-similarity chunk as "where meaning slipped
    # most", which the old cross-language de-dup could do.
    worst = []
    for lang_name, _code in target_items:
        recs = [r for r in all_chunk_records if r["language"] == lang_name]
        if not recs:
            continue
        w = max(recs, key=lambda r: r["loss"])
        if w["loss"] >= 0.20:
            worst.append(w)
    worst.sort(key=lambda r: r["loss"], reverse=True)

    return {
        "source_language": source_lang,
        "by_language": by_language,
        "worst_chunks": worst,
        "translations": translations,
        "chunks_analyzed": len(chunks),
        "note": (f"Back-translation round trip ({source_lang}→target→{source_lang}) "
                 f"via Google Translate (free). Lower similarity = more meaning "
                 f"lost on the round trip."),
    }
