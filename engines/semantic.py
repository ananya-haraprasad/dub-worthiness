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

import re
from typing import Callable

from engines.translate import SARVAM_LANG_CODES, translate as sarvam_translate

DEFAULT_SLEEP = 0.3        # gap between Sarvam translate calls (be gentle on limits)
MAX_CHUNKS = 16            # per-clip cap on sentence chunks (bounds translate calls)

# A faithful back-translation never returns a perfect match: benign paraphrase
# and register shifts cost a little similarity with zero real meaning loss. So we
# subtract a benign-drift floor and rescale the remainder to 0..1.
#
# The floor is calibrated to the TRANSLATOR'S baseline, and that baseline shifted
# when we moved from a generic engine to Sarvam Mayura. Measured Sarvam round-trip
# losses: clean factual ~0.03-0.04, a poem ~0.07, shayari ~0.15, dense idioms
# ~0.27. So a ~0.05-0.06 floor zeroes clean speech while preserving the real
# spread: idiom/wordplay-heavy content still registers loss and scores lower.
# (Sarvam is consistent across languages, so the per-language floors are close.)
#
# Honest blind spot this exposes: back-translation measures whether MEANING
# survives, not FORM. A poem's meaning round-trips fine (~0.07) even though its
# rhyme and rhythm are destroyed — so it reads "clean" here. Form/prosody loss is
# documented in the methodology, not captured by this signal.
BENIGN_LOSS_FLOOR = {"English": 0.05, "Hindi": 0.06, "Tamil": 0.06}
DEFAULT_LOSS_FLOOR = 0.06

ProgressCb = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


def _calibrated_loss(raw_loss: float, floor: float) -> float:
    """Subtract the source-language benign-drift floor, then rescale to 0..1."""
    return max(0.0, min(1.0, (raw_loss - floor) / (1 - floor)))


def _chunk_text(text: str) -> list[str]:
    """Split into SENTENCES, not fixed word-blocks, so the round-trip can flag the
    specific lines that break (a garbled sentence stands out instead of being
    averaged away across 150 words). Very short fragments glue onto the previous
    line. Capped at MAX_CHUNKS to bound translate calls / cost on long clips."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?।॥…])\s+", text.strip()) if s.strip()]
    merged: list[str] = []
    for s in parts:
        if merged and (len(s.split()) < 4 or len(merged[-1].split()) < 4):
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)
    return (merged or [text.strip()])[:MAX_CHUNKS]


def analyze(transcript: str, model, source_lang: str, targets: list[str],
            api_key: str = "", sleep: float = DEFAULT_SLEEP,
            progress_cb: ProgressCb = _noop) -> dict:
    """Round-trip each chunk source→target→source (via Sarvam) for each target;
    score how much meaning survives.

    `model` is a loaded SentenceTransformer (passed in so this stays pure and
    Streamlit-agnostic; the app caches it with @st.cache_resource).
    """
    from sentence_transformers import util

    target_items = [t for t in targets if t in SARVAM_LANG_CODES]

    chunks = [c for c in _chunk_text(transcript) if c.strip()]
    if not chunks:
        return {"by_language": {}, "worst_chunks": [], "translations": {},
                "note": "Transcript was empty; no semantic analysis run."}

    # Encode originals (in the source language) once; reused across targets.
    orig_emb = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)

    floor = BENIGN_LOSS_FLOOR.get(source_lang, DEFAULT_LOSS_FLOOR)

    by_language: dict[str, dict] = {}
    all_chunk_records: list[dict] = []
    translations: dict[str, str] = {}    # full forward translation per target
    total_steps = max(1, len(target_items) * len(chunks))
    step = 0

    for lang_name in target_items:
        sims: list[float] = []
        losses: list[float] = []   # calibrated (benign drift removed)
        fwd_texts: list[str] = []
        for ci, chunk in enumerate(chunks):
            step += 1
            progress_cb(
                0.55 + 0.25 * (step / total_steps),
                f"Back-translation check: {source_lang}→{lang_name} ({ci + 1}/{len(chunks)})",
            )

            fwd = sarvam_translate(chunk, source_lang, lang_name, api_key, sleep)
            if fwd:
                fwd_texts.append(fwd)
            back = sarvam_translate(fwd, lang_name, source_lang, api_key, sleep) if fwd else None
            if not back:
                continue  # skip chunks where MT failed rather than crash

            back_emb = model.encode(back, convert_to_tensor=True,
                                    normalize_embeddings=True)
            sim = max(0.0, min(1.0, float(util.cos_sim(orig_emb[ci], back_emb).item())))
            cal = _calibrated_loss(1 - sim, floor)
            sims.append(sim)
            losses.append(cal)
            all_chunk_records.append({
                "language": lang_name,
                "similarity": round(sim, 3),
                "loss": round(cal, 3),          # calibrated meaning loss
                "raw_loss": round(1 - sim, 3),  # before benign-drift floor
                "original": chunk,
                "forward": fwd,                 # what Sarvam translated this line to
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
    for lang_name in target_items:
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
                 f"via Sarvam Mayura. Lower similarity = more meaning lost on the "
                 f"round trip."),
    }
