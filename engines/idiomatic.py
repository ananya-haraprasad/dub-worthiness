"""Engine 4 — Idiomatic & Slang Density.

Idioms are phrases where surface meaning != actual meaning, so direct
translation produces nonsense; they need *adaptation*, not translation. This is
distinct from code-mixing: "kal meeting hai" is mixed but clean and translates
fine; "scene kya hai" is mixed AND opaque. We match the transcript against a
curated Hinglish/Indian-English idiom dictionary (data/hinglish_idioms.json) and
weight each hit by how untranslatable it is.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from engines import textutils

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "hinglish_idioms.json"
)

_RISK_LABEL = {
    0: "requires full adaptation",
    1: "needs heavy adaptation",
    2: "needs light adaptation",
    3: "translates literally",
}

# Single-token entries that are also ultra-common plain-English words: matching
# them fires constantly on ordinary English and inflates density. We keep them
# in the dictionary but skip them when matching. Stored NORMALISED, because
# textutils.normalize collapses doubled letters (full->ful, cool->col).
_GENERIC_SKIP = {textutils.normalize(w) for w in
                 {"like", "right", "full", "total", "fine", "very", "really",
                  "so", "cool", "okay", "actually", "literally"}}


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    # Precompute the normalised phrase once per entry.
    for e in data:
        e["_norm"] = textutils.normalize(e.get("phrase", ""))
    return [e for e in data if e["_norm"]]


def analyze(transcript: str) -> dict:
    words = max(1, len(transcript.split()))
    idioms = _load()
    # Match against the ORIGINAL transcript only (not romanised Devanagari):
    # romanisation barely helps idioms (spelling drift) but creates false hits
    # like और->"aura". This catches Roman/code-mixed Latin slang reliably and
    # under-counts Devanagari-rendered idioms (surfaced in the note).
    haystack = f" {textutils.normalize(transcript)} "

    found: list[dict] = []
    total_hits = 0
    for e in idioms:
        if e["_norm"] in _GENERIC_SKIP:
            continue
        hits = textutils.phrase_hits(e["_norm"], haystack)
        if hits == 0:
            continue
        total_hits += hits
        score = int(e.get("translatability_score", 0))
        found.append({
            "phrase": e.get("phrase", ""),
            "actual_meaning": e.get("actual_meaning", ""),
            "literal": e.get("literal", ""),
            "category": e.get("category", ""),
            "translatability_score": score,
            "risk": _RISK_LABEL.get(score, "needs adaptation"),
            "count": hits,
            "note": e.get("note", ""),
        })

    density = total_hits / words * 100
    if density <= 3:
        level = "low"
    elif density <= 10:
        level = "medium"
    else:
        level = "high"

    # Surface worst offenders first: least translatable, then most frequent.
    found.sort(key=lambda f: (f["translatability_score"], -f["count"]))
    adaptation_required = any(f["translatability_score"] == 0 for f in found)

    return {
        "idiom_density": round(density, 2),
        "density_level": level,
        "total_hits": total_hits,
        "unique_idioms": len(found),
        "found_idioms": found,
        "adaptation_required": adaptation_required,
        "dictionary_size": len(idioms),
        "note": ("Matched against a curated Hinglish idiom dictionary, on the "
                 "transcript as transcribed. Recall is strongest on Roman/"
                 "code-mixed tokens; idioms rendered in Devanagari are "
                 "under-counted."),
    }
