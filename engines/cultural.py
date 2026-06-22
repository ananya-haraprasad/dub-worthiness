"""Engine 5 — Cultural Reference Density.

References that need specific cultural knowledge to land. A Tamil audience may
not feel JEE/Kota anxiety or an RCB meme; a Hindi audience may not follow a
Kollywood "Thalapathy" reference or TNPSC. These don't fail at the word level —
they fail at the *knowledge* level, so translation alone won't save them.

We match the transcript against a curated reference base
(data/cultural_references.json), each tagged with per-audience familiarity
(high/medium/low for Hindi vs Tamil), then turn the asymmetry into per-language
risk. The point isn't a single number — it's showing *which* audience loses
*which* references.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from engines import textutils

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "cultural_references.json"
)
_FAM_WEIGHT = {"low": 1.0, "medium": 0.5, "high": 0.0}

# Honorifics / address terms are NOT distinctive enough to be match keys: "bhai"
# would map "Bhai scene kya hai" onto Salman Khan. Drop them from any reference's
# surface forms (a ref whose ONLY form is one of these is simply unmatchable).
_AMBIGUOUS_KEYS = {textutils.normalize(w) for w in {
    "bhai", "bhaiya", "bhaiyya", "sir", "madam", "boss", "amma", "appa", "anna",
    "akka", "da", "di", "mama", "machi", "machan", "guru", "beta", "didi",
    "paaji", "bro", "yaar", "dada", "boudi", "kanna", "thala", "anbu",
}}



def _is_acronym(s: str) -> bool:
    """ALL-CAPS short token like JEE, RCB, UP — matched case-sensitively so it
    can't fire on the lowercase common word (e.g. 'UP' vs 'listen up')."""
    s = s.strip()
    return s.isupper() and s.isalpha() and 2 <= len(s) <= 6


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    for e in data:
        surfaces = [e.get("reference", "")] + list(e.get("aliases", []) or [])
        # Acronyms matched case-sensitively; everything else via normalised text.
        e["_acronyms"] = sorted({s.strip() for s in surfaces if _is_acronym(s)},
                                key=len, reverse=True)
        norms = {textutils.normalize(s) for s in surfaces if not _is_acronym(s)}
        e["_norms"] = sorted(
            {n for n in norms if len(n) >= 3 and n not in _AMBIGUOUS_KEYS},
            key=len, reverse=True,
        )
    return [e for e in data if e["_norms"] or e["_acronyms"]]


def _risk_level(weighted_density: float) -> str:
    if weighted_density < 0.4:
        return "low"
    if weighted_density <= 1.2:
        return "medium"
    return "high"


def analyze(transcript: str, languages: list[str] | None = None) -> dict:
    languages = languages or ["Hindi", "Tamil"]
    words = max(1, len(transcript.split()))
    refs = _load()
    haystack = f" {textutils.build_search_text(transcript)} "

    found: list[dict] = []
    for e in refs:
        hits = sum(textutils.phrase_hits(n, haystack) for n in e["_norms"])
        # Acronyms: case-sensitive, word-bounded, against the ORIGINAL transcript.
        for ac in e["_acronyms"]:
            hits += len(re.findall(rf"(?<![A-Za-z]){re.escape(ac)}(?![A-Za-z])",
                                   transcript))
        if hits == 0:
            continue
        fam = {l: (e.get("familiarity", {}) or {}).get(l, "medium") for l in languages}
        found.append({
            "reference": e.get("reference", ""),
            "category": e.get("category", ""),
            "familiarity": fam,
            "count": hits,
            "note": e.get("note", ""),
        })

    # --- Group by audience asymmetry across the chosen target languages ------
    groups: dict[str, list[str]] = {
        "familiar_to_all": [], "audience_skewed": [], "niche_everywhere": [],
    }
    for f in found:
        fams = [f["familiarity"].get(l, "medium") for l in languages]
        if all(x == "high" for x in fams):
            groups["familiar_to_all"].append(f["reference"])
        elif any(x == "high" for x in fams):
            groups["audience_skewed"].append(f["reference"])
        else:
            groups["niche_everywhere"].append(f["reference"])

    # --- Per-language risk ---------------------------------------------------
    risk_by_language = {}
    for lang in languages:
        weighted = sum(_FAM_WEIGHT.get(f["familiarity"].get(lang, "medium"), 0.5)
                       for f in found)
        density = weighted / words * 100
        risk_by_language[lang] = _risk_level(density)

    # --- Top risky references (lowest combined familiarity) ------------------
    def _risk_key(f):
        return sum(_FAM_WEIGHT.get(f["familiarity"].get(l, "medium"), 0.5)
                   for l in languages)

    top_risky = [f["reference"] for f in
                 sorted(found, key=lambda f: (-_risk_key(f), -f["count"]))
                 if _risk_key(f) > 0][:5]

    return {
        "reference_density": round(len(found) / words * 100, 2),
        "total_references": len(found),
        "references_found": groups,
        "found_detail": found,
        "risk_by_language": risk_by_language,
        "top_risky_references": top_risky,
        "dictionary_size": len(refs),
        "note": ("Familiarity ratings are directional, per-audience estimates "
                 "(Hindi = North-Indian Hindi speaker; Tamil = Tamil Nadu)."),
    }
