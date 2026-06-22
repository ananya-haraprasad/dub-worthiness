"""Engine — Localization gap (the "show, don't just tell" view).

For English terms in the source that machine translation routinely TRANSLITERATES
(writes in the target script instead of finding a real word), this contrasts:
  * what free MT produces RIGHT NOW (fetched live — provably the current output),
  * against a curated NATURAL equivalent a human localizer would use.

It's deliberately nuanced: where the English loanword genuinely *is* the everyday
word in the target (computer, online, download), it's marked "fine to keep" — so
this isn't a naive "all English is bad" view. The point is to make the gap
between *translated* and *localized* tangible.

Mainly fires for English-source content (the terms are English); for native-
script sources there's usually nothing to flag, and the section stays quiet.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from functools import lru_cache

from engines.langs import LANG_CODES

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "loanword_glossary.json"
)
MAX_TERMS = 6            # cap live MT calls / keep the table scannable
INTER_CALL_SLEEP = 0.3


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    for e in data:
        term = (e.get("term") or "").strip()
        e["_pat"] = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) if term else None
    return [e for e in data if e.get("_pat")]


def _mt(term: str, target_code: str) -> str:
    from deep_translator import GoogleTranslator
    try:
        out = GoogleTranslator(source="en", target=target_code).translate(term)
        time.sleep(INTER_CALL_SLEEP)
        return out or ""
    except Exception:
        return ""


MATCH_THRESHOLD = 0.6
# How a verdict reads when MT does NOT match the recommended call.
_WRONG_MSG = {
    "localize": "transliterated — should use the native word",
    "keep_english": "over-translated — should stay English",
    "loanword_ok": "over-translated — the loanword is fine",
}
_RIGHT_MSG = {
    "localize": "localized correctly",
    "keep_english": "kept in English (correct)",
    "loanword_ok": "kept the loanword (correct)",
}


def _matches(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= MATCH_THRESHOLD


def analyze(transcript: str, targets: list[str]) -> dict:
    glossary = _load()
    all_found = [e for e in glossary if e["_pat"].search(transcript)]
    # Surface the instructive calls (localize / keep_english) before plain loanwords.
    order = {"localize": 0, "keep_english": 1, "loanword_ok": 2}
    all_found.sort(key=lambda e: order.get(e.get("recommendation"), 3))
    # Full burden (every English term the dub leans on), independent of the
    # display cap — the score uses this for anglicization density.
    all_hits = [{"term": e["term"], "recommendation": e.get("recommendation", "localize")}
                for e in all_found]
    found = all_found[:MAX_TERMS]   # only these get a live MT call, for the table

    by_language: dict[str, list] = {}
    for lang in targets:
        if lang == "English":
            by_language[lang] = []   # dubbing INTO English: nothing to localize
            continue
        code = LANG_CODES.get(lang, "")
        rows = []
        for e in found:
            rec = e.get("recommendation", "localize")
            recommended = (e.get(lang.lower()) or "").strip()
            mt = _mt(e["term"], code)
            correct = _matches(mt, recommended)
            rows.append({
                "term": e["term"],
                "recommendation": rec,            # localize | keep_english | loanword_ok
                "recommended": recommended,       # the form a good dub should use
                "mt_current": mt,                 # what free MT actually produced (live)
                "correct": correct,               # did MT make the right call?
                "verdict": (_RIGHT_MSG if correct else _WRONG_MSG).get(rec, ""),
                "why": e.get("why", ""),
            })
        by_language[lang] = rows

    return {
        "by_language": by_language,
        "total_found": len(found),
        "all_matches_count": len(all_found),   # full burden, before the display cap
        "all_hits": all_hits,
        "dictionary_size": len(glossary),
        "note": ("Common English terms in the source. 'Free MT' is the live machine "
                 "output; 'Right call' is what a good dub should do — localize to a "
                 "native word, keep a fixed term in English, or keep a naturalised "
                 "loanword. The verdict shows where MT got it right vs wrong."),
    }
