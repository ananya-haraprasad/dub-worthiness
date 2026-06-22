"""Shared text helpers for dictionary-matching engines (idiomatic, cultural).

Sarvam transcribes Hindi speech in Devanagari, but our idiom/cultural
dictionaries are written in Roman script. To match across that gap we build a
single normalised "search string" that combines:
  * the original transcript (catches Roman / code-mixed English tokens), and
  * a romanised copy of any Devanagari (catches native-Hindi terms).

Matching then normalises both sides the same way (lowercase, strip to a-z,
collapse repeated letters) so vowel-length / gemination differences between
romanisation schemes don't block a match. Recall on English loanwords rendered
in Devanagari is still imperfect — that limitation is surfaced in the UI.
"""
from __future__ import annotations

import re

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    _HAS_TRANSLIT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TRANSLIT = False

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_TAMIL = re.compile(r"[஀-௿]")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text or ""))


def has_tamil(text: str) -> bool:
    return bool(_TAMIL.search(text or ""))


def romanize(text: str) -> str:
    """Best-effort Indic-script -> Roman (ITRANS). Handles Devanagari and Tamil,
    so the (Roman) dictionaries can match Hindi *and* Tamil spoken content.
    Empty string if no Indic script present or the library is unavailable.
    """
    if not _HAS_TRANSLIT:
        return ""
    try:
        if has_devanagari(text):
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        if has_tamil(text):
            return transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
    except Exception:
        return ""
    return ""


def normalize(text: str) -> str:
    """Lowercase, keep only a-z + spaces, collapse repeated letters & spaces."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z\s]", " ", text)      # drop digits, punctuation, scripts
    text = re.sub(r"(.)\1+", r"\1", text)        # aa->a, ll->l (scheme tolerance)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_search_text(transcript: str) -> str:
    """Normalised haystack (original + romanised Devanagari) for phrase lookup."""
    roman = romanize(transcript)
    combined = f"{transcript} {roman}" if roman else transcript
    return normalize(combined)


def phrase_hits(norm_phrase: str, padded_haystack: str) -> int:
    """Count whole-token occurrences of a normalised phrase.

    `padded_haystack` must be the search text wrapped in single spaces, so we
    match on token boundaries (e.g. 'scene' won't match inside 'obscene').
    """
    if not norm_phrase:
        return 0
    needle = f" {norm_phrase} "
    count, start = 0, 0
    while True:
        idx = padded_haystack.find(needle, start)
        if idx == -1:
            return count
        count += 1
        start = idx + 1  # allow overlapping space delimiters
