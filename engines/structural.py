"""Engine 2 — Structural Code-Switch Analysis.

Code-mixing *percentage* is a near-useless dub signal: "kal meeting hai" is 100%
mixed yet trivially translatable. What predicts dubbing pain is the *shape* of
the switching:
  * CLEAN  — languages sit in separate sentences/clauses ("Aaj presentation hai.
             And I'm nervous."). A dub can re-voice unit by unit.
  * INTERLEAVED — languages fuse within one breath ("I was like yaar this is so
             not done na"). There's no clean seam to localise along.

Method — two complementary signals at the SENTENCE level (sentences, not tiny
sub-clauses: splitting on every comma/conjunction fragments real speech into
scraps too short to classify):
  1. SCRIPT MIX — a sentence containing a substantial mix of Devanagari AND
     Latin letters is code-mixed within itself. This is script-based, language-
     model-free, and robust on exactly what Sarvam returns (Hindi in Devanagari,
     English in Latin). It's the primary signal.
  2. LANGDETECT MIX — for single-script sentences, langdetect flags two
     languages with non-trivial probability.
  Only sentences with >=8 words are classified (langdetect is unreliable on
  short text); shorter ones are "undetermined" and excluded.

  CAVEAT: fully Romanised Hindi ("kal meeting hai") reads as English to both
  signals and is under-counted. Surfaced in the UI, not hidden.
"""
from __future__ import annotations

import re

import nltk
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # deterministic (interview-reproducible)

MIN_WORDS = 8
CONF_MIX = 0.30        # two langs each above this => mixed
MIX_SHARE = 0.20       # minority-script WORD share to call script-level mixing

_SCRIPTS = {
    "latin": re.compile(r"[A-Za-z]"),
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "tamil": re.compile(r"[஀-௿]"),
}


def _sentences(transcript: str) -> list[str]:
    # NLTK doesn't break on the Devanagari danda; normalise it to a period first.
    text = transcript.replace("।", ". ").replace("॥", ". ")
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def _word_script(word: str) -> str | None:
    """Dominant script of a word, or None if it has no script letters."""
    best, best_n = None, 0
    for name, rx in _SCRIPTS.items():
        n = len(rx.findall(word))
        if n > best_n:
            best, best_n = name, n
    return best


def _script_mixed(seg: str) -> bool:
    """True if two+ scripts co-occur across WORDS, with the second-most-common
    script holding at least MIX_SHARE of the words. Counting by word (not by
    character) avoids biasing toward Indic scripts, which use several code
    points per syllable and would otherwise drown out Latin loanwords."""
    from collections import Counter
    scripts = [s for w in seg.split() if (s := _word_script(w))]
    if not scripts:
        return False
    counts = Counter(scripts)
    if len(counts) < 2:
        return False
    second = counts.most_common(2)[1][1]
    return (second / sum(counts.values())) >= MIX_SHARE


def _langdetect_mixed(seg: str):
    """True/False if classifiable, None if undetermined."""
    try:
        langs = detect_langs(seg)
    except LangDetectException:
        return None
    if not langs:
        return None
    return len([l for l in langs if l.prob >= CONF_MIX]) >= 2


def _classify(seg: str) -> str:
    if len(seg.split()) < MIN_WORDS:
        return "undetermined"
    if _script_mixed(seg):
        return "interleaved"
    ld = _langdetect_mixed(seg)
    if ld is None:
        return "undetermined"
    return "interleaved" if ld else "clean"


def analyze(transcript: str) -> dict:
    segments = _sentences(transcript)
    classified = [(s, _classify(s)) for s in segments]
    qualifying = [(s, k) for s, k in classified if k != "undetermined"]

    total = len(qualifying)
    interleaved = [s for s, k in qualifying if k == "interleaved"]
    n_inter = len(interleaved)

    interleave_ratio = (n_inter / total) if total else 0.0
    clean_ratio = 1 - interleave_ratio if total else 0.0

    if total == 0:
        switch_type = "undetermined"
    elif interleave_ratio < 0.15:
        switch_type = "clean"
    elif interleave_ratio < 0.40:
        switch_type = "mixed"
    else:
        switch_type = "interleaved"

    examples = sorted(interleaved, key=lambda s: len(s.split()), reverse=True)[:3]

    return {
        "interleave_ratio": round(interleave_ratio, 3),
        "clean_switch_ratio": round(clean_ratio, 3),
        "switch_type": switch_type,
        "qualifying_clauses": total,
        "interleaved_clauses": n_inter,
        "examples": examples,
        "caveat": ("Sentence-level analysis on clauses of 8+ words. Detects "
                   "within-sentence Devanagari+Latin mixing and multi-language "
                   "sentences. Fully Romanised Hindi can read as English and be "
                   "under-counted — a structural indicator, not a census."),
    }
