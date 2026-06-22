"""Engine 6 — Prosody Dependency (text-based proxy).

How much does the meaning live in *delivery* rather than *words*? Prosody-heavy
content (fast, emphatic, exclamatory, self-interrupting) is harder to dub: lip
sync tightens, and TTS/voice actors must reproduce timing and stress the words
alone don't carry.

We deliberately DO NOT attempt sarcasm/irony detection from text — it produces
false signals. We stick to things actually measurable from a transcript and
label the whole engine a proxy. Real audio-feature prosody (pitch, energy) would
be a future upgrade.

Signals:
  1. Speaking rate (WPM) from word_count / duration -> lip-sync risk.
  2. Emphasis density: discourse intensifiers per 100 words.
  3. Emotion density: exclamations, laugh markers, ALL-CAPS shouting.
  4. Self-interruption: restart/hesitation markers (timing dependency).
"""
from __future__ import annotations

import re

EMPHASIS_MARKERS = [
    "literally", "actually", "seriously", "honestly", "basically", "totally",
    "i mean", "you know", "like", "right", "bro", "yaar", "matlab", "bhai",
    "trust me", "for real", "no cap",
]
INTERRUPTION_MARKERS = [
    "i mean", "wait", "no actually", "okay so", "so basically", "hold on",
    "let me", "the thing is", "but like", "and then like",
]
_LAUGH = re.compile(r"\b(haha+|hehe+|lol|lmao|rofl|lmfao)\b", re.IGNORECASE)


def _per_100(count: int, words: int) -> float:
    return (count / words * 100) if words else 0.0


def _count_markers(text_lower: str, markers: list[str]) -> int:
    total = 0
    for m in markers:
        total += len(re.findall(r"\b" + re.escape(m) + r"\b", text_lower))
    return total


def analyze(transcript: str, duration_seconds: float | None,
            word_count: int | None = None) -> dict:
    words = word_count if word_count is not None else len(transcript.split())
    low = transcript.lower()

    # --- Speaking rate -------------------------------------------------------
    if duration_seconds and duration_seconds > 0:
        wpm = words / (duration_seconds / 60.0)
    else:
        wpm = 0.0
    if wpm == 0:
        lip_sync_risk = "unknown"
    elif wpm < 100:
        lip_sync_risk = "low"
    elif wpm <= 160:
        lip_sync_risk = "medium"
    else:
        lip_sync_risk = "high"

    # --- Densities -----------------------------------------------------------
    emphasis_density = _per_100(_count_markers(low, EMPHASIS_MARKERS), words)
    # Exclamations + laugh markers only. ALL-CAPS is deliberately NOT counted:
    # STT doesn't encode vocal shouting as caps, so caps just catch acronyms
    # (JEE, RCB, UPSC) and would be pure noise on Indian content.
    emotion_count = transcript.count("!") + len(_LAUGH.findall(transcript))
    emotion_density = _per_100(emotion_count, words)
    interruption_density = _per_100(_count_markers(low, INTERRUPTION_MARKERS), words)

    # --- Aggregate into a level (transparent point system) -------------------
    points = 0
    if wpm > 160:
        points += 2
    elif wpm > 130:
        points += 1
    if emphasis_density > 6:
        points += 2
    elif emphasis_density > 3:
        points += 1
    if emotion_density > 4:
        points += 2
    elif emotion_density > 2:
        points += 1
    if interruption_density > 2:
        points += 1

    if points <= 1:
        dependency = "low"
    elif points <= 3:
        dependency = "medium"
    else:
        dependency = "high"

    return {
        "speaking_rate_wpm": round(wpm, 1),
        "lip_sync_risk": lip_sync_risk,
        "emphasis_density": round(emphasis_density, 2),
        "emotion_density": round(emotion_density, 2),
        "interruption_density": round(interruption_density, 2),
        "prosody_dependency": dependency,
        "points": points,
        "note": ("Text-based proxy only — audio features (pitch, energy, pauses) "
                 "are not analysed, and sarcasm is intentionally not inferred."),
    }
