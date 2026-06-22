"""Sample-dub generation — closes the loop with Sarvam's own TTS.

This is NOT a full dubbing product (that would mean re-timing and re-muxing the
whole video — off-thesis and credit-heavy). It produces a short, tangible
*sample*: the opening of the clip, translated into each target language and
voiced with Sarvam TTS. The point is to make the score concrete — and to expose
its honest tie-in: a prosody-heavy clip sounds flat when voiced literally, which
is exactly the risk the score flags.

Pipeline: source transcript → excerpt → translate (Sarvam Mayura) → Sarvam TTS.
The dub *text* and the voice are both Sarvam, so the whole sample runs on one
stack. Audio is synthesised on demand (button press) so we don't spend TTS
credits on every analysis.
"""
from __future__ import annotations

import base64

import requests

from engines.translate import translate as sarvam_translate

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
TTS_MODEL = "bulbul:v3"
# Match the dub voice to the source speaker's gender (both verified on v3).
TTS_SPEAKERS = {"male": "aditya", "female": "priya", "unknown": "priya"}
TTS_LANG_CODE = {"English": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN"}
EXCERPT_WORDS = 55       # ~ first 15-20s of speech
MAX_TTS_CHARS = 900      # well under bulbul:v3's 2500 cap; keeps samples cheap


class DubError(Exception):
    """User-facing dub/TTS failure."""


def excerpt(transcript: str, n: int = EXCERPT_WORDS) -> str:
    return " ".join(transcript.split()[:n]).strip()


def translate_excerpt(text: str, source_lang: str, target_lang: str,
                      api_key: str) -> str:
    if source_lang == target_lang:
        return text
    out = sarvam_translate(text[:MAX_TTS_CHARS], source_lang, target_lang, api_key)
    if out is None:
        raise DubError("Sarvam translation failed (check the key / credits).")
    return out


def build_excerpts(transcript: str, source_lang: str, targets: list[str],
                   api_key: str) -> dict:
    """Cheap, instant: source excerpt + a Sarvam-translated excerpt per target."""
    src_ex = excerpt(transcript)
    out = {"source_excerpt": src_ex, "by_language": {}}
    for t in targets:
        try:
            out["by_language"][t] = translate_excerpt(src_ex, source_lang, t, api_key)
        except DubError:
            out["by_language"][t] = ""
    return out


def synthesize(text: str, target_lang: str, api_key: str,
               gender: str = "unknown") -> bytes:
    """Voice the (already translated) text with Sarvam TTS. Returns WAV bytes.
    `gender` ('male'/'female') picks a matching voice."""
    if not text.strip():
        raise DubError("Nothing to synthesise.")
    payload = {
        "text": text[:MAX_TTS_CHARS],
        "target_language_code": TTS_LANG_CODE.get(target_lang, "en-IN"),
        "model": TTS_MODEL,
        "speaker": TTS_SPEAKERS.get(gender, TTS_SPEAKERS["unknown"]),
    }
    try:
        resp = requests.post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json=payload, timeout=90,
        )
        if resp.status_code in (401, 403):
            raise DubError("Sarvam rejected the API key for text-to-speech "
                           "(check the key has TTS access / credits).")
        resp.raise_for_status()
        audios = resp.json().get("audios", [])
        if not audios:
            raise DubError("TTS returned no audio.")
        return base64.b64decode(audios[0])
    except requests.exceptions.RequestException as exc:
        raise DubError(f"TTS request failed: {exc}")
