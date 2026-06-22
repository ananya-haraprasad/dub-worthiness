"""Translation via Sarvam's Mayura model — the same stack the rest of the app uses.

Everything that needs machine translation goes through here: the back-translation
meaning check, the sample-dub text, and the localization comparison. We use Sarvam
(not a generic third-party engine) on purpose: the whole pipeline then runs on one
Indian-language stack (STT -> Translate -> TTS, all Sarvam), and Mayura handles
colloquial Hindi/Tamil that generic MT transliterates instead of translating
(e.g. Tamil புரிஞ்சவங்க comes back as real English, not "Purinjavas").
"""
from __future__ import annotations

import time

import requests

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
TRANSLATE_MODEL = "mayura:v1"
MAX_INPUT_CHARS = 1000          # mayura:v1 input cap; chunks above are trimmed
RETRIES = 2

# Display name -> Sarvam language code.
SARVAM_LANG_CODES = {"English": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN"}


def translate(text: str, source_lang: str, target_lang: str, api_key: str,
              sleep: float = 0.0) -> str | None:
    """Translate `text` from source_lang to target_lang (display names) with Sarvam.

    Returns the translated string, "" when there's nothing to do (empty input or
    same language), or None on a hard failure so callers can skip that chunk
    rather than crash.
    """
    if not text or not text.strip():
        return ""
    if not api_key:
        return None
    src = SARVAM_LANG_CODES.get(source_lang, "auto")
    tgt = SARVAM_LANG_CODES.get(target_lang)
    if not tgt or src == tgt:
        return text
    payload = {
        "input": text[:MAX_INPUT_CHARS],
        "source_language_code": src,
        "target_language_code": tgt,
        "model": TRANSLATE_MODEL,
    }
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.post(SARVAM_TRANSLATE_URL, headers=headers,
                                 json=payload, timeout=30)
            if resp.status_code == 200:
                out = (resp.json() or {}).get("translated_text", "")
                if sleep:
                    time.sleep(sleep)
                return out or ""
            if resp.status_code in (401, 403):
                return None  # auth / credit issue — don't keep hammering
        except requests.exceptions.RequestException:
            pass
        if attempt < RETRIES:
            time.sleep(0.5 * (attempt + 1))
    return None
