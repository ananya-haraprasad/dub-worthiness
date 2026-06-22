"""Central language registry + source-language detection for the 3-way model.

The tool supports English, Hindi, and Tamil. The SOURCE language is auto-detected
from the transcript; the TARGETS are the other two languages. This is what lets
the same pipeline score every direction (Eng→Tamil, Tamil→Hindi, Hindi→English…)
rather than assuming English is always the source.
"""
from __future__ import annotations

# Display name -> translation/lang code (deep-translator + langdetect compatible)
LANG_CODES = {"English": "en", "Hindi": "hi", "Tamil": "ta"}
CODE_TO_NAME = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
ALL_LANGS = list(LANG_CODES.keys())


def detect_source(sarvam_language_code: str | None, transcript: str = "") -> str:
    """Map Sarvam's language_code (e.g. 'ta-IN') to a supported language NAME.

    Falls back to langdetect on the transcript, then to English.
    """
    code = (sarvam_language_code or "").split("-")[0].lower()
    if code in CODE_TO_NAME:
        return CODE_TO_NAME[code]
    try:
        from langdetect import detect
        d = detect(transcript) if transcript.strip() else ""
        if d in CODE_TO_NAME:
            return CODE_TO_NAME[d]
    except Exception:
        pass
    return "English"


def targets_for(source_name: str) -> list[str]:
    """The two languages to score dubbing INTO, given the source."""
    return [name for name in ALL_LANGS if name != source_name]
