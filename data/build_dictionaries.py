"""Assemble the final idiom + cultural dictionaries from the category batches.

Reproducible build: reads data/_build/idioms_*.json and cultural_*.json, then
validates structure, coerces types, drops malformed/empty rows, de-duplicates
(merging aliases for cultural refs), and writes the two canonical files:
    data/hinglish_idioms.json
    data/cultural_references.json

Run from anywhere:  python data/build_dictionaries.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engines import textutils  # noqa: E402

BUILD_DIR = os.path.join(ROOT, "data", "_build")
FAM_VALUES = {"low", "medium", "high"}

# Researchers occasionally invented granular sub-categories; fold them back into
# a clean canonical set so the UI's category breakdown stays tidy.
CATEGORY_FIX = {
    "exam-jargon": "exams",
    "careers": "education",
    "campus-jargon": "education",
    "institutions": "education",
    "institutions-coaching": "education",
    "places-education": "education",
    "media-education": "education",
    "politics-education": "exams",
}


def _fix_category(cat: str) -> str:
    return CATEGORY_FIX.get(cat, cat)


def _clamp_score(v) -> int:
    try:
        return max(0, min(3, int(v)))
    except (TypeError, ValueError):
        return 0


def build_idioms() -> list[dict]:
    out, seen, dropped = [], set(), 0
    for path in sorted(glob.glob(os.path.join(BUILD_DIR, "idioms_*.json"))):
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        for e in batch:
            phrase = (e.get("phrase") or "").strip()
            norm = textutils.normalize(phrase)
            if not norm or norm in seen:
                dropped += 1
                continue
            seen.add(norm)
            out.append({
                "phrase": phrase,
                "literal": (e.get("literal") or "").strip(),
                "actual_meaning": (e.get("actual_meaning") or "").strip(),
                "translatable": bool(e.get("translatable", False)),
                "translatability_score": _clamp_score(e.get("translatability_score", 0)),
                "category": (e.get("category") or "uncategorised").strip(),
                "note": (e.get("note") or "").strip(),
            })
    out.sort(key=lambda x: (x["category"], x["phrase"]))
    print(f"  idioms: kept {len(out)}, dropped {dropped} (dupes/empty)")
    return out


def _load_english_ratings() -> dict:
    """Map exact reference -> English-audience familiarity, from eng_*.json."""
    mapping = {}
    for path in sorted(glob.glob(os.path.join(BUILD_DIR, "eng_*.json"))):
        with open(path, encoding="utf-8") as f:
            for row in json.load(f):
                ref = (row.get("reference") or "").strip()
                val = str(row.get("English", "")).lower().strip()
                if ref and val in FAM_VALUES:
                    mapping[ref] = val
    return mapping


def _derive_english(hindi: str, tamil: str) -> str:
    """Fallback English rating for refs an agent didn't cover."""
    if hindi == "high" and tamil == "high":
        return "high"
    if "high" in (hindi, tamil):
        return "medium"
    return "low"


def build_cultural() -> list[dict]:
    by_key: dict[str, dict] = {}
    dropped = 0
    eng_ratings = _load_english_ratings()
    for path in sorted(glob.glob(os.path.join(BUILD_DIR, "cultural_*.json"))):
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        for e in batch:
            ref = (e.get("reference") or "").strip()
            norm = textutils.normalize(ref)
            if not norm:
                dropped += 1
                continue
            fam_in = e.get("familiarity", {}) or {}
            fam = {}
            for lang in ("Hindi", "Tamil"):
                v = str(fam_in.get(lang, "medium")).lower().strip()
                fam[lang] = v if v in FAM_VALUES else "medium"
            # English-audience rating: researched (eng_*.json) or derived.
            fam["English"] = eng_ratings.get(ref) or _derive_english(fam["Hindi"], fam["Tamil"])
            aliases = [a.strip() for a in (e.get("aliases") or []) if a and a.strip()]
            if norm in by_key:  # merge aliases into the first occurrence
                existing = by_key[norm]
                existing["aliases"] = sorted(set(existing["aliases"]) | set(aliases))
                dropped += 1
                continue
            by_key[norm] = {
                "reference": ref,
                "aliases": sorted(set(aliases)),
                "category": _fix_category((e.get("category") or "uncategorised").strip()),
                "familiarity": fam,
                "note": (e.get("note") or "").strip(),
            }
    out = sorted(by_key.values(), key=lambda x: (x["category"], x["reference"]))
    print(f"  cultural: kept {len(out)}, dropped {dropped} (dupes/empty)")
    return out


_RECS = {"localize", "keep_english", "loanword_ok"}


def build_glossary() -> list[dict]:
    """Localization-decision glossary: term -> recommendation + recommended form."""
    by_term, dropped = {}, 0
    for path in sorted(glob.glob(os.path.join(BUILD_DIR, "loan_*.json"))):
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        for e in batch:
            term = (e.get("term") or "").strip()
            key = term.lower()
            rec = str(e.get("recommendation", "")).strip().lower()
            if not term or key in by_term or rec not in _RECS:
                dropped += 1
                continue
            by_term[key] = {
                "term": term,
                "domain": (e.get("domain") or "general").strip(),
                "recommendation": rec,
                "hindi": (e.get("hindi") or "").strip(),
                "tamil": (e.get("tamil") or "").strip(),
                "why": (e.get("why") or "").strip(),
            }
    out = sorted(by_term.values(), key=lambda x: (x["domain"], x["term"]))
    print(f"  glossary: kept {len(out)}, dropped {dropped} (dupes/empty/bad-rec)")
    return out


def main() -> None:
    print("Building dictionaries from", BUILD_DIR)
    idioms = build_idioms()
    cultural = build_cultural()
    glossary = build_glossary()
    with open(os.path.join(ROOT, "data", "hinglish_idioms.json"), "w",
              encoding="utf-8") as f:
        json.dump(idioms, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "data", "cultural_references.json"), "w",
              encoding="utf-8") as f:
        json.dump(cultural, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "data", "loanword_glossary.json"), "w",
              encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(idioms)} idioms, {len(cultural)} cultural references, "
          f"{len(glossary)} glossary terms.")


if __name__ == "__main__":
    main()
