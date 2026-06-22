"""Engine 7 — Audience Opportunity.

Dubbing difficulty and audience upside are *orthogonal*. Hard-to-dub comedy can
still be worth localising if the regional audience is huge; an easy-to-dub niche
talk may not be. So this engine answers a different question from the quality
score — "how big is the prize per language?" — and the scorer keeps the two
strictly separate (never multiplied).

Category is detected by keyword scoring. Per-category, per-language affinity
encodes how large/under-served the dubbed-content opportunity is.

SOURCES for the affinity matrix (so the numbers are defensible, not vibes):
  * Hindi is the largest single online-language audience in India; Indian-
    language internet users now far outnumber English ones (KPMG–Google
    "Indian Languages: Defining India's Internet"; TRAI subscriber data).
  * Tamil is among the most monetised, highest-engagement regional YouTube
    markets, with especially strong entertainment/film consumption (industry
    YouTube India audience reports; IAMAI/Kantar "Internet in India").
  * Education/finance demand skews to the Hindi belt's exam/retail-investing
    boom; entertainment over-indexes in Tamil's film-first culture.
These are directional estimates for prioritisation, not audited market sizes.
"""
from __future__ import annotations

CATEGORY_KEYWORDS = {
    "Education": ["explain", "concept", "formula", "lecture", "study", "learn",
                  "jee", "neet", "upsc", "exam", "tutorial", "syllabus",
                  "chapter", "notes", "marks", "revision"],
    "Finance": ["stock", "stocks", "investment", "invest", "returns", "portfolio",
                "crypto", "market", "zerodha", "mutual fund", "sip", "nifty",
                "sensex", "trading", "savings", "tax"],
    "Entertainment": ["comedy", "funny", "vlog", "reaction", "storytime", "prank",
                      "challenge", "roast", "skit", "meme", "drama", "gossip"],
    "Tech": ["startup", "ai", "product", "coding", "code", "app", "software",
             "engineer", "build", "developer", "tech", "gadget", "review",
             "founder", "saas"],
    "Lifestyle": ["food", "travel", "fashion", "fitness", "recipe", "workout",
                  "skincare", "vlog", "routine", "makeup", "diet", "health"],
    "News": ["politics", "government", "election", "policy", "economy", "report",
             "breaking", "minister", "parliament", "verdict", "protest"],
}

# Per-category, per-language opportunity (0-100). See module docstring sources.
# English = reach to the urban English-medium Indian + international/diaspora
# audience (relevant when dubbing Indian-language content INTO English).
CATEGORY_AFFINITY = {
    "Education":     {"Hindi": 95, "Tamil": 80, "English": 85},
    "Finance":      {"Hindi": 90, "Tamil": 78, "English": 80},
    "Entertainment": {"Hindi": 85, "Tamil": 90, "English": 70},
    "Tech":         {"Hindi": 80, "Tamil": 82, "English": 88},
    "Lifestyle":    {"Hindi": 88, "Tamil": 82, "English": 75},
    "News":         {"Hindi": 92, "Tamil": 78, "English": 72},
    "General":      {"Hindi": 85, "Tamil": 80, "English": 78},
}

RATIONALE = {
    "Education": {
        "Hindi": "Largest exam-prep & how-to demand sits in the Hindi belt.",
        "Tamil": "Strong study audience, but Tamil already has dense local edu-creators.",
        "English": "Huge English-medium & global demand for Indian edu-content.",
    },
    "Finance": {
        "Hindi": "Retail-investing boom is overwhelmingly Hindi-first.",
        "Tamil": "Growing investor base; less saturated than Hindi finance.",
        "English": "English is the default register of Indian markets/business.",
    },
    "Entertainment": {
        "Hindi": "Massive reach, but the most competitive shelf.",
        "Tamil": "Film-first culture — entertainment over-indexes and travels well.",
        "English": "Competes globally; pan-India English + diaspora reach.",
    },
    "Tech": {
        "Hindi": "Big, under-served Hindi tech-explainer gap.",
        "Tamil": "Engaged tech audience; Chennai/Coimbatore dev base.",
        "English": "Tech is English-first; largest cross-border audience.",
    },
    "Lifestyle": {
        "Hindi": "Broad lifestyle appetite across the Hindi belt.",
        "Tamil": "Loyal regional lifestyle following.",
        "English": "Urban English-medium + diaspora appetite.",
    },
    "News": {
        "Hindi": "Highest news-consumption volume nationally.",
        "Tamil": "Strong but well-served by entrenched Tamil news media.",
        "English": "English national-news reach + diaspora.",
    },
    "General": {
        "Hindi": "Default: largest single online-language audience.",
        "Tamil": "Default: high-engagement, well-monetised regional market.",
        "English": "Default: broad English-medium + international reach.",
    },
}


def analyze(transcript: str, languages: list[str] | None = None) -> dict:
    languages = languages or ["Hindi", "Tamil"]
    low = f" {transcript.lower()} "

    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(low.count(f" {kw} ") + low.count(f" {kw}") for kw in kws)

    best = max(scores, key=scores.get)
    detected = best if scores[best] > 0 else "General"

    affinity = CATEGORY_AFFINITY.get(detected, CATEGORY_AFFINITY["General"])
    opportunity = {
        lang: {
            "score": affinity.get(lang, 80),
            "rationale": RATIONALE.get(detected, RATIONALE["General"]).get(lang, ""),
        }
        for lang in languages
    }
    priority = sorted(languages, key=lambda l: opportunity[l]["score"], reverse=True)

    return {
        "detected_category": detected,
        "category_scores": scores,
        "opportunity_by_language": opportunity,
        "priority_order": priority,
    }
