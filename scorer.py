"""Aggregate engine outputs into a per-language Travel Score.

Design rule (deliberate): the Travel Score and Audience Opportunity are SEPARATE
dimensions and are never multiplied. A video that dubs badly should score low on
quality even if the audience prize is huge. We show both and let the operator
make the trade-off.

The score starts at 100 and subtracts a capped penalty for each of six signals.
'localization' was added after testing showed the original five clustered every
clip at 85-95: meaning round-trips fine for short, literal speech, so semantic
loss alone barely moved. The localization signal catches the real failure mode,
content full of English terms that machine translation transliterates rather
than localizes (the skincare-clip problem).

  semantic loss        meaning lost on a translate -> back-translate round trip
  localization         share of common terms MT mishandles for this language
  idiomatic density    slang/idioms that need adapting, not translating
  cultural risk        references an audience may not recognise
  structural interleave mid-sentence language fusion (no clean seam)
  prosody dependency   meaning carried by delivery, not words

Known gap: idiom detection is tuned to Roman/code-mixed text, so for native-
script (Devanagari/Tamil) sources the idiomatic signal under-reads and the score
leans more on semantic loss and cultural risk.
"""
from __future__ import annotations

# Max penalty each signal can subtract from 100.
WEIGHTS = {
    "semantic_loss": 40,
    "localization": 30,
    "idiomatic_density": 20,
    "cultural_risk": 16,
    "structural_interleave": 8,
    "prosody_dependency": 8,
}

_CULTURAL_PENALTY = {"low": 0, "medium": 8, "high": 16}
_PROSODY_PENALTY = {"low": 0, "medium": 4, "high": 8}


def _grade(score: int) -> str:
    if score >= 85:
        return "Travels cleanly"
    if score >= 65:
        return "Light adaptation needed"
    if score >= 45:
        return "Heavy localisation"
    return "Not recommended"


def _recommendation(lang: str, score: int, opportunity: int) -> str:
    if score >= 80:
        return f"A strong, low-effort {lang} candidate. You can dub this with confidence."
    if score >= 60:
        return f"Worth dubbing into {lang}, with a light pass on the flagged lines."
    if score >= 40:
        opp = "high" if opportunity >= 85 else "moderate"
        return (f"Dub into {lang} only if the {opp} audience payoff is worth a heavy "
                f"localisation rewrite.")
    return f"Not a good fit for a straight {lang} dub. It would need a near-rewrite."


def compute_language_score(results: dict, lang: str) -> dict:
    sem = results.get("semantic", {}).get("by_language", {}).get(lang, {})
    loss = sem.get("loss")
    semantic_available = loss is not None
    loss = loss or 0.0

    idiom_density = results.get("idiomatic", {}).get("idiom_density", 0.0)
    cultural_risk = results.get("cultural", {}).get("risk_by_language", {}).get(lang, "low")
    interleave = results.get("structural", {}).get("interleave_ratio", 0.0)
    prosody_dep = results.get("prosody", {}).get("prosody_dependency", "low")

    loc_rows = results.get("localization", {}).get("by_language", {}).get(lang, [])
    loc_wrong = sum(1 for r in loc_rows if not r.get("correct", True))

    penalties = {
        "semantic": loss * 40,
        "localization": min(loc_wrong, 5) * 6,
        "idiomatic": min((idiom_density / 20) * 20, 20),
        "cultural": _CULTURAL_PENALTY.get(cultural_risk, 0),
        "structural": interleave * 8,
        "prosody": _PROSODY_PENALTY.get(prosody_dep, 0),
    }
    score = max(0, min(100, round(100 - sum(penalties.values()))))

    # --- Plain-English top risks (largest contributors first) ----------------
    n_risky = sum(
        1 for f in results.get("cultural", {}).get("found_detail", [])
        if f.get("familiarity", {}).get(lang) in ("low", "medium")
    )
    sentences = {
        "semantic": (f"About {round(loss * 100)}% of the meaning slips on a {lang} "
                     f"round trip. Idioms and wordplay don't survive."),
        "localization": (f"{loc_wrong} common term(s) don't localise cleanly into "
                         f"{lang}. Machine translation transliterates or mistranslates "
                         f"them, so a dub needs hand-fixing."),
        "idiomatic": (f"Idiom-heavy speech ({idiom_density:.1f} slang hits per 100 "
                      f"words). These need adapting, not translating."),
        "cultural": (f"{n_risky} culture-specific reference(s) may not land with a "
                     f"{lang} audience."),
        "structural": (f"Languages fuse mid-sentence in {round(interleave * 100)}% of "
                       f"clauses, so there's no clean seam to re-voice."),
        "prosody": ("A lot of the meaning rides on delivery and emphasis, which raises "
                    "lip-sync and tone risk."),
    }
    ranked = sorted(penalties.items(), key=lambda kv: kv[1], reverse=True)
    top_risks = [sentences[k] for k, v in ranked if v >= 3][:3]
    if not semantic_available:
        top_risks.append("Heads up: the back-translation check didn't run "
                          "(rate-limited or offline), so semantic loss is counted as 0.")
    if not top_risks:
        top_risks = [f"No major localisation risks for {lang}. Clean source."]

    opp = (results.get("opportunity", {})
           .get("opportunity_by_language", {})
           .get(lang, {}).get("score", 0))

    return {
        "dub_quality_score": score,
        "opportunity_score": opp,
        "grade": _grade(score),
        "penalties": {k: round(v, 1) for k, v in penalties.items()},
        "top_risks": top_risks,
        "recommendation": _recommendation(lang, score, opp),
        "semantic_available": semantic_available,
    }


def compute_scores(results: dict, languages: list[str] | None = None) -> dict:
    languages = languages or ["Hindi", "Tamil"]
    by_language = {lang: compute_language_score(results, lang) for lang in languages}

    # "Where to start": rank by quality, breaking ties with opportunity. This is
    # a transparent rule, not a blended score that hides the trade-off.
    priority = sorted(
        languages,
        key=lambda l: (by_language[l]["dub_quality_score"],
                       by_language[l]["opportunity_score"]),
        reverse=True,
    )
    return {
        "by_language": by_language,
        "quality_priority_order": priority,
        "opportunity_priority_order": results.get("opportunity", {}).get("priority_order", languages),
        "detected_category": results.get("opportunity", {}).get("detected_category", "General"),
        "weights": WEIGHTS,
    }
