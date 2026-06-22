"""Aggregate engine outputs into a per-language Dub Quality Score.

Design rule (deliberate): Dub Quality and Audience Opportunity are SEPARATE
dimensions and are never multiplied. A video that dubs badly should score low on
quality even if the audience prize is huge — we show both and let the operator
make the trade-off. Quality is a pure "how faithfully does this survive
localisation" measure.

The five quality penalties and weights (sum of weights = 1.0):
    semantic loss        0.35   most honest signal (meaning actually lost)
    idiomatic density    0.25   phrases needing adaptation, not translation
    cultural risk        0.20   references an audience may not recognise
    structural interleave 0.10  mid-clause language fusion (no clean seam)
    prosody dependency   0.10   meaning carried by delivery, not words
"""
from __future__ import annotations

WEIGHTS = {
    "semantic_loss": 0.35,
    "idiomatic_density": 0.25,
    "cultural_risk": 0.20,
    "structural_interleave": 0.10,
    "prosody_dependency": 0.10,
}

_CULTURAL_PENALTY = {"low": 0, "medium": 10, "high": 20}
_PROSODY_PENALTY = {"low": 0, "medium": 5, "high": 10}


def _grade(score: int) -> str:
    if score >= 80:
        return "Dubs cleanly"
    if score >= 60:
        return "Light adaptation needed"
    if score >= 40:
        return "Heavy localisation"
    return "Not recommended"


def _recommendation(lang: str, score: int, opportunity: int) -> str:
    if score >= 80:
        return f"High-confidence dub — a strong, low-effort {lang} candidate."
    if score >= 60:
        return f"Worth dubbing into {lang} with a light adaptation pass on flagged lines."
    if score >= 40:
        opp = "high" if opportunity >= 85 else "moderate"
        return (f"Only dub into {lang} if the {opp}-opportunity payoff justifies a "
                f"heavy localisation rewrite.")
    return f"Not recommended for a straight {lang} dub — it needs a near-rewrite."


def compute_language_score(results: dict, lang: str) -> dict:
    sem = results.get("semantic", {}).get("by_language", {}).get(lang, {})
    loss = sem.get("loss")
    semantic_available = loss is not None
    loss = loss or 0.0

    idiom_density = results.get("idiomatic", {}).get("idiom_density", 0.0)
    cultural_risk = results.get("cultural", {}).get("risk_by_language", {}).get(lang, "low")
    interleave = results.get("structural", {}).get("interleave_ratio", 0.0)
    prosody_dep = results.get("prosody", {}).get("prosody_dependency", "low")

    penalties = {
        "semantic": loss * 35,
        "idiomatic": min((idiom_density / 20) * 25, 25),
        "cultural": _CULTURAL_PENALTY.get(cultural_risk, 0),
        "structural": interleave * 10,
        "prosody": _PROSODY_PENALTY.get(prosody_dep, 0),
    }
    score = max(0, min(100, round(100 - sum(penalties.values()))))

    # --- Plain-English top risks (largest contributors first) ----------------
    n_risky = sum(
        1 for f in results.get("cultural", {}).get("found_detail", [])
        if f.get("familiarity", {}).get(lang) in ("low", "medium")
    )
    sentences = {
        "semantic": (f"~{round(loss * 100)}% of meaning is lost on a {lang} "
                     f"round-trip — idioms and wordplay don't survive translation."),
        "idiomatic": (f"Idiom-heavy speech ({idiom_density:.1f} slang hits per 100 "
                      f"words) needs adaptation, not literal translation."),
        "cultural": (f"{n_risky} culture-specific reference(s) may not land with a "
                     f"{lang} audience."),
        "structural": (f"Languages fuse mid-sentence in "
                       f"{round(interleave * 100)}% of clauses — no clean seam to "
                       f"re-voice along."),
        "prosody": ("Meaning leans on delivery and emphasis, raising lip-sync and "
                    "tone risk for the voice actor."),
    }
    ranked = sorted(penalties.items(), key=lambda kv: kv[1], reverse=True)
    top_risks = [sentences[k] for k, v in ranked if v >= 3][:3]
    if not semantic_available:
        top_risks.append("Note: back-translation check was unavailable "
                          "(rate-limited/offline), so semantic loss is treated as 0.")
    if not top_risks:
        top_risks = [f"No major localisation risks detected for {lang} — clean source."]

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
