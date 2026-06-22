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
#
# NOTE: an "untranslated words" signal (flagging zero-frequency tokens in the
# translation) was tried and removed. It could not tell a genuine failure (a
# colloquial word left transliterated) from a culturally-specific term that
# CORRECTLY stays as-is (golgappa, dish/place names, people's names) — both are
# zero-frequency. It false-flagged legitimate content (e.g. "Golgappas"), so it
# was doing more harm than good. The back-translation signal is what we trust for
# meaning; it does NOT measure fluency (see the methodology note in app.py).
WEIGHTS = {
    "semantic_loss": 58,
    "localization": 36,
    "idiomatic_density": 20,
    "cultural_risk": 16,
    "structural_interleave": 8,
    "prosody_dependency": 8,
}

_CULTURAL_PENALTY = {"low": 0, "medium": 8, "high": 16}
_PROSODY_PENALTY = {"low": 0, "medium": 4, "high": 8}


# Grade bands: 85+ travels cleanly, 70-84 workable with a pass, below 70 needs
# real localisation work. Calibrated to the actual score distribution under
# per-sentence scoring: clean clips land ~85-95 (a high-drift line pulls them off
# a perfect score), genuinely hard clips land below ~80, with a natural gap
# between. A 90 cutoff sat inside the clean cluster and mislabelled good clips.
def _grade(score: int) -> str:
    if score >= 85:
        return "Travels cleanly"
    if score >= 70:
        return "Light adaptation needed"
    if score >= 50:
        return "Heavy localisation"
    return "Not recommended"


def _recommendation(lang: str, score: int) -> str:
    # Thresholds match _grade() so the wording never contradicts the grade.
    if score >= 85:
        return f"A strong, low-effort {lang} candidate. You can dub this with confidence."
    if score >= 70:
        return f"Workable in {lang} with a light pass on the flagged lines."
    if score >= 50:
        return f"Needs a heavy localisation rewrite before it works in {lang}."
    return f"Not a good fit for a straight {lang} dub. It would need a near-rewrite."


def compute_language_score(results: dict, lang: str) -> dict:
    sem = results.get("semantic", {}).get("by_language", {}).get(lang, {})
    loss = sem.get("loss")
    semantic_available = loss is not None
    loss = loss or 0.0
    max_loss = sem.get("max_loss")
    max_loss = max_loss if max_loss is not None else loss
    # Blend the average loss with the worst single chunk, so one badly-mangled
    # segment still hurts and two targets with different worst cases score apart.
    eff_loss = 0.7 * loss + 0.3 * max_loss

    idiom_density = results.get("idiomatic", {}).get("idiom_density", 0.0)
    cultural_risk = results.get("cultural", {}).get("risk_by_language", {}).get(lang, "low")
    interleave = results.get("structural", {}).get("interleave_ratio", 0.0)
    prosody_dep = results.get("prosody", {}).get("prosody_dependency", "low")

    loc = results.get("localization", {})
    loc_rows = loc.get("by_language", {}).get(lang, [])
    loc_wrong = sum(1 for r in loc_rows if not r.get("correct", True))
    total_hits = loc.get("all_matches_count", 0)
    words = max((results.get("transcript", {}) or {}).get("word_count", 0) or 1, 1)
    # Anglicization burden: how much the source leans on English terms that won't
    # truly localize. A clip dense in English jargon (a skincare routine: serum,
    # exfoliate, dewy) barely reaches a Hindi/Tamil viewer even when "translated",
    # because the dub stays mostly English. Length-normalized (per 100 words) so a
    # short clip isn't unfairly safe; wrong MT calls (a localizable word that gets
    # transliterated) count double. This, not a count of wrong calls alone, is what
    # makes a genuinely un-localizable clip score low.
    loc_burden = (total_hits + loc_wrong) / words * 100
    localization_penalty = min(loc_burden * 3.2, WEIGHTS["localization"])

    penalties = {
        "semantic": eff_loss * 58,
        "localization": localization_penalty,
        "idiomatic": min((idiom_density / 20) * 20, 20),
        "cultural": _CULTURAL_PENALTY.get(cultural_risk, 0),
        "structural": interleave * 8,
        "prosody": _PROSODY_PENALTY.get(prosody_dep, 0),
    }
    # Cap at 97, not 100: a flawless score reads as overconfident, and no automated
    # estimate should claim a perfect dub (delivery, casting, and timing always
    # carry residual risk a transcript can't see). 97 still means "as clean as it gets".
    score = max(0, min(97, round(100 - sum(penalties.values()))))

    # --- Plain-English top risks (largest contributors first) ----------------
    n_risky = sum(
        1 for f in results.get("cultural", {}).get("found_detail", [])
        if f.get("familiarity", {}).get(lang) in ("low", "medium")
    )
    sentences = {
        "semantic": (f"About {round(loss * 100)}% of the meaning slips on a {lang} "
                     f"round trip. Idioms and wordplay don't survive."),
        "localization": (f"Leans heavily on English terms ({total_hits} in {words} "
                         f"words). A {lang} dub stays mostly English unless they're "
                         f"localized"
                         + (f", and MT transliterates {loc_wrong} that have a native "
                            f"word." if loc_wrong else ".")),
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
        "recommendation": _recommendation(lang, score),
        "semantic_available": semantic_available,
    }


def compute_scores(results: dict, languages: list[str] | None = None) -> dict:
    languages = languages or ["Hindi", "Tamil"]
    by_language = {lang: compute_language_score(results, lang) for lang in languages}

    # Rank by score; break ties by lower semantic loss (the better round-trip).
    sem = results.get("semantic", {}).get("by_language", {})
    priority = sorted(
        languages,
        key=lambda l: (by_language[l]["dub_quality_score"],
                       -(sem.get(l, {}).get("loss") or 0.0)),
        reverse=True,
    )
    return {
        "by_language": by_language,
        "quality_priority_order": priority,
        "opportunity_priority_order": results.get("opportunity", {}).get("priority_order", languages),
        "detected_category": results.get("opportunity", {}).get("detected_category", "General"),
        "weights": WEIGHTS,
    }
