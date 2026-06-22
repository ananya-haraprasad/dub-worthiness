# NLTK data must be present before any engine that tokenizes runs. Newer NLTK
# uses 'punkt_tab'; download both. Kept at the very top, before other imports.
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import hashlib
import html
import math
import os
import re
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from engines import (
    extractor,
    structural,
    semantic,
    idiomatic,
    cultural,
    prosody,
    opportunity,
    langs,
    dubber,
    localization,
)
import scorer

# --- Constants ---------------------------------------------------------------
# Source language is auto-detected (English / Hindi / Tamil); targets are the
# other two. Computed per analysis, so there is no fixed target list here.

GRADE_COLORS = {
    "Travels cleanly": "#2f7d4f",
    "Light adaptation needed": "#b9822b",
    "Heavy localisation": "#c2562f",
    "Not recommended": "#a8472f",
}

# Per-engine identity for the "why this score" breakdown bar (warm, distinct).
ENGINE_META = {
    "semantic":     ("Semantic loss", "#b4472b"),
    "localization": ("Localization",  "#1f6b5e"),
    "idiomatic":    ("Idioms",        "#c98a2b"),
    "cultural":     ("Cultural",      "#9c5a7c"),
    "structural":   ("Code-switch",   "#4f7a8a"),
    "prosody":      ("Prosody",       "#7d8450"),
}

_LEVEL_STYLE = {
    "low": ("#2f7d4f", "#dcfce7"), "medium": ("#b9822b", "#fef3c7"),
    "high": ("#a8472f", "#fee2e2"), "clean": ("#2f7d4f", "#dcfce7"),
    "mixed": ("#b9822b", "#fef3c7"), "interleaved": ("#a8472f", "#fee2e2"),
    "unknown": ("#8a8073", "#e2e8f0"), "undetermined": ("#8a8073", "#e2e8f0"),
}

# Sample videos a viewer can copy and paste in. Covers all three source languages.
# Presented as a neutral list, not a red/yellow/green verdict: the back-translation
# score reflects whether MEANING travels, not the fluency of one machine dub, so a
# confident per-language traffic-light isn't something to claim up front. Let the
# live score speak for itself. Label format: "Language · Sentence-case description".
SAMPLES = [
    ("English · Barbie monologue", "https://youtube.com/shorts/q9wKARQ8_pg"),
    ("Hindi · Antarctica vlog", "https://youtube.com/shorts/LPI9mLkIIS8"),
    ("Tamil · Assembly speech", "https://youtube.com/shorts/1CBtqHEDUXI"),
    ("Tamil · Shirt seller", "https://youtube.com/shorts/mRFs19QPAwI"),
    ("English · Skincare tips", "https://youtube.com/shorts/obRs6VrF9FE"),
]

# Hardcoded benchmark context so a single score feels grounded, not arbitrary.
BENCHMARKS = {
    "Education":     {"avg_score": 82, "note": "Dubs well. Structured speech, clear concepts."},
    "Finance":      {"avg_score": 78, "note": "Mostly clean but jargon-heavy."},
    "Tech":         {"avg_score": 71, "note": "English-heavy, but a fast-growing regional audience."},
    "Entertainment": {"avg_score": 54, "note": "Timing and slang make this hard."},
    "News":         {"avg_score": 68, "note": "Translates well, but cultural refs are risky."},
    "Lifestyle":    {"avg_score": 66, "note": "Casual register; some slang to adapt."},
    "General":      {"avg_score": 70, "note": "Mixed-genre baseline."},
}

st.set_page_config(page_title="Will It Travel?", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")


# --- Secrets / API key (works locally AND on Streamlit Cloud) ----------------
_PLACEHOLDER_KEYS = {"", "paste_your_key_here", "your_key_here", "none", "null"}


def get_api_key() -> str:
    try:
        key = st.secrets["SARVAM_API_KEY"]
    except Exception:
        key = os.getenv("SARVAM_API_KEY", "")
    key = (key or "").strip()
    return "" if key.lower() in _PLACEHOLDER_KEYS else key


@st.cache_resource(show_spinner=False)
def load_similarity_model():
    """~470MB multilingual model. Loaded once. Returns None if it can't fit."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


# --- Styling -----------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Hanken+Grotesk:wght@400;500;600;700;800&display=swap');
        :root { --paper:#f7f3ec; --surface:#fffdf8; --ink:#23201b; --muted:#736b5e;
                --faint:#a79d8c; --line:#e7decd; --clay:#c0573c; --teal:#1f6b5e; --gold:#b9822b; }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
        button, input, textarea, .stMarkdown, p, span, div, td, th, label {
            font-family:'Hanken Grotesk', system-ui, -apple-system, sans-serif; }
        .stApp { background: var(--paper); }
        .block-container { max-width: 1060px; padding-top: 2.2rem; }
        h1,h2,h3,h4 { color:var(--ink); font-family:'Fraunces',Georgia,serif; letter-spacing:-0.01em; }
        /* hide footer + deploy/menu, but KEEP the sidebar expand/collapse control */
        footer, [data-testid="stStatusWidget"], #MainMenu, .stDeployButton { display:none !important; }
        [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {
            display:flex !important; visibility:visible !important; opacity:1 !important; z-index:999; }

        .hero-title { font-family:'Fraunces',Georgia,serif; font-size:3rem; font-weight:600;
                      color:var(--ink); letter-spacing:-0.02em; line-height:1.04; margin-bottom:.2rem; }
        .hero-title .q { color:var(--clay); }
        .hero-sub { color:var(--muted); font-size:1.08rem; margin:.15rem 0 0; max-width:640px; line-height:1.5; }
        /* section label */
        .eyebrow { color:var(--clay); font-size:.72rem; font-weight:800; text-transform:uppercase;
                   letter-spacing:.14em; margin:0 0 .55rem; }

        /* verdict: flat, ruled at top, no pill or side bar */
        .verdict { border-top:2px solid var(--ink); padding:16px 0 2px; }
        .chips { margin:.1rem 0 .7rem; color:var(--muted); font-size:.86rem; }
        .chips b { color:var(--ink); font-weight:700; }
        .chips .sep { color:var(--faint); margin:0 9px; }
        .verdict-head { font-family:'Fraunces',serif; font-size:2rem; font-weight:600;
                        color:var(--ink); margin:.1rem 0; line-height:1.1; }
        .verdict-sub { color:var(--muted); font-size:.97rem; margin:.4rem 0 0; max-width:740px; line-height:1.55; }

        /* scorecards: flat bordered, no shadow */
        .card { border:1px solid var(--line); border-radius:6px; padding:20px 22px;
                background:var(--surface); height:100%; }
        .lang-row { display:flex; justify-content:space-between; align-items:baseline; }
        .lang-name { font-family:'Fraunces',serif; font-size:1.5rem; font-weight:600; color:var(--ink); }
        .grade { font-weight:800; font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; }
        .ring-wrap { display:flex; align-items:center; gap:18px; margin:.7rem 0 .2rem; }
        .opp { flex:1; }
        .opp-label { color:var(--faint); font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; }
        .opp-num { font-family:'Fraunces',serif; font-size:1.5rem; font-weight:600; color:var(--teal); line-height:1; }
        .opp-track { height:6px; background:#ece3d3; border-radius:4px; margin-top:7px; overflow:hidden; }
        .opp-fill { height:100%; background:var(--teal); border-radius:4px; }
        .reco { margin-top:.85rem; padding-top:.7rem; border-top:1px solid var(--line);
                font-weight:600; color:var(--ink); font-size:.92rem; }

        /* breakdown bar */
        .bd-head { color:var(--faint); font-size:.68rem; font-weight:800; text-transform:uppercase;
                   letter-spacing:.07em; margin:1rem 0 .35rem; }
        .bd-bar { display:flex; height:12px; border-radius:3px; overflow:hidden; background:#ece3d3; }
        .bd-bar > div { height:100%; }
        .bd-legend { margin-top:.5rem; font-size:.78rem; color:var(--muted); }
        .bd-legend .lg { display:inline-block; margin:0 12px 2px 0; white-space:nowrap; }
        .bd-legend .lg i { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; vertical-align:middle; }
        .lg-clean { color:#2f7d4f; font-weight:600; }

        /* risk: ruled rows, no boxes or side bars */
        .rp-row { display:flex; align-items:center; gap:14px; padding:13px 2px; border-bottom:1px solid var(--line); }
        .rp-name { font-weight:700; color:var(--ink); width:148px; flex:0 0 auto; }
        .rp-val { font-weight:800; color:var(--ink); width:92px; flex:0 0 auto; font-size:.95rem; }
        .rp-help { color:var(--muted); font-size:.86rem; flex:1; }
        .lvl { font-weight:800; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; width:74px; flex:0 0 auto; }

        /* transcript */
        mark.idiom { background:#f4da94; padding:0 2px; border-radius:2px; }
        mark.culture { background:#eac6b3; padding:0 2px; border-radius:2px; }
        .tx-box { border:1px solid var(--line); border-radius:6px; padding:16px 18px; background:var(--paper);
                  max-height:320px; overflow-y:auto; line-height:1.75; font-size:.95rem; color:var(--ink); }
        .legend { font-size:.8rem; color:var(--muted); margin-bottom:.4rem; }
        .empty { color:var(--faint); font-style:italic; }
        .bt-orig { color:var(--muted); margin-bottom:4px; } .bt-back { color:var(--ink); font-weight:600; }
        .small-muted { color:var(--muted); font-size:.86rem; }
        .dub-src { color:var(--muted); font-size:.92rem; font-style:italic; margin:.2rem 0 .9rem; }
        .dub-lang { font-family:'Fraunces',serif; font-weight:600; font-size:1.15rem; color:var(--ink); margin-bottom:.25rem; }
        .dub-text { border:1px solid var(--line); border-radius:6px; padding:11px 14px;
                    background:#eef4f1; color:#134a43; font-size:.92rem; line-height:1.6; min-height:58px; }
        .dub-empty { color:var(--faint); font-style:italic; }
        .dub-note { background:#fbf1df; border:1px solid #ecd6a8; border-radius:6px; padding:12px 15px;
                    color:#7a5a1e; font-size:.86rem; line-height:1.55; margin-top:.7rem; }
        .lg-table { width:100%; border-collapse:collapse; font-size:.9rem; margin-top:.3rem; }
        .lg-table th { text-align:left; color:var(--faint); font-size:.66rem; font-weight:800; text-transform:uppercase;
                       letter-spacing:.05em; padding:6px 10px; border-bottom:1px solid var(--line); }
        .lg-table td { padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
        .lg-term { font-weight:700; color:var(--ink); }
        .lg-mt { color:#a8472f; }
        .lg-nat { color:#2f7d4f; font-weight:700; }
        .lg-rec { font-size:.62rem; font-weight:800; color:var(--faint); text-transform:uppercase; letter-spacing:.04em; white-space:nowrap; }
        .vok { color:#2f7d4f; font-weight:800; text-align:center; }
        .vbad { color:#a8472f; font-weight:800; text-align:center; }
        .lg-score { font-size:.7rem; font-weight:700; color:var(--muted); }
        [data-testid="stMetricValue"] { font-size:1.4rem; font-weight:800; font-family:'Fraunces',serif; }

        /* home: steps as ruled editorial columns, signals as ruled rows */
        .step { border-top:2px solid var(--ink); padding:12px 16px 0 0; height:100%; }
        .step-n { font-family:'Fraunces',serif; font-size:1.6rem; color:var(--clay); font-weight:600; line-height:1; }
        .step-t { font-weight:700; color:var(--ink); margin:.4rem 0 .2rem; }
        .step-d { color:var(--muted); font-size:.88rem; line-height:1.5; }
        .signal { padding:0 0 13px; border-bottom:1px solid var(--line); margin-bottom:13px; }
        .signal-t { font-weight:700; color:var(--ink); font-size:.95rem; }
        .signal-d { color:var(--muted); font-size:.87rem; line-height:1.45; }

        /* loading */
        .load-wrap { text-align:center; padding:18px 0 8px; }
        .plane { font-size:1.7rem; display:inline-block; animation:fly 1.3s ease-in-out infinite; }
        @keyframes fly { 0%{transform:translateX(-7px) rotate(-4deg)} 50%{transform:translateX(7px) rotate(4deg)} 100%{transform:translateX(-7px) rotate(-4deg)} }
        .load-msg { font-family:'Fraunces',serif; font-size:1.2rem; color:var(--ink); margin-top:.5rem; }
        .load-track { height:8px; background:#ece3d3; border-radius:6px; overflow:hidden; max-width:440px; margin:12px auto 0; }
        .load-fill { height:100%; background:linear-gradient(90deg,var(--clay),var(--gold)); border-radius:6px; transition:width .35s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Small render helpers ----------------------------------------------------
def grade_pill(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#6b7280")
    return f'<span class="grade" style="color:{color}">{html.escape(grade)}</span>'


def level_chip(level: str) -> str:
    fg, _bg = _LEVEL_STYLE.get(level, ("#8a8073", "#e2e8f0"))
    return f'<span class="lvl" style="color:{fg}">{html.escape(level.upper())}</span>'


def score_ring(score: int, color: str, size: int = 118) -> str:
    r = 51
    circ = 2 * math.pi * r
    dash = circ * max(0, min(100, score)) / 100
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 118 118" style="flex:0 0 auto">'
        f'<circle cx="59" cy="59" r="{r}" fill="none" stroke="#ece3d3" stroke-width="11"/>'
        f'<circle cx="59" cy="59" r="{r}" fill="none" stroke="{color}" stroke-width="11" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}" '
        f'transform="rotate(-90 59 59)"/>'
        f'<text x="59" y="57" text-anchor="middle" font-size="30" font-weight="800" '
        f'fill="{color}">{score}</text>'
        f'<text x="59" y="76" text-anchor="middle" font-size="9.5" letter-spacing="1" '
        f'fill="#94a3b8">/ 100</text></svg>'
    )


def breakdown_bar(score: int, penalties: dict, grade_color: str) -> str:
    """A 100-wide bar: retained score + a coloured segment per points lost."""
    parts = [f'<div style="flex:0 0 {score}%;background:{grade_color}"></div>']
    legend = []
    for key, (label, color) in ENGINE_META.items():
        v = penalties.get(key, 0) or 0
        if v >= 0.5:
            parts.append(f'<div style="flex:0 0 {v}%;background:{color}" '
                         f'title="−{v:.0f} {label}"></div>')
            legend.append(f'<span class="lg"><i style="background:{color}"></i>'
                          f'{label} −{v:.0f}</span>')
    leg = (" ".join(legend) if legend
           else '<span class="lg-clean">No points lost. Clean source ✓</span>')
    return (f'<div class="bd-head">Why this score · starts at 100, minus risk</div>'
            f'<div class="bd-bar">{"".join(parts)}</div>'
            f'<div class="bd-legend">{leg}</div>')


# --- Analysis pipeline -------------------------------------------------------
# Playful, travel-themed loading copy by progress (no dry "Transcribing with…").
_LOAD_MSGS = [
    (0.10, "Packing your video's bags"),
    (0.50, "Listening closely and jotting down every word"),
    (0.85, "Sending it on a round trip to see what survives"),
    (0.95, "Checking what got lost along the way"),
    (1.01, "Almost there, stamping the passport"),
]


def _progress_cb_factory(placeholder):
    def cb(frac, _msg):
        f = min(1.0, max(0.0, float(frac)))
        playful = next(m for thr, m in _LOAD_MSGS if f < thr)
        placeholder.markdown(
            "<div class='load-wrap'><span class='plane'>✈️</span>"
            f"<div class='load-msg'>{playful}…</div>"
            f"<div class='load-track'><div class='load-fill' style='width:{int(f * 100)}%'>"
            "</div></div></div>",
            unsafe_allow_html=True)
    return cb


def run_analysis(api_key, source_sig, youtube_url, uploaded_path, cb):
    st.session_state.setdefault("_tx_cache", {})
    st.session_state.setdefault("_sem_cache", {})
    tx_cache = st.session_state["_tx_cache"]
    sem_cache = st.session_state["_sem_cache"]

    # 1) Transcription — reuse if we've already transcribed this exact source.
    if source_sig in tx_cache:
        cb(0.50, "Reusing cached transcript…")
        tx = tx_cache[source_sig]
    else:
        result = extractor.extract_and_transcribe(
            api_key, youtube_url=youtube_url, uploaded_path=uploaded_path,
            progress_cb=cb,
        )
        tx = result.to_dict()
        tx_cache[source_sig] = tx

    transcript = tx["transcript"]
    source_lang = langs.detect_source(tx.get("language_code"), transcript)
    targets = langs.targets_for(source_lang)
    res = {"transcript": tx, "source_lang": source_lang, "targets": targets}

    # 2) Cheap text engines
    cb(0.52, "Analysing structure, idioms, and cultural references…")
    res["structural"] = structural.analyze(transcript)
    res["idiomatic"] = idiomatic.analyze(transcript)
    res["cultural"] = cultural.analyze(transcript, targets)
    res["prosody"] = prosody.analyze(transcript, tx.get("duration_seconds"),
                                     tx.get("word_count"))
    res["opportunity"] = opportunity.analyze(transcript, targets)

    # 3) Semantic back-translation (slow; cache by source + transcript hash)
    th = hashlib.sha1((source_lang + "|" + transcript).encode("utf-8")).hexdigest()
    if th in sem_cache:
        cb(0.85, "Reusing cached semantic analysis…")
        res["semantic"] = sem_cache[th]
    else:
        model = load_similarity_model()
        if model is None:
            res["semantic"] = {
                "source_language": source_lang,
                "by_language": {l: {"similarity": None, "loss": None,
                                    "chunks_scored": 0} for l in targets},
                "worst_chunks": [], "translations": {},
                "note": "Semantic model could not load (likely memory limit).",
            }
        else:
            res["semantic"] = semantic.analyze(transcript, model, source_lang,
                                               targets, api_key, progress_cb=cb)
        sem_cache[th] = res["semantic"]

    # 4) Sample-dub excerpts (Sarvam-translated; audio is on-demand in UI)
    cb(0.90, "Preparing sample-dub text…")
    res["dub"] = dubber.build_excerpts(transcript, source_lang, targets, api_key)

    # 5) Localization gap (live Sarvam MT vs natural equivalents). MUST run before
    #    scoring: the localization difficulty feeds the Travel Score.
    cb(0.95, "Comparing Sarvam translation vs localisation…")
    res["localization"] = localization.analyze(transcript, targets, api_key)

    # 6) Score (uses semantic loss, localization difficulty, and the rest)
    cb(0.98, "Computing Travel Scores…")
    res["scores"] = scorer.compute_scores(res, targets)
    cb(1.0, "Done.")
    return res


# --- Transcript highlighting -------------------------------------------------
def highlight_transcript(transcript: str, idiom_phrases, culture_refs) -> str:
    spans = {}
    for ref in culture_refs:
        if ref.strip():
            spans[ref.strip().lower()] = "culture"
    for ph in idiom_phrases:  # idioms win ties (added last)
        if ph.strip():
            spans[ph.strip().lower()] = "idiom"
    if not spans:
        return html.escape(transcript)

    ordered = sorted(spans.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(p) for p in ordered), re.IGNORECASE)

    out, last = [], 0
    for m in pattern.finditer(transcript):
        out.append(html.escape(transcript[last:m.start()]))
        cls = spans.get(m.group(0).lower(), "idiom")
        out.append(f'<mark class="{cls}">{html.escape(m.group(0))}</mark>')
        last = m.end()
    out.append(html.escape(transcript[last:]))
    return "".join(out)


# --- Render: sections --------------------------------------------------------
def render_verdict(res):
    tx, pr = res["transcript"], res["prosody"]
    source, targets = res["source_lang"], res["targets"]
    cat = res["scores"]["detected_category"]
    dur = tx.get("duration_seconds") or 0
    by = res["scores"]["by_language"]

    best = res["scores"]["quality_priority_order"][0]
    best_s = by[best]

    scores = {l: by[l]["dub_quality_score"] for l in targets}
    if len(set(scores.values())) == 1:
        # Genuinely tied: don't imply one is better than the other.
        g = best_s["grade"]
        if g == "Travels cleanly":
            head = f"Travels cleanly into both {targets[0]} and {targets[1]}"
        else:
            head = f"{g} for both {targets[0]} and {targets[1]}"
    else:
        head = f"Dub into {best} first. {best_s['grade']} ({best_s['dub_quality_score']})"

    sep = '<span class="sep">·</span>'
    chips = (f'Source <b>{html.escape(source)}</b>{sep}<b>{html.escape(cat)}</b>{sep}'
             f'{int(dur // 60)}:{int(dur % 60):02d}{sep}'
             f'{pr["speaking_rate_wpm"]:.0f} wpm{sep}'
             f'{tx.get("word_count", 0):,} words')
    sub = (f"How well this {source} clip travels into "
           f"<b>{' and '.join(targets)}</b>. Each score starts at 100 and comes down "
           f"for the risks shown below.")
    st.markdown(
        f'<div class="verdict"><div class="chips">{chips}</div>'
        f'<div class="eyebrow">Recommendation</div>'
        f'<div class="verdict-head">{html.escape(head)}</div>'
        f'<div class="verdict-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render_scorecards(res):
    st.markdown('<div class="eyebrow">Per-language scorecard</div>', unsafe_allow_html=True)
    targets = res["targets"]
    cols = st.columns(len(targets))
    for col, lang in zip(cols, targets):
        s = res["scores"]["by_language"][lang]
        color = GRADE_COLORS.get(s["grade"], "#6b7280")
        bd = breakdown_bar(s["dub_quality_score"], s.get("penalties", {}), color)
        with col:
            st.markdown(
                f'<div class="card">'
                f'<div class="lang-row"><span class="lang-name">{lang}</span>'
                f'{grade_pill(s["grade"])}</div>'
                f'<div class="ring-wrap">'
                f'<div style="text-align:center"><div class="opp-label">Travel score</div>'
                f'{score_ring(s["dub_quality_score"], color)}</div>'
                f'</div>'
                f'<div class="reco">↳ {html.escape(s["recommendation"])}</div>'
                f'{bd}</div>',
                unsafe_allow_html=True,
            )


def render_risk_profile(res):
    st.markdown('<div class="eyebrow">Risk profile</div>', unsafe_allow_html=True)
    st.caption("The signals behind the score (localization is shown in its own "
               "panel above). Semantic loss and cultural risk are for the "
               "worst-affected target language.")
    targets = res["targets"]
    sem = res["semantic"]["by_language"]
    idi, cul, pr, stc = res["idiomatic"], res["cultural"], res["prosody"], res["structural"]

    worst_sem = max(targets, key=lambda l: (sem.get(l, {}).get("loss") or 0))
    loss = sem.get(worst_sem, {}).get("loss")
    loss_str = f"{loss*100:.0f}%" if loss is not None else "n/a"
    loss_lvl = ("high" if (loss or 0) >= 0.25 else "medium" if (loss or 0) >= 0.12 else "low")
    cul_worst = max(targets, key=lambda l: {"low": 0, "medium": 1, "high": 2}
                    .get(cul["risk_by_language"].get(l, "low"), 0))

    rows = [
        ("Semantic loss", loss_str, loss_lvl,
         f"Meaning lost on a {res['source_lang']}→{worst_sem}→{res['source_lang']} "
         f"round-trip. The most honest dubbability signal."),
        ("Idiomatic density", f"{idi['idiom_density']:.1f}/100w", idi["density_level"],
         (f"{idi['unique_idioms']} slang phrase(s) needing adaptation."
          if idi["unique_idioms"] else
          "No idioms matched. Note: detection is tuned for Roman/code-mixed text; "
          "native-script idioms aren't matched yet.")),
        ("Cultural risk",
         f"{cul['total_references']} ref" + ("" if cul['total_references'] == 1 else "s"),
         cul["risk_by_language"].get(cul_worst, "low"),
         (f"{cul_worst} viewers may be unfamiliar with some references."
          if cul["total_references"] else "No flagged cultural references.")),
        ("Code-switch", stc["switch_type"], stc["switch_type"],
         f"{int(stc['interleave_ratio']*100)}% of clauses fuse languages mid-sentence."
         if stc["qualifying_clauses"] else
         "Not enough long clauses to assess (short / fragmented speech)."),
        ("Prosody", pr["prosody_dependency"], pr["prosody_dependency"],
         "How much meaning rides on delivery vs words (text-based proxy)."),
    ]
    html_rows = ""
    for name, val, level, help_text in rows:
        html_rows += (
            f'<div class="rp-row">'
            f'<div class="rp-name">{html.escape(name)}</div>'
            f'<div class="rp-val">{html.escape(str(val))}</div>'
            f'{level_chip(level)}'
            f'<div class="rp-help">{html.escape(help_text)}</div></div>'
        )
    st.markdown(html_rows, unsafe_allow_html=True)


@st.fragment
def render_sample_dub(res, api_key):
    # A fragment so clicking "Voice" reruns only this block, not the whole page
    # (otherwise Streamlit dims the previous frame and you see ghosted cards).
    st.markdown('<div class="eyebrow">Hear it dubbed · sample</div>', unsafe_allow_html=True)
    st.caption("The opening, translated and voiced with Sarvam TTS so you can actually "
               "hear it. Audio is generated when you click, not on every run.")
    detected = res["transcript"].get("voice_gender", "unknown")
    det_label = detected if detected in ("male", "female") else "unclear"
    choice = st.radio(f"Dub voice  ·  auto-detected from the audio: **{det_label}**",
                      ["Auto", "Male", "Female"], horizontal=True, key="voice_choice")
    gender = {"Male": "male", "Female": "female"}.get(
        choice, detected if detected in ("male", "female") else "female")
    dub = res.get("dub", {})
    src = res["source_lang"]
    src_ex = dub.get("source_excerpt", "")
    if src_ex:
        st.markdown(f"<div class='dub-src'>Original ({html.escape(src)}): "
                    f"{html.escape(src_ex[:300])}</div>", unsafe_allow_html=True)

    th = hashlib.sha1(res["transcript"]["transcript"].encode("utf-8")).hexdigest()[:10]
    tts_cache = st.session_state.setdefault("_tts_cache", {})
    cols = st.columns(len(res["targets"]))
    for col, lang in zip(cols, res["targets"]):
        txt = dub.get("by_language", {}).get(lang, "")
        with col:
            st.markdown(f"<div class='dub-lang'>{lang}</div>", unsafe_allow_html=True)
            if txt:
                st.markdown(f"<div class='dub-text'>{html.escape(txt)}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='dub-text dub-empty'>Translation "
                            "unavailable (rate-limited).</div>", unsafe_allow_html=True)
            key = f"{th}:{lang}"
            if txt and st.button(f"🔊  Voice {lang} sample", key=f"tts_{lang}",
                                 use_container_width=True):
                try:
                    with st.spinner(f"Recording the {lang} voiceover…"):
                        tts_cache[key] = dubber.synthesize(txt, lang, api_key, gender)
                except dubber.DubError as e:
                    st.warning(f"Couldn't synthesise: {e}")
            if key in tts_cache:
                st.audio(tts_cache[key], format="audio/wav")

    st.markdown(
        "<div class='dub-note'><b>This is raw machine translation, not a finished dub.</b> "
        "It often writes English terms in the local script instead of finding a real word. "
        "For example “cuticle” becomes “க்யூட்டிகில்”, which is English in Tamil letters, "
        "not Tamil. That is a translation-quality issue, and it is deliberately not what the "
        "Travel Score measures. The score tells you whether a clip is worth dubbing. It "
        "can't vouch for one machine dub, because back-translation runs through the same "
        "engine both ways, so transliteration round-trips perfectly. A real dub needs "
        "Sarvam's Mayura translation and a native-speaker pass.</div>",
        unsafe_allow_html=True,
    )


def render_localization_gap(res):
    loc = res.get("localization", {})
    targets = [t for t in res["targets"] if t != "English"]
    if not any(loc.get("by_language", {}).get(t) for t in targets):
        return
    st.markdown('<div class="eyebrow">Localization · what Sarvam gets right vs. wrong</div>',
                unsafe_allow_html=True)
    st.caption("For common English terms, here is what Sarvam Mayura does live, next "
               "to the right call: localize to a native word, keep a fixed term in "
               "English (translating “baby oil” literally is wrong), or keep a "
               "naturalised loanword. A check means it got it right, a cross means it didn't.")
    rec_label = {"localize": "localize", "keep_english": "keep English",
                 "loanword_ok": "loanword"}
    cols = st.columns(len(targets))
    for col, lang in zip(cols, targets):
        rows = loc.get("by_language", {}).get(lang, [])
        with col:
            if not rows:
                st.markdown(f"<div class='dub-lang'>{lang}</div>"
                            "<div class='dub-empty'>No flagged terms.</div>",
                            unsafe_allow_html=True)
                continue
            n_ok = sum(1 for r in rows if r["correct"])
            st.markdown(f"<div class='dub-lang'>{lang} &nbsp;"
                        f"<span class='lg-score'>Sarvam correct: {n_ok}/{len(rows)}</span></div>",
                        unsafe_allow_html=True)
            body = ""
            for r in rows:
                ok = r["correct"]
                mt_cls = "lg-ok" if ok else "lg-mt"
                icon_cls, icon = ("vok", "✓") if ok else ("vbad", "✗")
                body += (
                    f"<tr><td class='lg-term'>{html.escape(r['term'])}</td>"
                    f"<td class='{mt_cls}'>{html.escape(r['mt_current'] or '—')}</td>"
                    f"<td><span class='lg-nat'>{html.escape(r['recommended'] or '—')}</span><br>"
                    f"<span class='lg-rec'>{rec_label.get(r['recommendation'], '')}</span></td>"
                    f"<td class='{icon_cls}'>{icon}</td></tr>")
            st.markdown(
                f"<table class='lg-table'><tr><th>Term</th><th>Sarvam now</th>"
                f"<th>Right call</th><th></th></tr>{body}</table>", unsafe_allow_html=True)


def render_transcript(res):
    st.markdown('<div class="eyebrow">Transcript explorer</div>', unsafe_allow_html=True)
    src, targets = res["source_lang"], res["targets"]
    translations = res["semantic"].get("translations", {})
    idi, cul = res["idiomatic"], res["cultural"]
    idiom_phrases = [f["phrase"] for f in idi["found_idioms"]]
    risky_refs = (cul["top_risky_references"]
                  + cul["references_found"].get("audience_skewed", [])
                  + cul["references_found"].get("niche_everywhere", []))
    highlighted = highlight_transcript(res["transcript"]["transcript"],
                                       idiom_phrases, risky_refs)
    tabs = st.tabs([f"{src} · original"] + [f"{t} · translated" for t in targets])
    with tabs[0]:
        if idiom_phrases or risky_refs:
            st.markdown(
                "<div class='legend'><mark class='idiom'>idiom / slang</mark> &nbsp; "
                "<mark class='culture'>cultural reference</mark></div>",
                unsafe_allow_html=True)
        st.markdown(f"<div class='tx-box'>{highlighted}</div>", unsafe_allow_html=True)
    for tab, lang in zip(tabs[1:], targets):
        with tab:
            txt = (translations.get(lang) or "").strip()
            box = (html.escape(txt) if txt
                   else "<span class='dub-empty'>Translation unavailable.</span>")
            st.markdown(f"<div class='tx-box'>{box}</div>", unsafe_allow_html=True)
    st.caption("The other tabs are machine translation, shown so a non-source-language "
               "reader can follow what was said.")

    worst = res["semantic"].get("worst_chunks", [])
    if worst:
        st.markdown("<div class='bd-head'>Lines with the most round-trip drift</div>",
                    unsafe_allow_html=True)
        st.caption("Scored line by line. These lost the most meaning on a round trip, "
                   "so they're worth a human check. Honest limit: this catches broad "
                   "drift, not every error (a garble that keeps the key words can still "
                   "slip), so read the full translation above too.")
        for w in worst[:4]:
            with st.expander(f"{w['language']} · {int(w['loss']*100)}% drift on this line"):
                st.markdown(
                    f"<div class='bt-orig'>Said: {html.escape(w['original'][:400])}</div>"
                    f"<div class='bt-back'>Sarvam → {html.escape(w['language'])}: "
                    f"{html.escape((w.get('forward') or '—')[:400])}</div>"
                    f"<div class='bt-back'>Round-tripped back: "
                    f"{html.escape(w['back_translated'][:400])}</div>",
                    unsafe_allow_html=True)


def render_footer(res):
    cat = res["scores"]["detected_category"]
    bench = BENCHMARKS.get(cat, BENCHMARKS["General"])
    best = res["scores"]["quality_priority_order"][0]
    your = res["scores"]["by_language"][best]["dub_quality_score"]
    delta = your - bench["avg_score"]
    dcolor, arrow = ("#15803d", "▲") if delta >= 0 else ("#b91c1c", "▼")

    with st.expander(f"Benchmark: your {your} vs. a typical {cat} clip ({bench['avg_score']})"):
        st.markdown(
            f"<p class='small-muted'>Your best-language score is <b>{your}</b>. "
            f"Typical <b>{html.escape(cat)}</b> content averages "
            f"<b>{bench['avg_score']}</b> "
            f"<span style='color:{dcolor};font-weight:700'>({arrow} {abs(delta)})</span>. "
            f"{html.escape(bench['note'])}</p>", unsafe_allow_html=True)
        rows = "".join(
            f"<tr style='background:{'#eef2ff' if k==cat else 'transparent'}'>"
            f"<td style='padding:5px 12px'>{html.escape(k)}</td>"
            f"<td style='padding:5px 12px;font-weight:700'>{v['avg_score']}</td>"
            f"<td style='padding:5px 12px;color:#64748b'>{html.escape(v['note'])}</td></tr>"
            for k, v in BENCHMARKS.items() if k != "General")
        st.markdown(
            f"<table style='border-collapse:collapse;font-size:.86rem;width:100%'>"
            f"<tr style='text-align:left;color:#64748b;border-bottom:1px solid #e5e7eb'>"
            f"<th style='padding:5px 12px'>Category</th><th style='padding:5px 12px'>Avg</th>"
            f"<th style='padding:5px 12px'>Note</th></tr>{rows}</table>",
            unsafe_allow_html=True)
    with st.expander("How the Travel Score is calculated"):
        st.markdown(
            """
The Travel Score starts at 100 and subtracts a capped penalty for six signals.
It tops out at 97, not 100, because no automated check should claim a perfect dub.
The numbers are the most each can take off:

| Signal | Max | What it measures |
|---|---|---|
| Semantic loss | 58 | Meaning lost on a translate then back-translate round trip. |
| Localization | 36 | Common terms machine translation mishandles for this language. |
| Idiomatic density | 20 | Slang and idioms that need adapting, not translating. |
| Cultural risk | 16 | References an audience may not recognise. |
| Structural interleave | 8 | Languages fused mid-sentence, with no clean seam to re-voice. |
| Prosody dependency | 8 | Meaning carried by delivery, not words (a text proxy). |

I added the **localization** signal after testing: the first five clustered every
clip at 85 to 95, because meaning round-trips fine for short, literal speech.
Localization catches the real failure, content full of English terms that get
transliterated instead of localized.

**What the score is, and isn't.** It rates whether a clip is *worth* dubbing, not
the fluency of one machine dub. Back-translation uses the same engine both ways,
so a transliteration like "cuticle" to "க்யூட்டிகில்" round-trips perfectly and
reads as clean. Output fluency needs a human pass (and Sarvam's Mayura
translation in production). The Localization panel above shows that gap directly.

**Calibrated round trip.** A faithful round trip never scores a perfect match.
Benign rewording, and for Hinglish the way the round trip normalises English
loanwords back to formal Hindi ("इंपॉर्टेंट" to "महत्वपूर्ण"), drop the similarity
without any real meaning loss. So a benign-drift floor is subtracted before this
counts against the score, and only loss beyond it is penalised. A clean clip then
reads clean; a clip whose slang genuinely collapses still drops.

**Honest limits.** Prosody is a text proxy. `langdetect` is weak on short text.
Idiom matching is tuned for Roman and code-mixed text, so for native-script
(Devanagari or Tamil) sources the score leans more on semantic loss and
localization. Translation and back-translation both run on Sarvam Mayura.
            """)
        st.caption(f"Idioms: {res['idiomatic']['dictionary_size']} · "
                   f"Cultural base: {res['cultural']['dictionary_size']} · "
                   f"STT: Sarvam saarika:v2.5 · Translation: Sarvam Mayura · "
                   f"TTS: Sarvam bulbul:v3 · Embeddings: MiniLM-L12-v2")


def render_dashboard(res, api_key):
    render_verdict(res)
    st.write("")
    render_scorecards(res)
    st.write("")
    render_sample_dub(res, api_key)
    st.write("")
    render_localization_gap(res)
    st.write("")
    render_risk_profile(res)
    st.write("")
    render_transcript(res)
    st.divider()
    render_footer(res)


def render_home_explainers():
    """Landing content shown before any analysis. The resume-facing front door."""
    st.write("")
    st.markdown('<div class="eyebrow">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Transcribe", "Sarvam speech-to-text turns the audio into text and "
         "keeps the languages as they were actually spoken."),
        ("2", "Analyse", "A set of engines read the transcript for meaning loss, "
         "slang, cultural references, code-switching, and pace."),
        ("3", "Score", "You get a Travel Score per language, the risks behind it, a "
         "sample dub you can hear, and where machine translation falls short."),
    ]
    cols = st.columns(3)
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f"<div class='step'><div class='step-n'>{n}</div>"
                     f"<div class='step-t'>{html.escape(t)}</div>"
                     f"<div class='step-d'>{html.escape(d)}</div></div>",
                     unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown('<div class="eyebrow">What the score looks at</div>', unsafe_allow_html=True)
    signals = [
        ("Meaning loss", "How much sense survives a translate and back-translate round trip."),
        ("Localization", "Whether everyday English terms get a real local word, or just "
         "get rewritten in the local script."),
        ("Idioms and slang", "Phrases where the literal meaning is not the real meaning."),
        ("Cultural references", "Names and ideas one audience knows and another may not."),
        ("Code-switching", "How tangled the language-mixing gets inside a sentence."),
        ("Pace and delivery", "How much of the meaning rides on tone and timing, not words."),
    ]
    cols = st.columns(2)
    for i, (t, d) in enumerate(signals):
        cols[i % 2].markdown(f"<div class='signal'><div class='signal-t'>{html.escape(t)}</div>"
                             f"<div class='signal-d'>{html.escape(d)}</div></div>",
                             unsafe_allow_html=True)


# --- App body ----------------------------------------------------------------
def main():
    inject_css()
    st.markdown('<div class="hero-title">Will It Travel<span class="q">?</span></div>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Drop in any video and see how well it travels '
                'into another language. Clear, honest localisation scoring, not a '
                'code-mix percentage.</p>', unsafe_allow_html=True)

    api_key = get_api_key()

    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "Drop in a YouTube link or a file. The tool transcribes it with "
            "**Sarvam STT**, then runs a handful of engines to estimate how well it "
            "dubs into another language."
        )
        st.markdown("**Languages, any direction**\n\nIt auto-detects the source, "
                    "currently **English, Hindi, or Tamil**, and scores dubbing into "
                    "the other two. More languages later.")
        st.caption(f"Clips are trimmed to the first "
                   f"{extractor.MAX_DURATION_SECONDS // 60} minutes to stay within the "
                   f"free Sarvam tier.")
        st.markdown("---")
        st.markdown("**Try an example**\n\nCopy one and paste it above.")
        _rows = "".join(
            f"<div class='s'><span class='t'>{html.escape(label)}</span>"
            f"<button onclick=\"cp('{url}',this)\">Copy</button></div>"
            for label, url in SAMPLES)
        components.html(
            "<style>@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600;700&display=swap');"
            "body{margin:0;font-family:'Hanken Grotesk',system-ui,sans-serif;background:transparent;}"
            ".s{display:flex;justify-content:space-between;align-items:center;gap:8px;"
            "padding:8px 0;border-bottom:1px solid #e7decd;}"
            ".t{font-size:.85rem;color:#23201b;}"
            "button{font-family:inherit;font-size:.72rem;font-weight:700;color:#7a4a3c;background:#fffdf8;"
            "border:1px solid #d8b9ad;border-radius:5px;padding:4px 11px;cursor:pointer;white-space:nowrap;}"
            "button:hover{background:#f3e7e1;}</style>"
            f"<div>{_rows}</div>"
            "<script>function cp(u,b){var t=document.createElement('textarea');t.value=u;"
            "document.body.appendChild(t);t.select();try{document.execCommand('copy');}catch(e){}"
            "document.body.removeChild(t);var o=b.textContent;b.textContent='Copied';"
            "setTimeout(function(){b.textContent=o;},1200);}</script>",
            height=len(SAMPLES) * 42 + 8,
        )

    if not api_key:
        st.error(
            "**No Sarvam API key found.** Add it to `.streamlit/secrets.toml` "
            "locally, or in **Settings, then Secrets** on Streamlit Cloud. You can "
            "get a key at dashboard.sarvam.ai."
        )
        st.stop()

    # --- Input ---------------------------------------------------------------
    st.markdown('<div class="eyebrow">Try it</div>', unsafe_allow_html=True)
    tab_url, tab_file = st.tabs(["🔗  YouTube link", "📁  Upload a file"])
    with tab_url:
        url = st.text_input("YouTube URL",
                            placeholder="Paste a YouTube link (shorts work great)",
                            label_visibility="collapsed")
    with tab_file:
        upload = st.file_uploader("Upload MP3, MP4 or WAV (up to 200 MB)",
                                  type=["mp3", "mp4", "wav", "m4a", "webm"],
                                  label_visibility="collapsed")

    analyse = st.button("See if it travels  →", type="primary")
    st.caption("Works for English, Hindi, and Tamil right now. Any video, any of the "
               "three as the source.")

    if analyse:
        if not url and not upload:
            st.warning("Add a YouTube link or a file first.")
            st.stop()

        load = st.empty()
        cb = _progress_cb_factory(load)
        tmp_path = None
        try:
            if upload is not None:
                data = upload.getvalue()
                source_sig = "file:" + hashlib.sha1(data).hexdigest()
                suffix = os.path.splitext(upload.name)[1] or ".bin"
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                res = run_analysis(api_key, source_sig, None, tmp_path, cb)
            else:
                res = run_analysis(api_key, "url:" + url.strip(), url.strip(), None, cb)
            st.session_state["analysis"] = res
        except extractor.ExtractionError as e:
            st.error(f"**Could not process this one.** {e}")
            st.stop()
        except Exception as e:  # noqa: BLE001 - surface a friendly message
            st.error("**Something went wrong.** Try a shorter or different clip. "
                     f"Details: {e}")
            st.stop()
        finally:
            load.empty()
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if "analysis" in st.session_state:
        st.divider()
        render_dashboard(st.session_state["analysis"], api_key)
    else:
        render_home_explainers()


if __name__ == "__main__":
    main()
