# NLTK data must be present before any engine that tokenizes runs. Newer NLTK
# uses 'punkt_tab'; download both. Kept at the very top, before other imports.
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import hashlib
import html
import json
import math
import os
import re
import tempfile

import streamlit as st

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
    "Travels cleanly": "#15803d",
    "Light adaptation needed": "#b45309",
    "Heavy localisation": "#c2410c",
    "Not recommended": "#b91c1c",
    "—": "#6b7280",
}

# Per-engine identity for the "why this score" breakdown bar.
ENGINE_META = {
    "semantic":     ("Semantic loss", "#6366f1"),
    "localization": ("Localization",  "#0891b2"),
    "idiomatic":    ("Idioms",        "#f59e0b"),
    "cultural":     ("Cultural",      "#ec4899"),
    "structural":   ("Code-switch",   "#14b8a6"),
    "prosody":      ("Prosody",       "#8b5cf6"),
}

_LEVEL_STYLE = {
    "low": ("#15803d", "#dcfce7"), "medium": ("#b45309", "#fef3c7"),
    "high": ("#b91c1c", "#fee2e2"), "clean": ("#15803d", "#dcfce7"),
    "mixed": ("#b45309", "#fef3c7"), "interleaved": ("#b91c1c", "#fee2e2"),
    "unknown": ("#475569", "#e2e8f0"), "undetermined": ("#475569", "#e2e8f0"),
}

# Hardcoded benchmark context so a single score feels grounded, not arbitrary.
BENCHMARKS = {
    "Education":     {"avg_score": 82, "note": "Dubs well — structured speech, clear concepts."},
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
        :root { --ink:#0f172a; --muted:#64748b; --faint:#94a3b8; --line:#e6e8ee;
                --soft:#f5f6fa; --brand:#4f46e5; }
        .block-container { max-width: 1140px; padding-top: 2rem; }
        h1,h2,h3 { color: var(--ink); letter-spacing:-0.01em; }
        .hero-title { font-size:2.4rem; font-weight:800; margin-bottom:.1rem;
                      background:linear-gradient(90deg,#4f46e5,#0ea5e9);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .hero-sub { color:var(--muted); font-size:1.04rem; margin:.1rem 0 0; }
        .eyebrow { color:var(--brand); font-size:.74rem; font-weight:800;
                   text-transform:uppercase; letter-spacing:.09em; margin:0 0 .35rem; }

        /* verdict band */
        .verdict { border:1px solid var(--line); border-radius:18px; padding:22px 26px;
                   background:linear-gradient(135deg,#f7f8ff,#ffffff 55%);
                   box-shadow:0 1px 3px rgba(16,24,40,.05); }
        .chips { margin:.1rem 0 .6rem; }
        .chip { display:inline-block; background:#fff; border:1px solid var(--line);
                border-radius:999px; padding:4px 12px; font-size:.8rem; font-weight:600;
                color:#334155; margin:0 6px 6px 0; }
        .chip b { color:var(--ink); }
        .verdict-head { font-size:1.65rem; font-weight:800; color:var(--ink); margin:.1rem 0; }
        .verdict-sub { color:var(--muted); font-size:.96rem; margin:.25rem 0 0; max-width:760px; }

        /* cards */
        .card { border:1px solid var(--line); border-radius:18px; padding:20px 22px;
                background:#fff; box-shadow:0 1px 3px rgba(16,24,40,.05); height:100%; }
        .lang-row { display:flex; justify-content:space-between; align-items:center; }
        .lang-name { font-size:1.3rem; font-weight:800; color:var(--ink); }
        .pill { display:inline-block; padding:4px 11px; border-radius:999px;
                font-size:.72rem; font-weight:700; color:#fff; }
        .ring-wrap { display:flex; align-items:center; gap:18px; margin:.5rem 0 .2rem; }
        .opp { flex:1; }
        .opp-label { color:var(--faint); font-size:.7rem; font-weight:800;
                     text-transform:uppercase; letter-spacing:.06em; }
        .opp-num { font-size:1.4rem; font-weight:800; color:#0ea5e9; line-height:1; }
        .opp-track { height:7px; background:#eef0f4; border-radius:5px; margin-top:6px; overflow:hidden; }
        .opp-fill { height:100%; background:#0ea5e9; border-radius:5px; }
        .reco { margin-top:.7rem; padding-top:.7rem; border-top:1px dashed var(--line);
                font-weight:600; color:#1e293b; font-size:.92rem; }

        /* breakdown bar */
        .bd-head { color:var(--faint); font-size:.7rem; font-weight:800;
                   text-transform:uppercase; letter-spacing:.06em; margin:.9rem 0 .3rem; }
        .bd-bar { display:flex; height:14px; border-radius:7px; overflow:hidden;
                  background:#eef0f4; }
        .bd-bar > div { height:100%; }
        .bd-legend { margin-top:.45rem; font-size:.78rem; color:#475569; }
        .bd-legend .lg { display:inline-block; margin:0 10px 2px 0; white-space:nowrap; }
        .bd-legend .lg i { display:inline-block; width:9px; height:9px; border-radius:2px;
                           margin-right:4px; vertical-align:middle; }
        .lg-clean { color:#15803d; font-weight:600; }

        /* risk profile rows */
        .rp-row { display:flex; align-items:center; gap:14px; padding:12px 16px;
                  border:1px solid var(--line); border-left-width:4px; border-radius:12px;
                  background:#fff; margin-bottom:8px; }
        .rp-name { font-weight:700; color:var(--ink); width:150px; flex:0 0 auto; }
        .rp-val { font-weight:800; color:var(--ink); width:92px; flex:0 0 auto; font-size:.95rem; }
        .rp-help { color:var(--muted); font-size:.86rem; flex:1; }
        .lvl { display:inline-block; padding:3px 9px; border-radius:6px; font-size:.7rem;
               font-weight:800; letter-spacing:.03em; width:84px; text-align:center; flex:0 0 auto; }

        /* transcript */
        mark.idiom { background:#fde68a; padding:0 2px; border-radius:3px; }
        mark.culture { background:#fbcfe8; padding:0 2px; border-radius:3px; }
        .tx-box { border:1px solid var(--line); border-radius:14px; padding:16px 18px;
                  background:var(--soft); max-height:320px; overflow-y:auto; line-height:1.75;
                  font-size:.95rem; color:#1e293b; }
        .legend { font-size:.8rem; color:var(--muted); margin-bottom:.4rem; }
        .empty { color:var(--faint); font-style:italic; }
        .bt-orig { color:#475569; margin-bottom:4px; } .bt-back { color:#0f172a; font-weight:600; }
        .small-muted { color:var(--muted); font-size:.85rem; }
        .dub-src { color:var(--muted); font-size:.9rem; font-style:italic;
                   border-left:3px solid var(--line); padding-left:10px; margin:.2rem 0 .8rem; }
        .dub-lang { font-weight:800; color:var(--ink); margin-bottom:.2rem; }
        .dub-text { border:1px solid var(--line); border-left:4px solid #0ea5e9;
                    border-radius:10px; padding:11px 14px; background:#f0f9ff;
                    color:#0c4a6e; font-size:.92rem; line-height:1.6; min-height:58px; }
        .dub-empty { color:var(--faint); font-style:italic; }
        .dub-note { background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
                    padding:12px 15px; color:#92400e; font-size:.86rem; line-height:1.55;
                    margin-top:.7rem; }
        .lg-table { width:100%; border-collapse:collapse; font-size:.9rem; margin-top:.3rem; }
        .lg-table th { text-align:left; color:var(--faint); font-size:.68rem; font-weight:800;
                       text-transform:uppercase; letter-spacing:.05em; padding:6px 10px;
                       border-bottom:1px solid var(--line); }
        .lg-table td { padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
        .lg-term { font-weight:700; color:var(--ink); }
        .lg-mt { color:#b91c1c; }
        .lg-nat { color:#15803d; font-weight:700; }
        .lg-tag { display:inline-block; font-size:.62rem; font-weight:800; padding:2px 6px;
                  border-radius:5px; margin-left:6px; }
        .lg-tag.bad { color:#b91c1c; background:#fee2e2; }
        .lg-tag.ok { color:#15803d; background:#dcfce7; }
        .lg-ok { color:#15803d; }
        .lg-rec { font-size:.64rem; font-weight:800; color:var(--faint); text-transform:uppercase;
                  letter-spacing:.04em; white-space:nowrap; }
        .vok { color:#15803d; font-weight:800; text-align:center; }
        .vbad { color:#b91c1c; font-weight:800; text-align:center; }
        .lg-score { font-size:.7rem; font-weight:700; color:var(--muted); }
        [data-testid="stMetricValue"] { font-size:1.4rem; font-weight:800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Small render helpers ----------------------------------------------------
def grade_pill(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#6b7280")
    return f'<span class="pill" style="background:{color}">{html.escape(grade)}</span>'


def level_chip(level: str) -> str:
    fg, bg = _LEVEL_STYLE.get(level, ("#475569", "#e2e8f0"))
    return f'<span class="lvl" style="color:{fg};background:{bg}">{html.escape(level.upper())}</span>'


def score_ring(score: int, color: str, size: int = 118) -> str:
    r = 51
    circ = 2 * math.pi * r
    dash = circ * max(0, min(100, score)) / 100
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 118 118" style="flex:0 0 auto">'
        f'<circle cx="59" cy="59" r="{r}" fill="none" stroke="#eef0f4" stroke-width="11"/>'
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
           else '<span class="lg-clean">No points lost — clean source ✓</span>')
    return (f'<div class="bd-head">Why this score · starts at 100, minus risk</div>'
            f'<div class="bd-bar">{"".join(parts)}</div>'
            f'<div class="bd-legend">{leg}</div>')


# --- Analysis pipeline -------------------------------------------------------
def _progress_cb_factory(bar, status):
    def cb(frac, msg):
        bar.progress(min(1.0, max(0.0, float(frac))))
        status.markdown(f"&nbsp;&nbsp;⏳ &nbsp;{html.escape(msg)}")
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
                "worst_chunks": [],
                "note": "Semantic model could not load (likely memory limit).",
            }
        else:
            res["semantic"] = semantic.analyze(transcript, model, source_lang,
                                               targets, progress_cb=cb)
        sem_cache[th] = res["semantic"]

    # 4) Score
    cb(0.92, "Computing Travel Scores…")
    res["scores"] = scorer.compute_scores(res, targets)

    # 5) Sample-dub excerpts (cheap text translations; audio is on-demand in UI)
    cb(0.95, "Preparing sample-dub text…")
    res["dub"] = dubber.build_excerpts(transcript, source_lang, targets)

    # 6) Localization gap — live MT vs natural equivalents for flagged terms
    cb(0.98, "Comparing machine translation vs localisation…")
    res["localization"] = localization.analyze(transcript, targets)
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
    opp_best = res["scores"]["opportunity_priority_order"][0]

    grades = {l: by[l]["grade"] for l in targets}
    if len(set(grades.values())) == 1 and best_s["grade"] == "Travels cleanly":
        head = f"Travels cleanly into both {targets[0]} and {targets[1]}"
    else:
        head = f"Best bet: dub into {best} — {best_s['grade'].lower()} ({best_s['dub_quality_score']})"

    chips = (
        f'<span class="chip">Source <b>{html.escape(source)}</b></span>'
        f'<span class="chip">{html.escape(cat)}</span>'
        f'<span class="chip">{int(dur//60)}:{int(dur%60):02d}</span>'
        f'<span class="chip">{pr["speaking_rate_wpm"]:.0f} wpm</span>'
        f'<span class="chip">{tx.get("word_count",0):,} words</span>'
    )
    sub = (f"Scoring how well this {source} content travels into "
           f"<b>{' and '.join(targets)}</b>. {opp_best} carries the largest audience "
           f"opportunity. Travel score and opportunity are scored separately below — "
           f"a hard-to-dub video can still be worth localising.")
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
        opp = s["opportunity_score"]
        bd = breakdown_bar(s["dub_quality_score"], s.get("penalties", {}), color)
        with col:
            st.markdown(
                f'<div class="card">'
                f'<div class="lang-row"><span class="lang-name">{lang}</span>'
                f'{grade_pill(s["grade"])}</div>'
                f'<div class="ring-wrap">'
                f'<div style="text-align:center"><div class="opp-label">Travel score</div>'
                f'{score_ring(s["dub_quality_score"], color)}</div>'
                f'<div class="opp"><div class="opp-label">Audience opportunity</div>'
                f'<div class="opp-num">{opp}<span style="font-size:.8rem;color:#94a3b8">/100</span></div>'
                f'<div class="opp-track"><div class="opp-fill" style="width:{opp}%"></div></div>'
                f'</div></div>'
                f'<div class="reco">↳ {html.escape(s["recommendation"])}</div>'
                f'{bd}</div>',
                unsafe_allow_html=True,
            )


def render_risk_profile(res):
    st.markdown('<div class="eyebrow">Risk profile</div>', unsafe_allow_html=True)
    st.caption("The five signals behind the score. Semantic loss and cultural risk "
               "are shown for the worst-affected target language.")
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
        fg, _ = _LEVEL_STYLE.get(level, ("#475569", "#e2e8f0"))
        html_rows += (
            f'<div class="rp-row" style="border-left-color:{fg}">'
            f'<div class="rp-name">{html.escape(name)}</div>'
            f'<div class="rp-val">{html.escape(str(val))}</div>'
            f'{level_chip(level)}'
            f'<div class="rp-help">{html.escape(help_text)}</div></div>'
        )
    st.markdown(html_rows, unsafe_allow_html=True)


def render_sample_dub(res, api_key):
    st.markdown('<div class="eyebrow">Hear it dubbed · sample</div>', unsafe_allow_html=True)
    st.caption("The opening, translated and voiced with Sarvam TTS — a tangible "
               "preview. Audio uses free Sarvam credits, so it's generated on click.")
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
                    with st.spinner(f"Synthesising {lang} with Sarvam TTS…"):
                        tts_cache[key] = dubber.synthesize(txt, lang, api_key)
                except dubber.DubError as e:
                    st.warning(f"Couldn't synthesise: {e}")
            if key in tts_cache:
                st.audio(tts_cache[key], format="audio/wav")

    st.markdown(
        "<div class='dub-note'>⚠️ <b>This is raw machine translation, not a finished "
        "dub.</b> It can <i>transliterate</i> English terms instead of finding natural "
        "equivalents — e.g. “cuticle” → “க்யூட்டிகில்”, which is English in Tamil "
        "letters, not real Tamil. That's a <b>translation-quality</b> problem, and it is "
        "deliberately <b>not</b> what the Travel Score measures. The score rates "
        "whether the source's meaning and structure are <i>worth</i> dubbing — it can't "
        "vouch for an auto-dub's fluency, because back-translation uses the same engine "
        "both ways and transliteration round-trips perfectly (so it reads as “clean”). "
        "A production dub needs Sarvam's Mayura translation + a native-speaker QA pass.</div>",
        unsafe_allow_html=True,
    )


def render_localization_gap(res):
    loc = res.get("localization", {})
    targets = [t for t in res["targets"] if t != "English"]
    if not any(loc.get("by_language", {}).get(t) for t in targets):
        return
    st.markdown('<div class="eyebrow">Localization · what MT gets right vs. wrong</div>',
                unsafe_allow_html=True)
    st.caption("For common English terms: what free machine translation does live, vs. "
               "the right call — localize to a native word, keep a fixed term in English "
               "(translating “baby oil” literally is wrong), or keep a naturalised "
               "loanword. ✓ = MT made the right call, ✗ = it didn't.")
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
                        f"<span class='lg-score'>MT correct: {n_ok}/{len(rows)}</span></div>",
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
                f"<table class='lg-table'><tr><th>Term</th><th>Free MT (now)</th>"
                f"<th>Right call</th><th></th></tr>{body}</table>", unsafe_allow_html=True)


def render_transcript(res):
    st.markdown('<div class="eyebrow">Transcript explorer</div>', unsafe_allow_html=True)
    idi, cul = res["idiomatic"], res["cultural"]
    idiom_phrases = [f["phrase"] for f in idi["found_idioms"]]
    risky_refs = (cul["top_risky_references"]
                  + cul["references_found"].get("audience_skewed", [])
                  + cul["references_found"].get("niche_everywhere", []))
    highlighted = highlight_transcript(res["transcript"]["transcript"],
                                       idiom_phrases, risky_refs)
    if idiom_phrases or risky_refs:
        st.markdown(
            "<div class='legend'><mark class='idiom'>idiom / slang</mark> &nbsp; "
            "<mark class='culture'>cultural reference</mark></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='legend empty'>No idioms or risky cultural "
                    "references detected to highlight.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tx-box'>{highlighted}</div>", unsafe_allow_html=True)

    worst = res["semantic"].get("worst_chunks", [])
    if worst:
        st.markdown("<div class='bd-head'>Where meaning slipped most "
                    "(back-translation)</div>", unsafe_allow_html=True)
        for w in worst[:3]:
            with st.expander(f"{w['language']} · {int(w['loss']*100)}% meaning lost"):
                st.markdown(
                    f"<div class='bt-orig'>Original: {html.escape(w['original'][:400])}</div>"
                    f"<div class='bt-back'>Round-tripped: {html.escape(w['back_translated'][:400])}</div>",
                    unsafe_allow_html=True)


def render_footer(res):
    cat = res["scores"]["detected_category"]
    bench = BENCHMARKS.get(cat, BENCHMARKS["General"])
    best = res["scores"]["quality_priority_order"][0]
    your = res["scores"]["by_language"][best]["dub_quality_score"]
    delta = your - bench["avg_score"]
    dcolor, arrow = ("#15803d", "▲") if delta >= 0 else ("#b91c1c", "▼")

    c1, c2 = st.columns([3, 1])
    with c1:
        with st.expander(f"📊  Benchmark — your {your} vs typical {cat} ({bench['avg_score']})"):
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
        with st.expander("🧮  How scores are calculated (methodology)"):
            st.markdown(
                """
The Travel Score starts at 100 and subtracts a capped penalty for six signals.
The numbers are the most each can take off:

| Signal | Max | What it measures |
|---|---|---|
| Semantic loss | 40 | Meaning lost on a translate then back-translate round trip. |
| Localization | 30 | Common terms machine translation mishandles for this language. |
| Idiomatic density | 20 | Slang and idioms that need adapting, not translating. |
| Cultural risk | 16 | References an audience may not recognise. |
| Structural interleave | 8 | Languages fused mid-sentence, with no clean seam to re-voice. |
| Prosody dependency | 8 | Meaning carried by delivery, not words (a text proxy). |

I added the **localization** signal after testing: the first five clustered every
clip at 85 to 95, because meaning round-trips fine for short, literal speech.
Localization catches the real failure, content full of English terms that get
transliterated instead of localized.

**Audience Opportunity** is a separate dimension and is never multiplied into the
score. A hard-to-dub clip can still be worth localising.

**What the score is, and isn't.** It rates whether a clip is *worth* dubbing, not
the fluency of one machine dub. Back-translation uses the same engine both ways,
so a transliteration like "cuticle" to "க்யூட்டிகில்" round-trips perfectly and
reads as clean. Output fluency needs a human pass (and Sarvam's Mayura
translation in production). The Localization panel above shows that gap directly.

**Honest limits.** Prosody is a text proxy. `langdetect` is weak on short text.
Idiom matching is tuned for Roman and code-mixed text, so for native-script
(Devanagari or Tamil) sources the score leans more on semantic loss and
localization. Back-translation uses free Google Translate and can rate-limit.
                """)
            st.caption(f"Idioms: {res['idiomatic']['dictionary_size']} · "
                       f"Cultural base: {res['cultural']['dictionary_size']} · "
                       f"STT: Sarvam saarika:v2.5 · TTS: Sarvam bulbul:v3 · "
                       f"Embeddings: MiniLM-L12-v2 · Dub text: Google Translate")
    with c2:
        st.markdown('<div class="eyebrow">Export</div>', unsafe_allow_html=True)
        st.download_button("⬇  Full report (JSON)",
                           data=json.dumps(res, ensure_ascii=False, indent=2, default=str),
                           file_name="dub_worthiness_report.json",
                           mime="application/json", use_container_width=True)


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


# --- App body ----------------------------------------------------------------
def main():
    inject_css()
    st.markdown('<div class="hero-title">Will It Travel?</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">How well does your Indian video/audio travel '
                'into other languages? Honest, explainable localisation scoring — '
                'not naïve code-mix percentage.</p>', unsafe_allow_html=True)

    api_key = get_api_key()

    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "Paste a YouTube link or upload audio/video. The tool transcribes it "
            "with **Sarvam STT**, then runs five scoring engines to estimate "
            "dubbability across Indian languages."
        )
        st.markdown("**Languages (any direction)**\n\n"
                    "Auto-detects the source — **English · Hindi · Tamil** — and "
                    "scores dubbing into the other two.")
        st.markdown(f"**API key:** {'✅ loaded' if api_key else '❌ missing'}")
        st.caption(f"Clips are trimmed to the first "
                   f"{extractor.MAX_DURATION_SECONDS // 60} minutes to stay within "
                   f"free Sarvam credits.")
        st.caption("Free stack: Sarvam free tier · yt-dlp · deep-translator · "
                   "sentence-transformers (local).")

    if not api_key:
        st.error(
            "**SARVAM_API_KEY not found.** Add it to `.streamlit/secrets.toml` "
            "for local dev, or in **Settings → Secrets** on Streamlit Cloud. "
            "Get a free key at https://dashboard.sarvam.ai ."
        )
        st.stop()

    # --- Input ---------------------------------------------------------------
    st.markdown("#### 1 · Choose content")
    tab_url, tab_file = st.tabs(["🔗  YouTube URL", "📁  Upload file"])
    with tab_url:
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…",
                            label_visibility="collapsed")
    with tab_file:
        upload = st.file_uploader("Upload MP3 / MP4 / WAV (max 200 MB)",
                                  type=["mp3", "mp4", "wav", "m4a", "webm"],
                                  label_visibility="collapsed")

    analyse = st.button("Analyse  →", type="primary")

    if analyse:
        if not url and not upload:
            st.warning("Paste a YouTube URL or upload a file first.")
            st.stop()

        bar = st.progress(0.0)
        status = st.empty()
        cb = _progress_cb_factory(bar, status)
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
            st.error(f"**Could not process this content.** {e}")
            st.stop()
        except Exception as e:  # noqa: BLE001 - surface a friendly message
            st.error("**Something went wrong during analysis.** Try a shorter or "
                     f"different clip. Details: {e}")
            st.stop()
        finally:
            bar.empty()
            status.empty()
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if "analysis" in st.session_state:
        st.divider()
        render_dashboard(st.session_state["analysis"], api_key)


if __name__ == "__main__":
    main()
