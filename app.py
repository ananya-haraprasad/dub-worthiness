# NLTK data must be present before any engine that tokenizes runs. Newer NLTK
# uses 'punkt_tab'; download both. Kept at the very top, before other imports.
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import hashlib
import html
import json
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
)
import scorer

# --- Constants ---------------------------------------------------------------
# Source language is auto-detected (English / Hindi / Tamil); targets are the
# other two. Computed per analysis, so there is no fixed target list here.

GRADE_COLORS = {
    "Dubs cleanly": "#15803d",
    "Light adaptation needed": "#a16207",
    "Heavy localisation": "#c2410c",
    "Not recommended": "#b91c1c",
    "—": "#6b7280",
}

# Hardcoded benchmark context so a single score feels grounded, not arbitrary.
BENCHMARKS = {
    "Education":            {"avg_score": 82, "note": "Dubs well — structured speech, clear concepts."},
    "Finance":             {"avg_score": 78, "note": "Mostly clean but jargon-heavy."},
    "Tech":                {"avg_score": 71, "note": "English-heavy, but a fast-growing regional audience."},
    "Entertainment":       {"avg_score": 54, "note": "Timing and slang make this hard."},
    "News":                {"avg_score": 68, "note": "Translates well, but cultural refs are risky."},
    "Lifestyle":           {"avg_score": 66, "note": "Casual register; some slang to adapt."},
    "General":             {"avg_score": 70, "note": "Mixed-genre baseline."},
}

st.set_page_config(page_title="Dub Worthiness Score", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")


# --- Secrets / API key (works locally AND on Streamlit Cloud) ----------------
# Treat the shipped placeholder values as "no key", so the UI never claims a key
# it doesn't actually have.
_PLACEHOLDER_KEYS = {"", "paste_your_key_here", "your_key_here", "none", "null"}


def get_api_key() -> str:
    try:
        key = st.secrets["SARVAM_API_KEY"]
    except Exception:
        key = os.getenv("SARVAM_API_KEY", "")
    key = (key or "").strip()
    return "" if key.lower() in _PLACEHOLDER_KEYS else key


# --- Cached heavy resource ---------------------------------------------------
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
        :root { --ink:#0f172a; --muted:#64748b; --line:#e5e7eb; --brand:#4f46e5; }
        .block-container { max-width: 1180px; padding-top: 2.2rem; }
        h1,h2,h3 { color: var(--ink); letter-spacing:-0.01em; }
        .hero-title { font-size:2.5rem; font-weight:800; margin-bottom:.1rem;
                      background:linear-gradient(90deg,#4f46e5,#0ea5e9);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .hero-sub { color:var(--muted); font-size:1.06rem; margin-top:0; }
        .pill { display:inline-block; padding:3px 10px; border-radius:999px;
                font-size:.74rem; font-weight:700; color:#fff; }
        .card { border:1px solid var(--line); border-radius:16px; padding:22px 24px;
                background:#fff; box-shadow:0 1px 2px rgba(16,24,40,.04); height:100%; }
        .score-num { font-size:3.4rem; font-weight:800; line-height:1; }
        .score-cap { color:var(--muted); font-size:.8rem; text-transform:uppercase;
                     letter-spacing:.06em; font-weight:700; }
        .lang-name { font-size:1.35rem; font-weight:800; color:var(--ink); }
        .metric-card { border:1px solid var(--line); border-radius:14px; padding:16px 18px;
                       background:#fff; height:100%; }
        .metric-label { color:var(--muted); font-size:.78rem; font-weight:700;
                        text-transform:uppercase; letter-spacing:.05em; }
        .metric-value { font-size:1.55rem; font-weight:800; color:var(--ink); margin:.15rem 0; }
        .metric-level { font-size:.82rem; font-weight:700; }
        .metric-help { color:var(--muted); font-size:.8rem; margin-top:.35rem; line-height:1.35; }
        .risk-line { font-size:.92rem; color:#334155; margin:.18rem 0; }
        mark.idiom { background:#fde68a; padding:0 2px; border-radius:3px; }
        mark.culture { background:#fecaca; padding:0 2px; border-radius:3px; }
        .tx-box { border:1px solid var(--line); border-radius:12px; padding:16px 18px;
                  background:#fafafa; max-height:340px; overflow-y:auto; line-height:1.7;
                  font-size:.95rem; }
        .legend { font-size:.8rem; color:var(--muted); }
        .bt-orig { color:#475569; } .bt-back { color:#0f172a; font-weight:600; }
        .small-muted { color:var(--muted); font-size:.85rem; }
        [data-testid="stMetricValue"] { font-size:1.5rem; font-weight:800; }
        [data-testid="stMetricLabel"] p { font-size:.74rem; font-weight:700;
            text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def grade_pill(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#6b7280")
    return f'<span class="pill" style="background:{color}">{html.escape(grade)}</span>'


def level_color(level: str) -> str:
    return {"low": "#15803d", "medium": "#a16207", "high": "#b91c1c",
            "unknown": "#6b7280", "undetermined": "#6b7280",
            "clean": "#15803d", "mixed": "#a16207", "interleaved": "#b91c1c"}.get(level, "#334155")


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

    # 1) Transcription — reuse if we've already transcribed this exact source
    #    (avoids re-spending Sarvam credits on a repeat run).
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
    # Auto-detect source language; targets are the other two languages.
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

    # 3) Semantic back-translation (slow; cache by transcript hash + targets)
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
    cb(0.94, "Computing Dub Worthiness Scores…")
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
def render_summary(res):
    tx = res["transcript"]
    cat = res["scores"]["detected_category"]
    pr = res["prosody"]
    source = res["source_lang"]
    targets = res["targets"]
    st.subheader("Executive summary")
    c = st.columns(5)
    dur = tx.get("duration_seconds") or 0
    c[0].metric("Source", source)
    c[1].metric("Duration", f"{int(dur // 60)}:{int(dur % 60):02d}")
    c[2].metric("Words", f"{tx.get('word_count', 0):,}")
    c[3].metric("WPM", f"{pr['speaking_rate_wpm']:.0f}")
    c[4].metric("Type", cat)

    best = res["scores"]["quality_priority_order"][0]
    best_score = res["scores"]["by_language"][best]["dub_quality_score"]
    opp_best = res["scores"]["opportunity_priority_order"][0]
    tgt_str = " and ".join(targets)
    st.markdown(
        f"<p class='small-muted'>Detected as <b>{source}</b> content "
        f"(<b>{html.escape(cat)}</b> genre); scoring how well it dubs into "
        f"<b>{tgt_str}</b>. It dubs most cleanly into <b>{best}</b> "
        f"(quality {best_score}/100), while <b>{opp_best}</b> carries the largest "
        f"audience opportunity. Quality and opportunity are scored separately below "
        f"— a hard-to-dub video can still be worth localising if the upside is "
        f"large.</p>",
        unsafe_allow_html=True,
    )


def render_scorecards(res):
    st.subheader("Language scorecard")
    st.caption("Dub Quality (how faithfully it survives localisation) and "
               "Audience Opportunity (how big the prize is) are independent — "
               "never multiplied.")
    targets = res["targets"]
    cols = st.columns(len(targets))
    for col, lang in zip(cols, targets):
        s = res["scores"]["by_language"][lang]
        color = GRADE_COLORS.get(s["grade"], "#6b7280")
        risks = "".join(f"<div class='risk-line'>• {html.escape(r)}</div>"
                        for r in s["top_risks"][:3])
        with col:
            st.markdown(
                f"""
                <div class="card">
                  <div class="lang-name">{lang}</div>
                  <div style="display:flex;align-items:flex-end;gap:18px;margin:.5rem 0;">
                    <div>
                      <div class="score-cap">Dub quality</div>
                      <div class="score-num" style="color:{color}">{s['dub_quality_score']}</div>
                    </div>
                    <div style="padding-bottom:.5rem">
                      <div class="score-cap">Opportunity</div>
                      <div style="font-size:1.7rem;font-weight:800;color:#0ea5e9">{s['opportunity_score']}</div>
                    </div>
                  </div>
                  <div style="margin:.3rem 0 .8rem">{grade_pill(s['grade'])}</div>
                  <div class="score-cap" style="margin-bottom:.25rem">Top risks</div>
                  {risks}
                  <div style="margin-top:.9rem;padding-top:.7rem;border-top:1px dashed var(--line);
                              font-weight:600;color:#1e293b">↳ {html.escape(s['recommendation'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_benchmark(res):
    cat = res["scores"]["detected_category"]
    bench = BENCHMARKS.get(cat, BENCHMARKS["General"])
    best = res["scores"]["quality_priority_order"][0]
    your = res["scores"]["by_language"][best]["dub_quality_score"]
    delta = your - bench["avg_score"]
    arrow = "▲" if delta >= 0 else "▼"
    dcolor = "#15803d" if delta >= 0 else "#b91c1c"
    st.subheader("Benchmark context")
    st.markdown(
        f"<p class='small-muted'>Your best-language score is <b>{your}</b>. "
        f"Typical <b>{html.escape(cat)}</b> content averages <b>{bench['avg_score']}</b> "
        f"<span style='color:{dcolor};font-weight:700'>({arrow} {abs(delta)})</span>. "
        f"{html.escape(bench['note'])}</p>",
        unsafe_allow_html=True,
    )
    rows = "".join(
        f"<tr style='background:{'#eef2ff' if k==cat else 'transparent'}'>"
        f"<td style='padding:6px 12px'>{html.escape(k)}</td>"
        f"<td style='padding:6px 12px;font-weight:700'>{v['avg_score']}</td>"
        f"<td style='padding:6px 12px;color:#64748b'>{html.escape(v['note'])}</td></tr>"
        for k, v in BENCHMARKS.items() if k != "General"
    )
    st.markdown(
        f"<table style='border-collapse:collapse;font-size:.88rem;width:100%'>"
        f"<tr style='text-align:left;color:#64748b;border-bottom:1px solid #e5e7eb'>"
        f"<th style='padding:6px 12px'>Category</th><th style='padding:6px 12px'>Avg</th>"
        f"<th style='padding:6px 12px'>Note</th></tr>{rows}</table>",
        unsafe_allow_html=True,
    )


def _metric_card(label, value, level, help_text):
    color = level_color(level)
    return (
        f"<div class='metric-card'><div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(str(value))}</div>"
        f"<div class='metric-level' style='color:{color}'>{html.escape(level.upper())}</div>"
        f"<div class='metric-help'>{html.escape(help_text)}</div></div>"
    )


def render_risk_cards(res):
    st.subheader("Risk deep-dive")
    targets = res["targets"]
    sem = res["semantic"]["by_language"]
    worst_lang = max(targets, key=lambda l: (sem.get(l, {}).get("loss") or 0))
    loss = sem.get(worst_lang, {}).get("loss")
    loss_str = f"{loss*100:.0f}%" if loss is not None else "n/a"
    loss_level = ("high" if (loss or 0) >= 0.25 else
                  "medium" if (loss or 0) >= 0.12 else "low")

    idi = res["idiomatic"]
    cul = res["cultural"]
    pr = res["prosody"]
    stc = res["structural"]
    cul_worst = max(targets,
                    key=lambda l: {"low": 0, "medium": 1, "high": 2}
                    .get(cul["risk_by_language"].get(l, "low"), 0))

    cards = [
        _metric_card("Semantic loss", loss_str, loss_level,
                     f"Meaning lost on a round-trip ({worst_lang}, worst case). "
                     f"The most honest dubbability signal."),
        _metric_card("Idiomatic density", f"{idi['idiom_density']:.1f}/100w",
                     idi["density_level"],
                     f"{idi['unique_idioms']} slang phrase(s) needing adaptation, "
                     f"not literal translation."),
        _metric_card("Cultural risk", f"{cul['total_references']} refs",
                     cul["risk_by_language"].get(cul_worst, "low"),
                     f"References an audience may not recognise ({cul_worst} most "
                     f"affected)."),
        _metric_card("Code-switch type", stc["switch_type"].title(),
                     stc["switch_type"],
                     f"{int(stc['interleave_ratio']*100)}% of clauses fuse "
                     f"languages mid-sentence."),
        _metric_card("Prosody dependency", pr["prosody_dependency"].title(),
                     pr["prosody_dependency"],
                     "How much meaning rides on delivery vs words (text proxy)."),
    ]
    cols = st.columns(5)
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def render_transcript(res):
    st.subheader("Transcript explorer")
    idi = res["idiomatic"]
    cul = res["cultural"]
    idiom_phrases = [f["phrase"] for f in idi["found_idioms"]]
    risky_refs = cul["top_risky_references"] + \
        cul["references_found"].get("audience_skewed", []) + \
        cul["references_found"].get("niche_everywhere", [])
    highlighted = highlight_transcript(res["transcript"]["transcript"],
                                       idiom_phrases, risky_refs)
    st.markdown(
        "<span class='legend'><mark class='idiom'>idiom / slang</mark> &nbsp; "
        "<mark class='culture'>cultural reference</mark></span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='tx-box'>{highlighted}</div>", unsafe_allow_html=True)

    worst = res["semantic"].get("worst_chunks", [])
    if worst:
        st.markdown("**Where meaning slipped most (back-translation):**")
        for w in worst[:3]:
            with st.expander(f"{w['language']} · {int(w['loss']*100)}% meaning lost"):
                st.markdown(f"<div class='bt-orig'>Original: "
                            f"{html.escape(w['original'][:400])}</div>"
                            f"<div class='bt-back'>Round-tripped: "
                            f"{html.escape(w['back_translated'][:400])}</div>",
                            unsafe_allow_html=True)


def render_export(res):
    st.subheader("Export")
    payload = json.dumps(res, ensure_ascii=False, indent=2, default=str)
    st.download_button("⬇  Download full report (JSON)", data=payload,
                       file_name="dub_worthiness_report.json",
                       mime="application/json")


def render_methodology(res):
    with st.expander("How scores are calculated (methodology)"):
        st.markdown(
            """
**Dub Quality (0–100)** starts at 100 and subtracts five weighted penalties:

| Engine | Weight | What it measures |
|---|---|---|
| Semantic loss | 35% | Meaning lost on a translate→back-translate round-trip (most honest signal). |
| Idiomatic density | 25% | Slang/idioms where surface meaning ≠ real meaning; need adaptation. |
| Cultural risk | 20% | References an audience may not recognise (per-language familiarity). |
| Structural interleave | 10% | Languages fused mid-sentence — no clean seam to re-voice. |
| Prosody dependency | 10% | Meaning carried by delivery, not words (**text-based proxy only**). |

**Audience Opportunity** is a *separate* dimension (TRAI/IAMAI-style internet-
penetration estimates by category × language) and is **never multiplied** into
quality — a hard-to-dub video can still be worth localising.

**Honest limitations:** prosody is inferred from text, not audio. `langdetect`
is unreliable on short or Romanised text. Idiom/cultural matching is
dictionary-based (574 idioms, 646 references) and strongest on Roman/code-mixed
tokens. Back-translation uses free Google Translate and may rate-limit.
            """
        )
        st.caption(f"Idiom dictionary: {res['idiomatic']['dictionary_size']} entries · "
                   f"Cultural base: {res['cultural']['dictionary_size']} entries · "
                   f"STT: Sarvam saarika:v2.5 · Embeddings: paraphrase-multilingual-MiniLM-L12-v2")


def render_dashboard(res):
    render_summary(res)
    st.divider()
    render_scorecards(res)
    st.divider()
    render_benchmark(res)
    st.divider()
    render_risk_cards(res)
    st.divider()
    render_transcript(res)
    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        render_export(res)
    render_methodology(res)


# --- App body ----------------------------------------------------------------
def main():
    inject_css()
    st.markdown('<div class="hero-title">Dub Worthiness Score</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Predicts how well Indian video/audio will '
                'localise into other Indian languages — honest, explainable '
                'scoring, not naïve code-mix percentage.</p>',
                unsafe_allow_html=True)

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

    analyse = st.button("Analyse  →", type="primary", use_container_width=False)

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
        render_dashboard(st.session_state["analysis"])


if __name__ == "__main__":
    main()
