# 🎬 Dub Worthiness Score

Predicts how well Indian video/audio content will **localise into other Indian
languages** — with honest, explainable scoring instead of a naïve code-mixing
percentage.

Paste a YouTube URL or upload an MP3/MP4/WAV. The app transcribes it with the
**Sarvam STT API**, runs five scoring engines, and returns a per-language **Dub
Worthiness Score (0–100)**, a risk breakdown, a transcript explorer, and
language-priority recommendations.

**Works in any direction across English, Hindi, and Tamil** — it auto-detects the
source language and scores dubbing into the other two (English→Tamil,
Tamil→Hindi, Hindi→English, etc.).

> The core idea: code-mixing *percentage* is a useless signal. "Kal meeting hai"
> is 100% mixed yet trivially dubbable. What actually predicts dubbing pain is
> *meaning loss*, *idiomatic density*, *cultural distance*, *structural
> interleaving*, and *prosody dependence*. This tool measures those instead.

---

## How it scores

**Dub Quality (0–100)** — starts at 100, subtracts five weighted penalties:

| Engine | Weight | Measures |
|---|---|---|
| Semantic loss (back-translation) | 35% | Meaning lost on a translate→back-translate round-trip — the most honest signal. |
| Idiomatic density | 25% | Slang/idioms where surface meaning ≠ real meaning (574-entry dictionary). |
| Cultural reference risk | 20% | References an audience may not recognise (646-entry base, per-audience familiarity for English/Hindi/Tamil). |
| Structural interleaving | 10% | Languages fused mid-sentence — no clean seam to re-voice along. |
| Prosody dependency | 10% | Meaning carried by delivery, not words (text-based proxy). |

**Audience Opportunity** is scored **separately** and is *never multiplied* into
quality — a hard-to-dub video can still be worth localising if the audience
upside is large. The dashboard shows both side by side.

---

## Quickstart (local)

```bash
# 1. Create an isolated environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install ffmpeg (system dependency)
#    macOS:  brew install ffmpeg
#    Ubuntu: sudo apt-get install ffmpeg

# 3. Add your Sarvam API key (NEVER commit this file — it's git-ignored)
#    Edit .streamlit/secrets.toml:
#       SARVAM_API_KEY = "your_real_key_here"
#    Get a free key at https://dashboard.sarvam.ai

# 4. Run
streamlit run app.py
```

### Handling your API key securely
- **Local:** put it only in `.streamlit/secrets.toml` (git-ignored, lock it with
  `chmod 600 .streamlit/secrets.toml`). Or export `SARVAM_API_KEY` as an env var.
- **Deployment:** put it only in Streamlit Cloud → **Settings → Secrets**.
- Never hard-code it, never commit it. `.gitignore` already excludes the secrets
  file and `.env`.

---

## Architecture

```
dub-worthiness/
├── app.py                  # Streamlit dashboard (UI, caching, orchestration)
├── scorer.py               # Aggregates engine outputs → per-language scores
├── engines/
│   ├── extractor.py        # yt-dlp / upload → audio → Sarvam STT (chunked <30s)
│   ├── semantic.py         # back-translation round-trip + embedding similarity
│   ├── idiomatic.py        # Hinglish idiom dictionary matching
│   ├── cultural.py         # cultural-reference matching + per-audience risk
│   ├── structural.py       # sentence-level code-switch shape (script + langdetect)
│   ├── prosody.py          # speaking rate / emphasis / emotion (text proxy)
│   ├── opportunity.py      # category detection + audience-size affinity
│   └── textutils.py        # normalisation + Devanagari romanisation helpers
├── data/
│   ├── hinglish_idioms.json        # 574 curated idioms
│   ├── cultural_references.json    # 646 curated references
│   └── build_dictionaries.py       # reproducible merge/validate/dedupe build
├── requirements.txt        # Python deps (CPU-only torch pinned for cloud)
└── packages.txt            # system deps for Streamlit Cloud (ffmpeg)
```

### Notable engineering decisions
- **Chunked STT:** Sarvam's sync endpoint caps at ~30 s, so audio is split into
  <30 s windows, transcribed, and stitched — the naïve single-call approach fails
  on anything longer than a short clip.
- **`saarika:v2.5`, deliberately:** it transcribes faithfully, preserving
  code-switching. `saaras:v3` can normalise/translate the mix, which would erase
  the exact signal the structural and semantic engines measure.
- **CPU-only PyTorch pinned** (`--extra-index-url …/cpu`) to fit Streamlit Cloud's
  ~1 GB tier; the embedding model loads via `@st.cache_resource`.
- **Honest limitations are surfaced, not hidden:** prosody is a text proxy;
  `langdetect` is weak on short/Romanised text; dictionary matching favours
  Roman/code-mixed tokens. Each is labelled in the UI.

---

## Tech stack
Python · Streamlit · yt-dlp · Sarvam STT · deep-translator (free Google
Translate) · sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`,
local) · pydub + ffmpeg · NLTK · langdetect · indic-transliteration.

Everything runs on free tiers. No paid APIs.
