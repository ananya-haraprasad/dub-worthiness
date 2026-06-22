"""Engine 1 — Content Extraction.

Pipeline:
  1. Acquire audio from a YouTube URL (yt-dlp) OR a user-uploaded file.
  2. Normalise to mono 16 kHz and trim to MAX_DURATION_SECONDS.
  3. Transcribe with Sarvam STT.
  4. Always clean up temp files (try/finally).

WHY WE CHUNK (and don't send the whole clip in one call, as a naive impl would):
  Sarvam's *synchronous* /speech-to-text endpoint only accepts clips under 30 s.
  A single 8-minute request would return HTTP 400. So we split the audio into
  <30 s windows, transcribe each, and stitch the transcripts back together. This
  keeps us on the free sync endpoint — the Batch API supports long files but
  requires object-storage upload + job polling, which is overkill here.

WHY saarika:v2.5 (not saaras:v3):
  saarika is a faithful ASR model: it transcribes what was actually said,
  preserving code-switching as spoken. saaras:v3 can translate / normalise the
  mix, which would erase the very code-switching signal the structural and
  semantic engines exist to measure. So the "older" model is the correct one
  for this product. Switching is a one-line change to STT_MODEL.
"""
from __future__ import annotations

import glob
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from pydub import AudioSegment

# --- Config -------------------------------------------------------------------
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
STT_MODEL = "saarika:v2.5"
MAX_DURATION_SECONDS = 8 * 60       # cap input to protect free Sarvam credits
CHUNK_MS = 29_000                   # just under the 30 s sync limit
TARGET_SAMPLE_RATE = 16_000         # standard ASR rate; keeps uploads small
REQUEST_TIMEOUT = 90                # per-chunk HTTP timeout (seconds)
INTER_CALL_SLEEP = 0.5              # politeness gap between chunk calls
MAX_RETRIES = 3                     # retry transient 429 / 5xx with backoff

ProgressCb = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


class ExtractionError(Exception):
    """User-facing extraction/transcription failure."""


@dataclass
class TranscriptResult:
    transcript: str
    language_code: str
    language_distribution: dict          # {"hi-IN": 7, "en-IN": 3}
    word_count: int
    duration_seconds: float
    num_chunks: int
    chunks: list = field(default_factory=list)  # [{idx, start_s, text, lang}]

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "language_code": self.language_code,
            "language_distribution": self.language_distribution,
            "word_count": self.word_count,
            "duration_seconds": round(self.duration_seconds, 1),
            "num_chunks": self.num_chunks,
            "chunks": self.chunks,
        }


# --- Audio acquisition --------------------------------------------------------
def _download_youtube_audio(url: str, workdir: str) -> str:
    """Download best audio track from a YouTube URL into workdir. Returns path."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("yt-dlp is not installed.") from exc

    out_tmpl = os.path.join(workdir, "yt_audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 4,
        "fragment_retries": 4,
        # YouTube intermittently 403s the default web client. Trying the mobile
        # clients first bypasses most of those blocks. (Let pydub/ffmpeg do the
        # final conversion, so no double-encode here.)
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise ExtractionError(
            "Could not download that video. YouTube sometimes blocks automated "
            "downloads (a 403); try again, try a different link, or use the upload "
            "option. It may also be private, age-restricted, or region-locked."
        ) from exc

    files = [f for f in glob.glob(os.path.join(workdir, "yt_audio.*"))]
    if not files:
        raise ExtractionError("Download produced no audio file.")
    return files[0]


def _load_and_normalise(path: str) -> tuple[AudioSegment, float]:
    """Load any audio/video file, downmix to mono 16 kHz, trim to the cap."""
    try:
        audio = AudioSegment.from_file(path)
    except Exception as exc:
        raise ExtractionError(
            "Could not read the audio. If you uploaded a video, make sure it "
            "has an audio track. Supported: MP3, WAV, MP4, M4A, WebM."
        ) from exc

    original_seconds = len(audio) / 1000.0
    if original_seconds > MAX_DURATION_SECONDS:
        audio = audio[: MAX_DURATION_SECONDS * 1000]

    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    return audio, len(audio) / 1000.0


# --- Transcription ------------------------------------------------------------
def _transcribe_chunk(chunk_path: str, api_key: str) -> dict:
    """Transcribe a single <30 s chunk, retrying transient failures."""
    headers = {"api-subscription-key": api_key}
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            with open(chunk_path, "rb") as f:
                files = {"file": ("chunk.mp3", f, "audio/mpeg")}
                # language_code="unknown" => Sarvam auto-detects per chunk,
                # which also lets us observe code-switching across the clip.
                data = {"model": STT_MODEL, "language_code": "unknown"}
                resp = requests.post(
                    SARVAM_STT_URL, headers=headers, files=files,
                    data=data, timeout=REQUEST_TIMEOUT,
                )

            if resp.status_code == 429 or resp.status_code >= 500:
                # Transient — back off and retry.
                time.sleep(1.5 * (attempt + 1))
                last_exc = ExtractionError(f"Sarvam returned {resp.status_code}")
                continue

            if resp.status_code == 401 or resp.status_code == 403:
                raise ExtractionError(
                    "Sarvam rejected the API key (401/403). Check that the key "
                    "is correct and has STT access / remaining credits."
                )

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))

    raise ExtractionError(
        f"Transcription failed after {MAX_RETRIES} attempts: {last_exc}"
    )


def transcribe(audio: AudioSegment, duration_seconds: float, api_key: str,
               progress_cb: ProgressCb = _noop) -> TranscriptResult:
    """Chunk the audio and transcribe each chunk via Sarvam, then stitch."""
    if not api_key:
        raise ExtractionError("No Sarvam API key provided.")

    n_chunks = max(1, (len(audio) + CHUNK_MS - 1) // CHUNK_MS)
    texts: list[str] = []
    chunk_meta: list[dict] = []
    lang_dist: dict[str, int] = {}
    lang_prob_weight: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_chunks):
            start_ms = i * CHUNK_MS
            piece = audio[start_ms: start_ms + CHUNK_MS]
            if len(piece) < 200:  # skip sub-0.2s tail fragments
                continue

            chunk_path = os.path.join(tmp, f"chunk_{i}.mp3")
            piece.export(chunk_path, format="mp3", bitrate="64k")

            progress_cb(
                0.05 + 0.45 * (i / n_chunks),
                f"Transcribing with Sarvam… chunk {i + 1} of {n_chunks}",
            )
            result = _transcribe_chunk(chunk_path, api_key)

            text = (result.get("transcript") or "").strip()
            lang = result.get("language_code") or "unknown"
            prob = result.get("language_probability") or 0.0
            if text:
                texts.append(text)
            chunk_meta.append({
                "idx": i,
                "start_s": round(start_ms / 1000.0, 1),
                "text": text,
                "lang": lang,
            })
            if lang and lang != "unknown":
                lang_dist[lang] = lang_dist.get(lang, 0) + 1
                lang_prob_weight[lang] = lang_prob_weight.get(lang, 0.0) + (prob or 1.0)

            time.sleep(INTER_CALL_SLEEP)

    full = " ".join(texts).strip()
    if not full:
        raise ExtractionError(
            "Transcription returned no text. The clip may be silent, music-only, "
            "or in a language Sarvam could not detect."
        )

    dominant = (
        max(lang_prob_weight, key=lang_prob_weight.get)
        if lang_prob_weight else "unknown"
    )
    return TranscriptResult(
        transcript=full,
        language_code=dominant,
        language_distribution=lang_dist,
        word_count=len(full.split()),
        duration_seconds=duration_seconds,
        num_chunks=len([c for c in chunk_meta if c["text"]]),
        chunks=chunk_meta,
    )


# --- Orchestrator -------------------------------------------------------------
def extract_and_transcribe(api_key: str, youtube_url: Optional[str] = None,
                           uploaded_path: Optional[str] = None,
                           progress_cb: ProgressCb = _noop) -> TranscriptResult:
    """End-to-end: acquire -> normalise -> transcribe. Cleans up temp files."""
    if not (youtube_url or uploaded_path):
        raise ExtractionError("Provide a YouTube URL or an uploaded file.")

    workdir = tempfile.mkdtemp(prefix="dubworth_")
    try:
        if youtube_url:
            progress_cb(0.02, "Downloading audio from YouTube…")
            src = _download_youtube_audio(youtube_url, workdir)
        else:
            src = uploaded_path  # app already wrote the upload to disk

        progress_cb(0.04, "Preparing audio…")
        audio, duration = _load_and_normalise(src)

        return transcribe(audio, duration, api_key, progress_cb=progress_cb)
    finally:
        # Best-effort cleanup of the working directory.
        try:
            for f in glob.glob(os.path.join(workdir, "*")):
                os.remove(f)
            os.rmdir(workdir)
        except OSError:
            pass


# --- Manual test harness ------------------------------------------------------
if __name__ == "__main__":
    import sys
    key = os.getenv("SARVAM_API_KEY", "")
    url = sys.argv[1] if len(sys.argv) > 1 else None
    if not key or not url:
        print("Usage: SARVAM_API_KEY=... python -m engines.extractor <youtube_url>")
        raise SystemExit(1)

    def _cli(frac, msg):
        print(f"[{frac*100:5.1f}%] {msg}")

    res = extract_and_transcribe(key, youtube_url=url, progress_cb=_cli)
    print("\n--- RESULT ---")
    print("language:", res.language_code, "| dist:", res.language_distribution)
    print("duration:", res.duration_seconds, "s | words:", res.word_count,
          "| chunks:", res.num_chunks)
    print("\ntranscript (first 600 chars):\n", res.transcript[:600])
