"""
transcribe.py — interactive single-video transcriber.

Just run:
    python transcribe.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Locate ffmpeg / yt-dlp (same logic as transcribe_video.py)
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    winget = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links"
    if (winget / "ffmpeg.exe").exists():
        return str(winget / "ffmpeg.exe")
    for p in (Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages").glob(
        "Gyan.FFmpeg_*/ffmpeg-*-full_build/bin/ffmpeg.exe"
    ):
        if p.exists():
            return str(p)
    return "ffmpeg"


def _find_ytdlp() -> str:
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        return exe
    for pkg in (Path.home() / "AppData" / "Local" / "Packages").glob(
        "PythonSoftwareFoundation.Python.*"
    ):
        p = pkg / "LocalCache" / "local-packages" / "Python313" / "Scripts" / "yt-dlp.exe"
        if p.exists():
            return str(p)
    raise FileNotFoundError("yt-dlp not found — run: pip install yt-dlp")


_FFMPEG = _find_ffmpeg()
_YTDLP  = _find_ytdlp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(text: str) -> str:
    """Turn a video title into a safe filename."""
    safe = re.sub(r'[\\/*?:"<>|]', "", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:80] or "transcript"


def fetch_title(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    r = subprocess.run(
        [_YTDLP, "--dump-json", "--no-playlist", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return video_id
    try:
        return json.loads(r.stdout).get("title", video_id)
    except Exception:
        return video_id


def load_whisper(model_size: str):
    from faster_whisper import WhisperModel
    import ctranslate2
    try:
        device = "cuda" if ctranslate2.get_supported_compute_types("cuda") else "cpu"
    except Exception:
        device = "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"
    print(f"  Loading Whisper {model_size} on {device.upper()} ({compute_type}) ...", flush=True)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def download_audio(video_id: str, out_dir: Path) -> Path:
    url = f"https://www.youtube.com/watch?v={video_id}"
    print("  Downloading audio ...", flush=True)
    r = subprocess.run(
        [_YTDLP, "-f", "bestaudio/best", "--no-playlist",
         "-o", str(out_dir / "audio_native.%(ext)s"), url],
        capture_output=True,
    )
    if r.returncode not in (0, 1):
        raise RuntimeError(f"yt-dlp failed (exit {r.returncode})")

    downloaded = next(out_dir.glob("audio_native.*"), None)
    if not downloaded:
        raise FileNotFoundError("yt-dlp produced no audio file")

    print("  Converting to 16kHz mono WAV ...", flush=True)
    wav = out_dir / "audio.wav"
    r = subprocess.run(
        [_FFMPEG, "-y", "-i", str(downloaded), "-ar", "16000", "-ac", "1", str(wav)],
        capture_output=True,
    )
    downloaded.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode('utf-8', errors='replace')[-300:]}")
    return wav


def transcribe(model, wav_path: Path) -> list[dict]:
    print("  Transcribing ...", flush=True)
    segments, info = model.transcribe(str(wav_path), beam_size=5)
    print(f"  Language: {info.language}  (confidence {info.language_probability:.0%})")
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def format_time(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}" if h else f"{int(m):02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------

def main():
    import questionary

    print("\n── YouTube Transcriber ─────────────────────────────\n")

    # 1. Video ID
    video_id = questionary.text(
        "YouTube video ID:",
        validate=lambda v: True if v.strip() else "Please enter a video ID",
    ).ask()
    if not video_id:
        sys.exit(0)
    video_id = video_id.strip()

    # 2. Model selection
    model_size = questionary.select(
        "Whisper model:",
        choices=[
            questionary.Choice("large-v3  — best accuracy, slower  (recommended for Urdu/mixed)", value="large-v3"),
            questionary.Choice("medium    — good balance", value="medium"),
            questionary.Choice("small     — fast", value="small"),
            questionary.Choice("base      — fastest, least accurate", value="base"),
            questionary.Choice("tiny      — ultra fast", value="tiny"),
        ],
        default="large-v3",
    ).ask()
    if not model_size:
        sys.exit(0)

    # 3. Load model FIRST (must happen before any subprocess calls on Windows)
    print()
    model = load_whisper(model_size)

    # 4. Fetch title to suggest a default output filename
    print("  Fetching video title ...", flush=True)
    title = fetch_title(video_id)
    default_output = f"{safe_filename(title)}.txt"

    # 5. Output filename
    print()
    output_file = questionary.text(
        "Output file:",
        default=default_output,
    ).ask()
    if not output_file:
        output_file = default_output
    output_file = output_file.strip()

    # 6. Run pipeline
    print(f"\nTitle : {title}")
    print(f"Output: {output_file}\n")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = download_audio(video_id, Path(tmp))
        segments = transcribe(model, wav_path)

    # Build transcript
    lines = [
        f"[{format_time(s['start'])} --> {format_time(s['end'])}]  {s['text']}"
        for s in segments
    ]
    transcript = "\n".join(lines)

    Path(output_file).write_text(
        f"Title: {title}\nURL: https://www.youtube.com/watch?v={video_id}\n\n{transcript}\n",
        encoding="utf-8",
    )

    print(f"\n✓ {len(segments)} segments saved to: {Path(output_file).resolve()}\n")


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
