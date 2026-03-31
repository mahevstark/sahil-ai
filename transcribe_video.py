"""
transcribe_video.py — download a single YouTube video and transcribe it.

Usage:
    python transcribe_video.py <youtube_url>
    python transcribe_video.py <youtube_url> --model large-v3
    python transcribe_video.py <youtube_url> --output transcript.txt
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers: locate ffmpeg and yt-dlp regardless of PATH
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str | None:
    """Return path to ffmpeg.exe, or None if already on PATH."""
    if shutil.which("ffmpeg"):
        return None
    winget = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links"
    if winget.exists() and (winget / "ffmpeg.exe").exists():
        return str(winget / "ffmpeg.exe")
    pkg = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    for candidate in pkg.glob("Gyan.FFmpeg_*/ffmpeg-*-full_build/bin/ffmpeg.exe"):
        if candidate.exists():
            return str(candidate)
    return "ffmpeg"   # last resort: hope it's on PATH at call time


def _find_ytdlp() -> str:
    """Return path to yt-dlp executable."""
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        return exe
    for pkg_dir in (Path.home() / "AppData" / "Local" / "Packages").glob(
        "PythonSoftwareFoundation.Python.*"
    ):
        p = pkg_dir / "LocalCache" / "local-packages" / "Python313" / "Scripts" / "yt-dlp.exe"
        if p.exists():
            return str(p)
    raise FileNotFoundError("yt-dlp not found — run: pip install yt-dlp")


_FFMPEG = _find_ffmpeg()
_YTDLP  = _find_ytdlp()


# ---------------------------------------------------------------------------
# Step 1: Load Whisper model (must happen BEFORE any subprocess calls on
#         Windows Store Python to avoid CUDA DLL conflict)
# ---------------------------------------------------------------------------

def load_whisper(model_size: str):
    from faster_whisper import WhisperModel
    import ctranslate2

    try:
        device = "cuda" if ctranslate2.get_supported_compute_types("cuda") else "cpu"
    except Exception:
        device = "cpu"

    compute_type = "int8_float16" if device == "cuda" else "int8"
    print(f"Loading Whisper {model_size} on {device.upper()} ({compute_type}) ...", flush=True)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
# Step 2: Download audio via yt-dlp + ffmpeg
# ---------------------------------------------------------------------------

def download_audio(url: str, out_dir: Path) -> tuple[Path, str]:
    # Fetch metadata
    print("  Fetching video info ...", flush=True)
    r = subprocess.run(
        [_YTDLP, "--dump-json", "--no-playlist", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed:\n{r.stderr[:500]}")
    info = json.loads(r.stdout)
    title = info.get("title", "Unknown")

    # Download best audio in native format
    print("  Downloading audio ...", flush=True)
    r = subprocess.run(
        [_YTDLP, "-f", "bestaudio/best", "--no-playlist",
         "-o", str(out_dir / "audio_native.%(ext)s"), url],
        capture_output=True,
    )
    if r.returncode not in (0, 1):
        raise RuntimeError(
            f"yt-dlp download failed (exit {r.returncode})\n"
            f"{r.stderr.decode('utf-8', errors='replace')[-300:]}"
        )

    downloaded = next(out_dir.glob("audio_native.*"), None)
    if not downloaded:
        raise FileNotFoundError(f"yt-dlp produced no audio file in {out_dir}")

    # Convert to 16 kHz mono WAV
    print("  Converting to 16kHz mono WAV ...", flush=True)
    wav_path = out_dir / "audio.wav"
    r = subprocess.run(
        [_FFMPEG, "-y", "-i", str(downloaded), "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True,
    )
    downloaded.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode('utf-8', errors='replace')[-500:]}")

    return wav_path, title


# ---------------------------------------------------------------------------
# Step 3: Transcribe
# ---------------------------------------------------------------------------

def transcribe(model, wav_path: Path) -> list[dict]:
    print("Transcribing ...", flush=True)
    segments, info = model.transcribe(str(wav_path), beam_size=5)
    print(f"Detected language: {info.language} (confidence {info.language_probability:.0%})")
    print()

    results = []
    for seg in segments:
        results.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Transcribe a YouTube video using Whisper")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model", default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3 for best Urdu/mixed accuracy)",
    )
    parser.add_argument("--output", default=None, help="Save transcript to this .txt file")
    args = parser.parse_args()

    # Load model FIRST — must happen before any subprocess calls on Windows
    model = load_whisper(args.model)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"\nDownloading: {args.url}")
        wav_path, title = download_audio(args.url, Path(tmp))
        print(f"Title: {title}\n")

        segments = transcribe(model, wav_path)

    lines = [
        f"[{format_time(s['start'])} --> {format_time(s['end'])}]  {s['text']}"
        for s in segments
    ]
    transcript = "\n".join(lines)
    print(transcript)

    if args.output:
        Path(args.output).write_text(
            f"Title: {title}\nURL: {args.url}\n\n{transcript}\n", encoding="utf-8"
        )
        print(f"\nSaved to: {Path(args.output).resolve()}")
    else:
        print(f"\n--- {len(segments)} segments | {title} ---")


if __name__ == "__main__":
    main()
    # ctranslate2's CUDA cleanup calls exit(127) during Python shutdown on Windows;
    # os._exit(0) bypasses that cleanup and returns a clean exit code.
    import os as _os
    _os._exit(0)
