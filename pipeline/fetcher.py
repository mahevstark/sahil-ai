import datetime
import shutil
import sys
from pathlib import Path

import yt_dlp


def _find_ytdlp() -> str:
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        return exe
    # Windows-only fallback: Microsoft Store Python puts scripts in AppData
    if sys.platform == "win32":
        for pkg in (Path.home() / "AppData" / "Local" / "Packages").glob(
            "PythonSoftwareFoundation.Python.*"
        ):
            p = pkg / "LocalCache" / "local-packages" / "Python313" / "Scripts" / "yt-dlp.exe"
            if p.exists():
                return str(p)
    raise FileNotFoundError("yt-dlp not found — run: pip install yt-dlp")


_YTDLP = _find_ytdlp()


def list_channel_videos(channel_url: str) -> list:
    ydl_opts = {
        "flat_playlist": True,
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    entries = info.get("entries", [])
    videos = []
    for entry in entries:
        if entry is None:
            continue

        raw_date = entry.get("upload_date")
        if raw_date:
            uploaded_at = datetime.datetime.strptime(raw_date, "%Y%m%d").date()
        else:
            uploaded_at = None

        videos.append(
            {
                "video_id": entry.get("id"),
                "title": entry.get("title"),
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
                "channel": entry.get("channel") or info.get("channel") or info.get("uploader"),
                "duration": entry.get("duration"),
                "uploaded_at": uploaded_at,
            }
        )

    return videos


def fetch_video_meta(video_id: str) -> dict:
    """Fetch title/channel/duration/date for a single video via yt-dlp."""
    import json
    import subprocess

    url = f"https://www.youtube.com/watch?v={video_id}"
    fallback = {"video_id": video_id, "title": video_id, "channel": None,
                "url": url, "duration": None, "uploaded_at": None}
    r = subprocess.run(
        [_YTDLP, "--dump-json", "--no-playlist",
         "--extractor-args", "youtube:player_client=android_vr", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return fallback
    try:
        data = json.loads(r.stdout)
        raw  = data.get("upload_date")
        return {
            "video_id":    video_id,
            "title":       data.get("title", video_id),
            "channel":     data.get("channel") or data.get("uploader"),
            "url":         url,
            "duration":    data.get("duration"),
            "uploaded_at": datetime.datetime.strptime(raw, "%Y%m%d").date() if raw else None,
        }
    except Exception:
        return fallback


def download_audio(video_id: str, output_dir: Path, log=print) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{video_id}.%(ext)s")

    def _progress_hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            speed = d.get("speed") or 0
            if total:
                pct = downloaded / total * 100
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                mbps = speed / (1024 * 1024)
                log(f"\r  Downloading audio ... {pct:.0f}% ({mb:.1f}/{total_mb:.1f} MB) {mbps:.1f} MB/s",
                    end="")
            else:
                mb = downloaded / (1024 * 1024)
                log(f"\r  Downloading audio ... {mb:.1f} MB", end="")
        elif d["status"] == "finished":
            log(f"\r  Download complete. Converting to WAV ...        ")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
        "extractor_args": {"youtube": {"player_client": ["android_vr"]}},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": {
            "FFmpegExtractAudio": ["-ar", "16000", "-ac", "1"],
        },
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    wav_path = output_dir / f"{video_id}.wav"
    return wav_path
