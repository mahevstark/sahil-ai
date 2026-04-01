import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_CACHE = Path.home() / ".cache" / "faster-whisper"
WORK_DIR    = Path.home() / ".cache" / "faster-whisper-transcripts"
CHUNK_SECS  = 180  # 3 minutes


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


_FFMPEG = _find_ffmpeg()


def load_model(model_size: str = "large-v3") -> WhisperModel:
    import ctranslate2
    try:
        device = "cuda" if ctranslate2.get_supported_compute_types("cuda") else "cpu"
    except Exception:
        device = "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    cached = MODEL_CACHE / model_size

    if cached.exists():
        print(f"  Loading Whisper {model_size} from local cache ...", flush=True)
    else:
        print(f"  Downloading Whisper {model_size} (~3 GB, one-time) ...", flush=True)

    done = threading.Event()

    def _progress():
        symbols = ["|", "/", "-", "\\"]
        i = 0
        while not done.is_set():
            print(f"\r  Please wait ... {symbols[i % 4]}", end="", flush=True)
            i += 1
            time.sleep(0.5)
        print("\r  Model ready.                    ", flush=True)

    t = threading.Thread(target=_progress, daemon=True)
    t.start()
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type,
                             download_root=str(MODEL_CACHE))
    finally:
        done.set()
        t.join()
    return model


def get_audio_duration(wav_path: Path) -> float:
    r = subprocess.run([_FFMPEG, "-i", str(wav_path)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0.0


def split_audio(wav_path: Path, video_dir: Path) -> list:
    """Split wav into CHUNK_SECS pieces. Returns [(chunk_path, offset_sec), ...]."""
    duration = get_audio_duration(wav_path)
    if duration <= CHUNK_SECS:
        return [(wav_path, 0.0)]

    total = int(duration // CHUNK_SECS) + (1 if duration % CHUNK_SECS else 0)
    print(f"  Audio {duration/60:.1f} min — splitting into {total} chunks ...", flush=True)

    chunks = []
    for i in range(total):
        start = i * CHUNK_SECS
        chunk_path = video_dir / f"chunk_{i:03d}.wav"
        if not chunk_path.exists():
            r = subprocess.run(
                [_FFMPEG, "-y", "-i", str(wav_path),
                 "-ss", str(start), "-t", str(CHUNK_SECS),
                 "-ar", "16000", "-ac", "1", str(chunk_path)],
                capture_output=True,
            )
            if r.returncode != 0 or not chunk_path.exists():
                raise RuntimeError(f"ffmpeg chunk {i} failed")
            print(f"  Split {i+1}/{total}", flush=True)
        chunks.append((chunk_path, float(start)))

    return chunks


def transcribe(model: WhisperModel, wav_path: Path) -> list:
    """Transcribe a single wav file. Returns segments without timestamp offset."""
    segments, _info = model.transcribe(str(wav_path), beam_size=5)
    return [{"text": s.text, "start": s.start, "end": s.end} for s in segments]


def chunk_segments(segments: list, max_chars: int = 1500) -> list:
    """Group whisper segments into semantic chunks for embedding."""
    chunks = []
    current_texts, current_start, current_end, accumulated, chunk_index = [], None, None, 0, 0

    for seg in segments:
        text, start, end = seg["text"], seg["start"], seg["end"]
        if current_start is None:
            current_start = start
        if accumulated + len(text) > max_chars and current_texts:
            chunks.append({"chunk_index": chunk_index,
                           "text": "".join(current_texts).strip(),
                           "start_sec": current_start, "end_sec": current_end})
            chunk_index += 1
            current_texts, current_start, accumulated = [text], start, len(text)
        else:
            current_texts.append(text)
            accumulated += len(text)
        current_end = end

    if current_texts:
        chunks.append({"chunk_index": chunk_index,
                       "text": "".join(current_texts).strip(),
                       "start_sec": current_start, "end_sec": current_end})
    return chunks
