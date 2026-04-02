"""
transcribe.py — interactive single-video transcriber with DB-backed resume.

Just run:
    python transcribe.py
"""

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Locate ffmpeg / yt-dlp
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
# Paths
# ---------------------------------------------------------------------------

MODEL_CACHE = Path.home() / ".cache" / "faster-whisper"
WORK_DIR    = Path.home() / ".cache" / "faster-whisper-transcripts"
CHUNK_SECS  = 180  # 3 minutes per chunk


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_conn():
    import psycopg2
    from pgvector.psycopg2 import register_vector
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set in .env")
    # First connection: create extension + tables (vector type may not exist yet)
    bootstrap = psycopg2.connect(url)
    schema_sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with bootstrap.cursor() as cur:
        cur.execute(schema_sql)
    bootstrap.commit()
    bootstrap.close()
    # Second connection: vector type now exists, safe to register
    # TCP keepalives prevent Railway from killing idle connections during
    # long audio-download/split phases (can be 10-20 min with no DB activity).
    conn = psycopg2.connect(
        url,
        keepalives=1,
        keepalives_idle=60,
        keepalives_interval=10,
        keepalives_count=5,
    )
    register_vector(conn)
    return conn


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcription_jobs (
                video_id     TEXT  NOT NULL,
                model_size   TEXT  NOT NULL,
                title        TEXT,
                status       TEXT  DEFAULT 'in_progress',
                total_chunks INT,
                created_at   TIMESTAMP DEFAULT NOW(),
                updated_at   TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (video_id, model_size)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcription_chunks (
                id           SERIAL PRIMARY KEY,
                video_id     TEXT  NOT NULL,
                model_size   TEXT  NOT NULL,
                chunk_index  INT   NOT NULL,
                offset_secs  FLOAT NOT NULL,
                status       TEXT  DEFAULT 'pending',
                text         TEXT,
                created_at   TIMESTAMP DEFAULT NOW(),
                updated_at   TIMESTAMP DEFAULT NOW(),
                UNIQUE (video_id, model_size, chunk_index)
            )
        """)
    conn.commit()


def upsert_job(conn, video_id: str, model_size: str, title: str, total_chunks: int):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO transcription_jobs (video_id, model_size, title, total_chunks, status)
            VALUES (%s, %s, %s, %s, 'in_progress')
            ON CONFLICT (video_id, model_size) DO UPDATE SET
                title        = EXCLUDED.title,
                total_chunks = EXCLUDED.total_chunks,
                status       = 'in_progress',
                updated_at   = NOW()
        """, (video_id, model_size, title, total_chunks))
    conn.commit()


def get_completed_indices(conn, video_id: str, model_size: str) -> set:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_index FROM transcription_chunks
            WHERE video_id = %s AND model_size = %s AND status = 'completed'
        """, (video_id, model_size))
        return {row[0] for row in cur.fetchall()}


def save_chunk(conn, video_id: str, model_size: str, chunk_index: int,
               offset_secs: float, segments: list):
    """Persist a completed chunk's segments as JSON."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO transcription_chunks
                (video_id, model_size, chunk_index, offset_secs, status, text)
            VALUES (%s, %s, %s, %s, 'completed', %s)
            ON CONFLICT (video_id, model_size, chunk_index) DO UPDATE SET
                status     = 'completed',
                text       = EXCLUDED.text,
                updated_at = NOW()
        """, (video_id, model_size, chunk_index, offset_secs, json.dumps(segments)))
    conn.commit()


def load_all_chunks(conn, video_id: str, model_size: str) -> dict:
    """Return {chunk_index: [segments]} for all completed chunks."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_index, text FROM transcription_chunks
            WHERE video_id = %s AND model_size = %s AND status = 'completed'
            ORDER BY chunk_index
        """, (video_id, model_size))
        return {row[0]: json.loads(row[1]) for row in cur.fetchall()}


def mark_job_done(conn, video_id: str, model_size: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE transcription_jobs
            SET status = 'completed', updated_at = NOW()
            WHERE video_id = %s AND model_size = %s
        """, (video_id, model_size))
    conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(text: str) -> str:
    safe = re.sub(r'[\\/*?:"<>|]', "", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:80] or "transcript"


def fetch_video_meta(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    r = subprocess.run(
        [_YTDLP, "--dump-json", "--no-playlist", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    fallback = {"video_id": video_id, "title": video_id, "channel": None,
                "url": url, "duration": None, "uploaded_at": None}
    if r.returncode != 0:
        return fallback
    try:
        data = json.loads(r.stdout)
        raw_date = data.get("upload_date")
        uploaded_at = datetime.datetime.strptime(raw_date, "%Y%m%d").date() if raw_date else None
        return {
            "video_id":    video_id,
            "title":       data.get("title", video_id),
            "channel":     data.get("channel") or data.get("uploader"),
            "url":         url,
            "duration":    data.get("duration"),
            "uploaded_at": uploaded_at,
        }
    except Exception:
        return fallback


def fetch_title(video_id: str) -> str:
    return fetch_video_meta(video_id)["title"]


def load_whisper(model_size: str):
    from faster_whisper import WhisperModel
    import ctranslate2

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

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
        print(f"  Downloading Whisper {model_size} → {cached} (one-time, ~3GB for large-v3) ...", flush=True)

    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(MODEL_CACHE),
    )


def get_video_dir(video_id: str) -> Path:
    d = WORK_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_audio(video_id: str) -> Path:
    video_dir = get_video_dir(video_id)
    wav = video_dir / "audio.wav"

    if wav.exists():
        print("  Found cached audio, skipping download.", flush=True)
        return wav

    url = f"https://www.youtube.com/watch?v={video_id}"
    print("  Downloading audio ...", flush=True)
    r = subprocess.run(
        [_YTDLP, "-f", "bestaudio/best", "--no-playlist",
         "-o", str(video_dir / "audio_native.%(ext)s"), url],
        capture_output=True,
    )
    if r.returncode not in (0, 1):
        raise RuntimeError(f"yt-dlp failed (exit {r.returncode})")

    downloaded = next(video_dir.glob("audio_native.*"), None)
    if not downloaded:
        raise FileNotFoundError("yt-dlp produced no audio file")

    print("  Converting to 16kHz mono WAV ...", flush=True)
    r = subprocess.run(
        [_FFMPEG, "-y", "-i", str(downloaded), "-ar", "16000", "-ac", "1", str(wav)],
        capture_output=True,
    )
    downloaded.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode('utf-8', errors='replace')[-300:]}")
    return wav


def get_audio_duration(wav_path: Path) -> float:
    r = subprocess.run([_FFMPEG, "-i", str(wav_path)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0.0


def split_audio(wav_path: Path, video_dir: Path) -> list:
    """Split wav into CHUNK_SECS-second pieces. Returns [(chunk_path, offset_sec), ...]."""
    duration = get_audio_duration(wav_path)
    if duration <= CHUNK_SECS:
        return [(wav_path, 0.0)]

    total = int(duration // CHUNK_SECS) + (1 if duration % CHUNK_SECS else 0)
    print(f"  Audio is {duration/60:.1f} min — splitting into {total} chunks of {CHUNK_SECS//60} min ...", flush=True)

    chunks = []
    for i in range(total):
        start = i * CHUNK_SECS
        chunk_path = video_dir / f"chunk_{i:03d}.wav"
        if chunk_path.exists():
            print(f"  Chunk {i+1}/{total} already on disk, skipping split.", flush=True)
        else:
            r = subprocess.run(
                [_FFMPEG, "-y", "-i", str(wav_path),
                 "-ss", str(start), "-t", str(CHUNK_SECS),
                 "-ar", "16000", "-ac", "1", str(chunk_path)],
                capture_output=True,
            )
            if r.returncode != 0 or not chunk_path.exists():
                raise RuntimeError(f"ffmpeg chunking failed at chunk {i}")
            print(f"  Split {i+1}/{total}", flush=True)
        chunks.append((chunk_path, float(start)))

    return chunks


def transcribe(model, wav_path: Path, video_dir: Path,
               video_id: str, model_size: str, conn) -> list:
    chunks = split_audio(wav_path, video_dir)
    total  = len(chunks)

    done = get_completed_indices(conn, video_id, model_size)
    upsert_job(conn, video_id, model_size, None, total)

    if done:
        print(f"  Resuming: {len(done)}/{total} chunks already transcribed.", flush=True)

    detected_lang = None

    for i, (chunk_path, offset) in enumerate(chunks):
        if i in done:
            print(f"  Chunk {i+1}/{total} — already done, skipping.", flush=True)
            continue

        label = f"chunk {i+1}/{total}" if total > 1 else ""
        print(f"  Transcribing {label} (offset {format_time(offset)}) ...", flush=True)

        segments_raw, info = model.transcribe(str(chunk_path), beam_size=5)

        if detected_lang is None:
            detected_lang = info.language
            print(f"  Language: {info.language}  (confidence {info.language_probability:.0%})", flush=True)

        segs = [
            {"start": s.start + offset, "end": s.end + offset, "text": s.text.strip()}
            for s in segments_raw
        ]
        save_chunk(conn, video_id, model_size, i, offset, segs)
        print(f"  ✓ Chunk {i+1}/{total} saved to DB  ({len(segs)} segments)", flush=True)

    # Reconstruct full segment list from DB — handles fresh + resumed chunks uniformly
    all_chunks = load_all_chunks(conn, video_id, model_size)
    all_segments = []
    for idx in sorted(all_chunks.keys()):
        all_segments.extend(all_chunks[idx])

    return all_segments


def embed_and_store(conn, video_id: str, meta: dict, segments: list):
    from pipeline.storage import upsert_video, upsert_chunks, mark_processed
    from pipeline.embedder import embed_texts

    upsert_video(conn, meta)

    texts = [s["text"] for s in segments]
    print(f"  Embedding {len(texts)} segments via OpenAI ...", flush=True)
    embeddings = embed_texts(texts, batch_size=100)

    chunks = [
        {
            "chunk_index": i,
            "text":        seg["text"],
            "start_sec":   seg["start"],
            "end_sec":     seg["end"],
            "embedding":   emb,
        }
        for i, (seg, emb) in enumerate(zip(segments, embeddings))
    ]

    print(f"  Storing {len(chunks)} chunks with embeddings ...", flush=True)
    upsert_chunks(conn, video_id, chunks)
    mark_processed(conn, video_id)
    print(f"  ✓ Embeddings stored. Each chunk tagged with video_id={video_id} + timestamp.", flush=True)


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

    # 3. Connect to DB and ensure tables exist
    print("\n  Connecting to database ...", flush=True)
    conn = get_db_conn()
    ensure_tables(conn)

    # 4. Load model
    print()
    model = load_whisper(model_size)

    # 5. Fetch video metadata (title, channel, duration, upload date)
    print("  Fetching video metadata ...", flush=True)
    meta  = fetch_video_meta(video_id)
    title = meta["title"]
    default_output = f"{safe_filename(title)}.txt"

    # 6. Output filename
    print()
    output_file = questionary.text("Output file:", default=default_output).ask()
    if not output_file:
        output_file = default_output
    output_file = output_file.strip()

    print(f"\nTitle : {title}")
    print(f"Output: {output_file}\n")

    # Register job in DB (total_chunks updated after split inside transcribe)
    upsert_job(conn, video_id, model_size, title, 0)

    # 7. Run pipeline
    wav_path  = download_audio(video_id)
    video_dir = get_video_dir(video_id)
    segments  = transcribe(model, wav_path, video_dir, video_id, model_size, conn)

    mark_job_done(conn, video_id, model_size)

    # Embed every segment and store in chunks table with video_id + timestamps
    print()
    embed_and_store(conn, video_id, meta, segments)
    conn.close()

    # 8. Write output file
    lines = [
        f"[{format_time(s['start'])} --> {format_time(s['end'])}]  {s['text']}"
        for s in segments
    ]
    Path(output_file).write_text(
        f"Title: {title}\nURL: https://www.youtube.com/watch?v={video_id}\n\n"
        + "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"\n✓ {len(segments)} segments saved to: {Path(output_file).resolve()}\n")


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
