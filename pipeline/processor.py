"""
pipeline/processor.py — core video processing shared by main.py and worker.py.
"""
from pipeline.embedder import embed_texts
from pipeline.fetcher import download_audio as _fetch_audio
from pipeline.storage import (
    get_completed_chunk_indices,
    load_transcription_chunks,
    mark_processed,
    mark_transcription_job_done,
    save_transcription_chunk,
    upsert_chunks,
    upsert_transcription_job,
    upsert_video,
)
from pipeline.transcriber import WORK_DIR, chunk_segments, split_audio, transcribe


def process_video(conn, model, model_size: str, meta: dict, log=print):
    """Download → chunk → transcribe (with per-chunk DB resume) → embed → store."""
    video_id = meta["video_id"]
    title    = meta.get("title") or video_id

    upsert_video(conn, meta)

    video_dir = WORK_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    wav = video_dir / "audio.wav"

    if wav.exists():
        log("  Audio cached, skipping download.")
    else:
        log("  Downloading audio ...")
        tmp = _fetch_audio(video_id, video_dir)
        if tmp != wav:
            tmp.rename(wav)

    chunks = split_audio(wav, video_dir)
    total  = len(chunks)

    upsert_transcription_job(conn, video_id, model_size, title, total)
    done = get_completed_chunk_indices(conn, video_id, model_size)
    if done:
        log(f"  Resuming: {len(done)}/{total} chunks already done.")

    for i, (chunk_path, offset) in enumerate(chunks):
        if i in done:
            log(f"  Chunk {i+1}/{total} already done, skipping.")
            continue
        log(f"  Transcribing chunk {i+1}/{total} (offset {offset:.0f}s) ...")
        segs = transcribe(model, chunk_path)
        for s in segs:
            s["start"] += offset
            s["end"]   += offset
        save_transcription_chunk(conn, video_id, model_size, i, offset, segs)
        log(f"  Chunk {i+1}/{total} saved ({len(segs)} segments).")

    # Reconstruct full segment list from DB (fresh + resumed chunks)
    saved    = load_transcription_chunks(conn, video_id, model_size)
    all_segs = []
    for idx in sorted(saved.keys()):
        all_segs.extend(saved[idx])

    sem_chunks = chunk_segments(all_segs)
    texts      = [c["text"] for c in sem_chunks]
    log(f"  Embedding {len(texts)} semantic chunks ...")
    embeddings = embed_texts(texts)
    for c, emb in zip(sem_chunks, embeddings):
        c["embedding"] = emb

    upsert_chunks(conn, video_id, sem_chunks)
    mark_processed(conn, video_id)
    mark_transcription_job_done(conn, video_id, model_size)
    log(f"  Done — {len(sem_chunks)} chunks embedded.")
