from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from pipeline.embedder import embed_texts
from pipeline.fetcher import download_audio, list_channel_videos
from pipeline.storage import (
    get_connection,
    init_db,
    mark_processed,
    search_chunks,
    upsert_chunks,
    upsert_video,
    video_exists,
)
from pipeline.transcriber import chunk_segments, load_model, transcribe


@click.group()
def cli():
    pass


@cli.command("init-db")
def init_db_cmd():
    conn = get_connection()
    try:
        init_db(conn)
        click.echo("Database initialised successfully.")
    finally:
        conn.close()


@cli.command("run")
@click.option("--channel", required=True, help="YouTube channel URL")
@click.option(
    "--model",
    "model_size",
    default=None,
    show_default=True,
    help="Whisper model size (base/small/medium/large-v3)",
)
@click.option("--limit", default=None, type=int, help="Max number of videos to process")
@click.option("--audio-dir", "audio_dir", default="./audio_tmp", show_default=True)
def run_cmd(channel, model_size, limit, audio_dir):
    import os

    if model_size is None:
        model_size = os.environ.get("WHISPER_MODEL", "base")

    audio_path = Path(audio_dir)
    audio_path.mkdir(parents=True, exist_ok=True)

    # Load Whisper model FIRST — must happen before any subprocess/yt-dlp calls
    # on Windows to avoid ctranslate2 CUDA DLL conflict
    click.echo(f"Loading Whisper model '{model_size}' ...")
    whisper_model = load_model(model_size)

    conn = get_connection()
    try:
        click.echo(f"Listing videos from {channel} ...")
        videos = list_channel_videos(channel)
        click.echo(f"Found {len(videos)} videos.")

        if limit is not None:
            videos = videos[:limit]

        processed_count = 0
        for meta in videos:
            video_id = meta["video_id"]
            title = meta.get("title") or video_id

            if video_exists(conn, video_id):
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT processed FROM videos WHERE video_id = %s", (video_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        click.echo(f"Skipping already-processed: {title}")
                        continue

            try:
                click.echo(f"Processing: {title}")
                upsert_video(conn, meta)

                wav_path = download_audio(video_id, audio_path)

                segments = transcribe(whisper_model, wav_path)
                chunks = chunk_segments(segments)

                texts = [c["text"] for c in chunks]
                embeddings = embed_texts(texts)

                for chunk, emb in zip(chunks, embeddings):
                    chunk["embedding"] = emb

                upsert_chunks(conn, video_id, chunks)
                mark_processed(conn, video_id)

                wav_path.unlink(missing_ok=True)

                click.echo(f"  Done — {len(chunks)} chunks embedded.")
                processed_count += 1

            except Exception as exc:
                click.echo(f"  Error processing '{title}': {exc}", err=True)
                continue

        click.echo(f"Finished. Processed {processed_count} video(s).")
    finally:
        conn.close()


@cli.command("search")
@click.argument("query")
@click.option("--top-k", "top_k", default=5, show_default=True, type=int)
def search_cmd(query, top_k):
    click.echo(f"Embedding query ...")
    embeddings = embed_texts([query])
    query_embedding = embeddings[0]

    conn = get_connection()
    try:
        results = search_chunks(conn, query_embedding, top_k=top_k)
    finally:
        conn.close()

    if not results:
        click.echo("No results found.")
        return

    for rank, row in enumerate(results, start=1):
        start = row["start_sec"]
        end = row["end_sec"]
        similarity = row["similarity"]
        title = row["title"] or row["video_id"]
        snippet = (row["text"] or "")[:200]

        click.echo(
            f"\n[{rank}] similarity={similarity:.4f} | {title} | "
            f"{start:.1f}s – {end:.1f}s"
        )
        click.echo(f"    {snippet}")


if __name__ == "__main__":
    cli()
