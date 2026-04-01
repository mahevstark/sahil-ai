import os
import time
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from pipeline.embedder import embed_texts
from pipeline.fetcher import fetch_video_meta, list_channel_videos
from pipeline.processor import process_video
from pipeline.storage import (
    get_connection,
    get_queue_stats,
    get_worker_nodes,
    queue_videos,
    search_chunks,
    video_exists,
)
from pipeline.transcriber import load_model


@click.group()
def cli():
    pass


# ---------------------------------------------------------------------------
# queue — admin queues videos for distributed workers
# ---------------------------------------------------------------------------

@cli.command("queue")
@click.option("--videos", "videos_opt", default=None,
              help="Comma-separated video IDs, or @file.txt (one ID per line)")
@click.option("--channel", default=None,
              help="YouTube channel URL — queues all unprocessed videos")
def queue_cmd(videos_opt, channel):
    """Queue videos for distributed processing by workers."""
    if not videos_opt and not channel:
        raise click.UsageError("Provide --videos or --channel")

    conn = get_connection()
    try:
        if videos_opt:
            if videos_opt.startswith("@"):
                ids = Path(videos_opt[1:]).read_text(encoding="utf-8").split()
            else:
                ids = [v.strip() for v in videos_opt.split(",") if v.strip()]
        else:
            click.echo(f"Fetching video list from {channel} ...")
            videos = list_channel_videos(channel)
            ids = [v["video_id"] for v in videos]
            click.echo(f"Found {len(ids)} videos.")

        added = queue_videos(conn, ids)
        click.echo(f"Queued {added} new video(s). ({len(ids) - added} already in queue.)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# status — show queue + active workers
# ---------------------------------------------------------------------------

@cli.command("status")
def status_cmd():
    """Show queue stats and registered worker PCs."""
    conn = get_connection()
    try:
        stats   = get_queue_stats(conn)
        workers = get_worker_nodes(conn)
    finally:
        conn.close()

    click.echo("\n--- Queue ---")
    if not stats:
        click.echo("  (empty)")
    for status, count in sorted(stats.items()):
        click.echo(f"  {status:12s}: {count}")

    click.echo("\n--- Workers ---")
    if not workers:
        click.echo("  No workers registered yet.")
    for w in workers:
        click.echo(f"  {w['hostname']:25s}  {w['status']:8s}  last seen: {w['last_heartbeat']}")
    click.echo()


# ---------------------------------------------------------------------------
# run — direct processing on this machine (no queue)
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option("--channel",  default=None, help="YouTube channel URL")
@click.option("--videos",   "videos_opt", default=None,
              help="Comma-separated video IDs, or @file.txt")
@click.option("--model",    "model_size", default=None,
              help="Whisper model size (default: $WHISPER_MODEL or large-v3)")
@click.option("--limit",    default=None, type=int, help="Max videos per pass")
@click.option("--retries",  default=3, show_default=True, type=int)
@click.option("--watch",    is_flag=True, default=False,
              help="Keep running; re-check for new videos every --interval seconds")
@click.option("--interval", default=3600, show_default=True, type=int,
              help="Seconds between heartbeat passes (requires --watch)")
def run_cmd(channel, videos_opt, model_size, limit, retries, watch, interval):
    """Process videos directly on this machine (bypasses the queue)."""
    if not channel and not videos_opt:
        raise click.UsageError("Provide --channel or --videos")

    if model_size is None:
        model_size = os.environ.get("WHISPER_MODEL", "large-v3")

    click.echo(f"Loading Whisper '{model_size}' ...")
    model = load_model(model_size)
    conn  = get_connection()
    pass_num = 0

    try:
        while True:
            pass_num += 1
            if watch:
                click.echo(f"\n--- Pass {pass_num} ---")

            if videos_opt:
                if videos_opt.startswith("@"):
                    ids = Path(videos_opt[1:]).read_text(encoding="utf-8").split()
                else:
                    ids = [v.strip() for v in videos_opt.split(",") if v.strip()]
                videos = [fetch_video_meta(vid) for vid in ids]
            else:
                click.echo(f"Listing videos from {channel} ...")
                videos = list_channel_videos(channel)
                click.echo(f"Found {len(videos)} videos.")

            if limit:
                videos = videos[:limit]

            processed = 0
            for meta in videos:
                video_id = meta["video_id"]
                title    = meta.get("title") or video_id

                if video_exists(conn, video_id):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT processed FROM videos WHERE video_id = %s", (video_id,)
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            click.echo(f"Skipping (done): {title}")
                            continue

                for attempt in range(1, retries + 1):
                    try:
                        click.echo(f"\nProcessing: {title}"
                                   + (f" (attempt {attempt}/{retries})" if attempt > 1 else ""))
                        process_video(conn, model, model_size, meta)
                        processed += 1
                        break
                    except Exception as exc:
                        click.echo(f"  Error: {exc}", err=True)
                        if attempt < retries:
                            wait = 2 ** attempt
                            click.echo(f"  Retrying in {wait}s ...")
                            time.sleep(wait)
                        else:
                            click.echo(f"  Failed after {retries} attempts, moving on.")

            click.echo(f"\nPass complete. Processed {processed} video(s).")

            if not watch:
                break
            click.echo(f"Next check in {interval}s  (Ctrl-C to stop) ...")
            time.sleep(interval)

    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# search — semantic search over stored chunks
# ---------------------------------------------------------------------------

@cli.command("search")
@click.argument("query")
@click.option("--top-k", "top_k", default=5, show_default=True, type=int)
def search_cmd(query, top_k):
    """Semantic search — returns video + timestamp for each result."""
    embeddings      = embed_texts([query])
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
        title   = row["title"] or row["video_id"]
        snippet = (row["text"] or "")[:200]
        url     = (f"https://www.youtube.com/watch?v={row['video_id']}"
                   f"&t={int(row['start_sec'])}")
        click.echo(
            f"\n[{rank}] similarity={row['similarity']:.4f} | {title} | "
            f"{row['start_sec']:.1f}s – {row['end_sec']:.1f}s"
        )
        click.echo(f"    {snippet}")
        click.echo(f"    {url}")


if __name__ == "__main__":
    cli()
