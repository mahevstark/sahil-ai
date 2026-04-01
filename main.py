"""
main.py — Sahil AI interactive console.

Run:  python main.py
"""
import os
import sys
import time
import textwrap
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Interactive mode (default when no subcommand given)
# ---------------------------------------------------------------------------

def _print_header():
    print("\n" + "=" * 52)
    print("   Sahil AI  —  Video Transcription & Search")
    print("=" * 52 + "\n")


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(
        prefix + line
        for paragraph in text.split("\n")
        for line in (textwrap.wrap(paragraph, width=76) or [""])
    )


# ── Queue ──────────────────────────────────────────────────────────────────

def _do_queue(conn):
    import questionary
    from pipeline.fetcher import list_channel_videos
    from pipeline.storage import queue_videos

    source = questionary.select(
        "Queue videos from:",
        choices=[
            questionary.Choice("Enter video IDs", value="ids"),
            questionary.Choice("Text file (one ID per line)", value="file"),
            questionary.Choice("YouTube channel URL", value="channel"),
        ],
    ).ask()
    if not source:
        return

    if source == "ids":
        raw = questionary.text("Video IDs (comma-separated):").ask()
        if not raw:
            return
        ids = [v.strip() for v in raw.replace("\n", ",").split(",") if v.strip()]

    elif source == "file":
        path = questionary.text("Path to file:").ask()
        if not path:
            return
        ids = [i for i in Path(path.strip()).read_text(encoding="utf-8").split() if i]
        print(f"  Read {len(ids)} video IDs from file.", flush=True)

    else:
        channel = questionary.text("Channel URL:").ask()
        if not channel:
            return
        print("  Fetching video list ...", flush=True)
        videos = list_channel_videos(channel.strip())
        ids = [v["video_id"] for v in videos]
        print(f"  Found {len(ids)} videos.", flush=True)

    if not ids:
        print("  No video IDs found.")
        return

    print(f"  Queueing {len(ids)} video(s) ...", flush=True)
    added = queue_videos(conn, ids)
    skip  = len(ids) - added
    print(f"  Done. Queued {added} new video(s)."
          + (f"  ({skip} already in queue)" if skip else ""))


# ── Clear queue ────────────────────────────────────────────────────────────

def _do_clear_queue(conn):
    import questionary

    scope = questionary.select(
        "Which jobs to remove?",
        choices=[
            questionary.Choice("Queued only  (waiting, not yet started)", value="queued"),
            questionary.Choice("Failed only  (exhausted retries)",         value="failed"),
            questionary.Choice("All          (queued + failed + processing)", value="all"),
        ],
    ).ask()
    if not scope:
        return

    confirm = questionary.confirm(
        f"Delete all '{scope}' jobs from the queue? This cannot be undone.",
        default=False,
    ).ask()
    if not confirm:
        print("  Cancelled.")
        return

    with conn.cursor() as cur:
        if scope == "all":
            cur.execute("DELETE FROM job_queue")
        else:
            cur.execute("DELETE FROM job_queue WHERE status = %s", (scope,))
        deleted = cur.rowcount
    conn.commit()
    print(f"  Removed {deleted} job(s) from the queue.")


# ── Workers & benchmarks ───────────────────────────────────────────────────

def _do_workers(conn):
    from pipeline.storage import get_benchmarks, get_worker_nodes

    workers = get_worker_nodes(conn)

    print("\n  -- Registered Workers --")
    if not workers:
        print("    No workers registered yet.")
    else:
        print(f"    {'ID':<25} {'Status':<8} {'OS':<18} {'CPU':<30} {'RAM':>6}  Last seen")
        print("    " + "-" * 100)
        for w in workers:
            last = str(w["last_heartbeat"])[:19] if w["last_heartbeat"] else "-"
            os_  = (w["os"] or "-")[:17]
            cpu  = (w["cpu"] or "-")[:29]
            ram  = f"{w['ram_gb']} GB" if w["ram_gb"] else "-"
            print(f"    {w['worker_id']:<25} {w['status']:<8} {os_:<18} {cpu:<30} {ram:>6}  {last}")

    print("\n  -- Transcription Speed (WPM = words per minute processed) --")
    rows = get_benchmarks(conn)
    if not rows:
        print("    No benchmark data yet — workers will record it after completing jobs.")
    else:
        print(f"    {'Worker':<25} {'Model':<10} {'Jobs':>4}  {'Avg WPM':>8}  {'Best WPM':>9}  {'Avg audio':>10}")
        print("    " + "-" * 78)
        for r in rows:
            avg_audio = f"{r['avg_audio_mins']} min" if r["avg_audio_mins"] else "-"
            print(f"    {r['worker_id']:<25} {r['model_size'] or '-':<10} {r['jobs']:>4}"
                  f"  {r['avg_wpm']:>8}  {r['best_wpm']:>9}  {avg_audio:>10}")
    print()


# ── Status ─────────────────────────────────────────────────────────────────

def _do_status(conn):
    from pipeline.storage import get_queue_stats, get_worker_nodes

    stats   = get_queue_stats(conn)
    workers = get_worker_nodes(conn)

    print("\n  -- Job Queue --")
    if not stats:
        print("    (empty)")
    for status, count in sorted(stats.items()):
        bar = "#" * min(count, 40)
        print(f"    {status:12s}: {count:4d}  {bar}")
    print(f"    {'total':12s}: {sum(stats.values()):4d}")

    print("\n  -- Workers --")
    if not workers:
        print("    No workers registered yet.")
    for w in workers:
        print(f"    {w['hostname']:25s}  {w['status']:8s}  "
              f"last seen: {str(w['last_heartbeat'])[:19]}")

    print("\n  -- Recently Processed --")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.title, j.video_id, j.model_size, j.updated_at
            FROM transcription_jobs j
            JOIN videos v ON v.video_id = j.video_id
            WHERE j.status = 'completed'
            ORDER BY j.updated_at DESC LIMIT 8
        """)
        recent = cur.fetchall()
    if not recent:
        print("    None yet.")
    for title, vid, model, ts in recent:
        print(f"    {str(ts)[:19]}  [{model}]  {title or vid}")


# ── Search ─────────────────────────────────────────────────────────────────

def _do_search(conn):
    import questionary
    from pipeline.embedder import embed_texts
    from pipeline.storage import search_chunks

    query = questionary.text("Search query:").ask()
    if not query:
        return
    raw_k = questionary.text("Number of results:", default="5").ask()
    top_k = int(raw_k) if raw_k and raw_k.isdigit() else 5

    print(f"\n  Searching ...", flush=True)
    results = search_chunks(conn, embed_texts([query])[0], top_k=top_k)

    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results, 1):
        url = (f"https://www.youtube.com/watch?v={r['video_id']}"
               f"&t={int(r['start_sec'])}")
        print(f"\n  [{i}] {r['title'] or r['video_id']}")
        print(f"       similarity : {r['similarity']:.4f}")
        print(f"       timestamp  : {r['start_sec']:.1f}s - {r['end_sec']:.1f}s")
        print(f"       url        : {url}")
        snippet = (r["text"] or "")[:200].replace("\n", " ")
        print(f"       \"{snippet}\"")


# ── Summarize ──────────────────────────────────────────────────────────────

def _do_summarize(conn):
    import questionary
    from pipeline.qa import summarize_video

    video_id = questionary.text("Video ID:").ask()
    if not video_id:
        return
    video_id = video_id.strip()

    print(f"\n  Summarizing {video_id} ...\n", flush=True)
    summary, meta = summarize_video(conn, video_id)

    if meta:
        print(f"  Title  : {meta.get('title', video_id)}")
        print(f"  Channel: {meta.get('channel') or '-'}")
        print(f"  URL    : {meta.get('url') or '-'}")
        print(f"  Chunks : {meta.get('chunk_count', '?')}")

    print(f"\n  Summary:\n")
    print(_wrap(summary, indent=4))


# ── Ask ────────────────────────────────────────────────────────────────────

def _do_ask(conn):
    import questionary
    from pipeline.qa import answer_question

    question = questionary.text("Your question:").ask()
    if not question:
        return
    raw_k = questionary.text("Max sources to use:", default="5").ask()
    top_k = int(raw_k) if raw_k and raw_k.isdigit() else 5

    print(f"\n  Thinking ...\n", flush=True)
    result = answer_question(conn, question, top_k=top_k)

    print("  Answer:\n")
    print(_wrap(result["answer"], indent=4))

    if result["sources"]:
        print(f"\n  Sources used ({len(result['sources'])}):\n")
        for s in result["sources"]:
            snippet = (s["text"] or "")[:150].replace("\n", " ")
            print(f"  [Source {s['index']}] {s['title'] or s['video_id']}")
            print(f"             similarity : {s['similarity']:.4f}")
            print(f"             timestamp  : {s['start_sec']:.1f}s - {s['end_sec']:.1f}s")
            print(f"             url        : {s['url']}")
            print(f"             \"{snippet}\"")


# ── Process now (claims from queue) ───────────────────────────────────────

def _do_run(conn):
    import questionary
    from pipeline.fetcher import fetch_video_meta
    from pipeline.processor import process_video
    from pipeline.storage import get_queue_stats
    from pipeline.transcriber import load_model

    stats  = get_queue_stats(conn)
    queued = stats.get("queued", 0)
    if queued == 0:
        print("  Queue is empty. Use 'Queue videos' first.")
        return
    print(f"  {queued} video(s) waiting in queue.", flush=True)

    model_size = questionary.select(
        "Whisper model:",
        choices=[
            questionary.Choice("large-v3  (best, recommended for Urdu/mixed)", value="large-v3"),
            questionary.Choice("medium", value="medium"),
            questionary.Choice("small    (fast)", value="small"),
            questionary.Choice("base     (fastest)", value="base"),
        ],
        default=os.environ.get("WHISPER_MODEL", "large-v3"),
    ).ask()
    if not model_size:
        return

    print(f"\n  Loading Whisper '{model_size}' ...", flush=True)
    model = load_model(model_size)
    print(f"  Processing queue — press Ctrl-C to stop.\n", flush=True)

    done = 0
    while True:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE job_queue
                SET status = 'processing', worker_id = NULL, started_at = NOW()
                WHERE id = (
                    SELECT id FROM job_queue WHERE status = 'queued'
                    ORDER BY queued_at LIMIT 1 FOR UPDATE SKIP LOCKED
                )
                RETURNING id, video_id
            """)
            row = cur.fetchone()
        conn.commit()

        if row is None:
            print(f"\n  Queue empty. Processed {done} video(s) this session.")
            break

        job_id, video_id = row
        meta = fetch_video_meta(video_id)
        print(f"  [{done+1}/{done + queued}] {meta.get('title') or video_id}", flush=True)

        # Use a fresh connection per video so the long download/split phase
        # (no DB activity) doesn't kill the shared menu-session connection.
        # Both connections have TCP keepalives (set in get_connection()).
        proc_conn = None
        try:
            from pipeline.storage import get_connection as _mkconn
            proc_conn = _mkconn()
            process_video(proc_conn, model, model_size, meta)
            with proc_conn.cursor() as cur:
                cur.execute(
                    "UPDATE job_queue SET status='completed', completed_at=NOW() WHERE id=%s",
                    (job_id,)
                )
            proc_conn.commit()
            done += 1
        except Exception as exc:
            print(f"       Error: {exc}", flush=True)
            if proc_conn:
                try:
                    proc_conn.rollback()
                except Exception:
                    pass
            # Update error status — use a fresh conn in case proc_conn is dead
            try:
                from pipeline.storage import get_connection as _mkconn
                err_conn = _mkconn()
                with err_conn.cursor() as cur:
                    cur.execute("""
                        UPDATE job_queue SET
                            retries = retries + 1, error_msg = %s,
                            worker_id = NULL, started_at = NULL,
                            status = CASE WHEN retries + 1 >= 3 THEN 'failed' ELSE 'queued' END
                        WHERE id = %s
                    """, (str(exc)[:500], job_id))
                err_conn.commit()
                err_conn.close()
            except Exception:
                pass
        finally:
            if proc_conn:
                try:
                    proc_conn.close()
                except Exception:
                    pass


# ── Main interactive loop ──────────────────────────────────────────────────

def _interactive_mode():
    import questionary
    from pipeline.storage import get_connection

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _print_header()

    print("  Connecting to database ...", flush=True)
    conn = get_connection()
    print("  Connected.\n", flush=True)

    MENU = [
        questionary.Choice("Queue videos for processing",            value="queue"),
        questionary.Choice("Process videos now (on this machine)",   value="run"),
        questionary.Choice("Check progress & worker status",         value="status"),
        questionary.Choice("Workers & performance benchmarks",        value="workers"),
        questionary.Choice("Search videos by text",                  value="search"),
        questionary.Choice("Summarize a video",                      value="summarize"),
        questionary.Choice("Ask a question (answers from videos)",   value="ask"),
        questionary.Choice("Clear queue",                            value="clear_queue"),
        questionary.Choice("Exit",                                   value="exit"),
    ]

    try:
        while True:
            action = questionary.select("What do you want to do?", choices=MENU).ask()

            if not action or action == "exit":
                break

            print()
            try:
                if action == "queue":             _do_queue(conn)
                elif action == "run":             _do_run(conn)
                elif action == "status":          _do_status(conn)
                elif action == "workers":         _do_workers(conn)
                elif action == "search":          _do_search(conn)
                elif action == "summarize":       _do_summarize(conn)
                elif action == "ask":             _do_ask(conn)
                elif action == "clear_queue":     _do_clear_queue(conn)
            except KeyboardInterrupt:
                conn.rollback()
                print("  (cancelled)")
            except Exception as exc:
                conn.rollback()
                print(f"\n  Error: {exc}")
            print()

    except KeyboardInterrupt:
        pass
    finally:
        conn.close()

    print("\nBye.\n")


# ---------------------------------------------------------------------------
# Click CLI (for scripting / automation — same features without the menu)
# ---------------------------------------------------------------------------

import click

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Sahil AI — run with no subcommand for the interactive menu."""
    if ctx.invoked_subcommand is None:
        _interactive_mode()


@cli.command("queue")
@click.option("--videos",  "videos_opt", default=None,
              help="Comma-separated video IDs or @file.txt")
@click.option("--channel", default=None, help="YouTube channel URL")
def queue_cmd(videos_opt, channel):
    """Queue videos for distributed processing."""
    if not videos_opt and not channel:
        raise click.UsageError("Provide --videos or --channel")
    from pipeline.fetcher import list_channel_videos
    from pipeline.storage import get_connection, queue_videos
    conn = get_connection()
    try:
        if videos_opt:
            ids = (Path(videos_opt[1:]).read_text(encoding="utf-8").split()
                   if videos_opt.startswith("@")
                   else [v.strip() for v in videos_opt.split(",") if v.strip()])
        else:
            click.echo(f"Fetching {channel} ...")
            ids = [v["video_id"] for v in list_channel_videos(channel)]
            click.echo(f"Found {len(ids)} videos.")
        added = queue_videos(conn, ids)
        click.echo(f"Queued {added} / {len(ids)} video(s).")
    finally:
        conn.close()


@cli.command("status")
def status_cmd():
    """Show queue stats and registered workers."""
    from pipeline.storage import get_connection
    conn = get_connection()
    try:
        _do_status(conn)
    finally:
        conn.close()


@cli.command("search")
@click.argument("query")
@click.option("--top-k", default=5, show_default=True, type=int)
def search_cmd(query, top_k):
    """Semantic search over stored video chunks."""
    from pipeline.embedder import embed_texts
    from pipeline.storage import get_connection, search_chunks
    conn = get_connection()
    try:
        results = search_chunks(conn, embed_texts([query])[0], top_k=top_k)
    finally:
        conn.close()
    for i, r in enumerate(results, 1):
        url = (f"https://www.youtube.com/watch?v={r['video_id']}"
               f"&t={int(r['start_sec'])}")
        click.echo(f"\n[{i}] {r['title'] or r['video_id']} | "
                   f"sim={r['similarity']:.4f} | {r['start_sec']:.1f}s")
        click.echo(f"    {(r['text'] or '')[:200]}")
        click.echo(f"    {url}")


@cli.command("ask")
@click.argument("question")
@click.option("--top-k", default=5, show_default=True, type=int)
def ask_cmd(question, top_k):
    """Answer a question using video embeddings (RAG)."""
    from pipeline.qa import answer_question
    from pipeline.storage import get_connection
    conn = get_connection()
    try:
        result = answer_question(conn, question, top_k=top_k)
    finally:
        conn.close()
    click.echo(f"\nAnswer:\n{result['answer']}\n")
    click.echo("Sources:")
    for s in result["sources"]:
        click.echo(f"  [{s['index']}] {s['title'] or s['video_id']} | "
                   f"{s['start_sec']:.1f}s | {s['url']}")


@cli.command("summarize")
@click.argument("video_id")
def summarize_cmd(video_id):
    """Summarize a processed video."""
    from pipeline.qa import summarize_video
    from pipeline.storage import get_connection
    conn = get_connection()
    try:
        summary, meta = summarize_video(conn, video_id)
    finally:
        conn.close()
    if meta:
        click.echo(f"Title: {meta.get('title')}")
    click.echo(f"\n{summary}")


@cli.command("run")
@click.option("--channel",  default=None)
@click.option("--videos",   "videos_opt", default=None)
@click.option("--model",    "model_size", default=None)
@click.option("--limit",    default=None, type=int)
@click.option("--retries",  default=3,    type=int)
@click.option("--watch",    is_flag=True, default=False)
@click.option("--interval", default=3600, type=int)
def run_cmd(channel, videos_opt, model_size, limit, retries, watch, interval):
    """Process videos directly on this machine (bypasses queue)."""
    if not channel and not videos_opt:
        raise click.UsageError("Provide --channel or --videos")
    from pipeline.fetcher import fetch_video_meta, list_channel_videos
    from pipeline.processor import process_video
    from pipeline.storage import get_connection, video_exists
    from pipeline.transcriber import load_model

    if not model_size:
        model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    click.echo(f"Loading Whisper '{model_size}' ...")
    model = load_model(model_size)
    conn  = get_connection()

    try:
        while True:
            if videos_opt:
                ids = (Path(videos_opt[1:]).read_text(encoding="utf-8").split()
                       if videos_opt.startswith("@")
                       else [v.strip() for v in videos_opt.split(",") if v.strip()])
                videos = [fetch_video_meta(vid) for vid in ids]
            else:
                click.echo(f"Listing {channel} ...")
                videos = list_channel_videos(channel)
                click.echo(f"Found {len(videos)}.")
            if limit:
                videos = videos[:limit]
            done = 0
            for meta in videos:
                vid = meta["video_id"]
                if video_exists(conn, vid):
                    with conn.cursor() as cur:
                        cur.execute("SELECT processed FROM videos WHERE video_id=%s", (vid,))
                        r = cur.fetchone()
                        if r and r[0]:
                            click.echo(f"Skip: {meta.get('title') or vid}")
                            continue
                for attempt in range(1, retries + 1):
                    try:
                        click.echo(f"\nProcessing: {meta.get('title') or vid}"
                                   + (f" (attempt {attempt})" if attempt > 1 else ""))
                        process_video(conn, model, model_size, meta)
                        done += 1
                        break
                    except Exception as exc:
                        click.echo(f"  Error: {exc}", err=True)
                        if attempt < retries:
                            time.sleep(2 ** attempt)
                        else:
                            click.echo("  Giving up.")
            click.echo(f"\nPass done. Processed {done}.")
            if not watch:
                break
            click.echo(f"Watching — next in {interval}s ...")
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    cli()
