"""
worker.py — distributed transcription worker.

Launch via run_worker.bat (double-click) or run_worker.sh (Mac/Linux), or:
    python worker.py
"""

import os
import platform
import socket
import threading
import time

import psutil
from dotenv import load_dotenv

load_dotenv()

WORKER_ID          = socket.gethostname()
HEARTBEAT_INTERVAL = 30    # seconds between DB heartbeat pings
STALE_TIMEOUT      = 300   # seconds before a silent worker's jobs are re-queued
POLL_INTERVAL      = 15    # seconds between job polls when idle
MAX_RETRIES        = 3


# ---------------------------------------------------------------------------
# DB helpers (worker-specific)
# ---------------------------------------------------------------------------

def register_worker(conn):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO worker_nodes (worker_id, hostname, status)
            VALUES (%s, %s, 'idle')
            ON CONFLICT (worker_id) DO UPDATE SET
                hostname       = EXCLUDED.hostname,
                status         = 'idle',
                last_heartbeat = NOW()
        """, (WORKER_ID, socket.gethostname()))
    conn.commit()


def set_status(conn, status: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE worker_nodes SET status = %s, last_heartbeat = NOW()
            WHERE worker_id = %s
        """, (status, WORKER_ID))
    conn.commit()


def ping_heartbeat(conn):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE worker_nodes SET last_heartbeat = NOW()
            WHERE worker_id = %s
        """, (WORKER_ID,))
    conn.commit()


def requeue_stale(conn):
    """Re-queue jobs whose worker hasn't pinged in STALE_TIMEOUT seconds."""
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE job_queue SET status = 'queued', worker_id = NULL, started_at = NULL
            WHERE status = 'processing'
              AND worker_id IN (
                  SELECT worker_id FROM worker_nodes
                  WHERE last_heartbeat < NOW() - INTERVAL '{STALE_TIMEOUT} seconds'
              )
        """)
        count = cur.rowcount
    conn.commit()
    if count:
        print(f"[worker] Re-queued {count} stale job(s) from dead workers.", flush=True)


def claim_job(conn):
    """Atomically claim the oldest queued job. Returns (job_id, video_id) or None."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE job_queue
            SET status = 'processing', worker_id = %s, started_at = NOW()
            WHERE id = (
                SELECT id FROM job_queue
                WHERE status = 'queued'
                ORDER BY queued_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, video_id
        """, (WORKER_ID,))
        row = cur.fetchone()
    conn.commit()
    return row  # (job_id, video_id) or None


def complete_job(conn, job_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE job_queue SET status = 'completed', completed_at = NOW()
            WHERE id = %s
        """, (job_id,))
    conn.commit()


def fail_job(conn, job_id: int, error: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE job_queue SET
                retries   = retries + 1,
                error_msg = %s,
                worker_id = NULL,
                started_at = NULL,
                status = CASE WHEN retries + 1 >= %s THEN 'failed' ELSE 'queued' END
            WHERE id = %s
        """, (str(error)[:500], MAX_RETRIES, job_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------

def _heartbeat_loop(conn):
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        try:
            ping_heartbeat(conn)
        except Exception:
            pass  # Don't crash the thread — main loop will detect DB issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _collect_device_info(model_size: str) -> dict:
    cpu = platform.processor() or platform.machine()
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    os_name = f"{platform.system()} {platform.release()}"
    return {"os": os_name, "cpu": cpu, "ram_gb": ram_gb, "whisper_model": model_size}


def main():
    from pipeline.fetcher import fetch_video_meta
    from pipeline.processor import process_video
    from pipeline.storage import get_connection, record_benchmark, update_worker_device_info
    from pipeline.transcriber import load_model

    # Sanity-check env
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set in .env — cannot connect to database.")
        input("Press Enter to exit...")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env — cannot generate embeddings.")
        input("Press Enter to exit...")
        return

    model_size = os.environ.get("WHISPER_MODEL", "large-v3")

    print(f"[worker] Host      : {WORKER_ID}", flush=True)
    print(f"[worker] Model     : {model_size}", flush=True)
    print(f"[worker] Connecting to database ...", flush=True)

    conn = get_connection()
    register_worker(conn)

    device = _collect_device_info(model_size)
    update_worker_device_info(conn, WORKER_ID, **device)
    print(f"[worker] Device    : {device['os']} | {device['cpu']} | {device['ram_gb']} GB RAM", flush=True)
    print(f"[worker] Registered in DB.", flush=True)

    # Background heartbeat
    threading.Thread(target=_heartbeat_loop, args=(conn,), daemon=True).start()

    print(f"[worker] Loading Whisper model ...", flush=True)
    model = load_model(model_size)

    print(f"[worker] Ready — polling every {POLL_INTERVAL}s. Press Ctrl-C to stop.\n", flush=True)

    try:
        while True:
            requeue_stale(conn)
            job = claim_job(conn)

            if job is None:
                set_status(conn, "idle")
                time.sleep(POLL_INTERVAL)
                continue

            job_id, video_id = job
            set_status(conn, "busy")
            print(f"[worker] Claimed job {job_id}: {video_id}", flush=True)

            meta = fetch_video_meta(video_id)
            print(f"[worker] Title: {meta.get('title') or video_id}", flush=True)

            try:
                stats = process_video(conn, model, model_size, meta,
                                      log=lambda m: print(m, flush=True))
                complete_job(conn, job_id)
                record_benchmark(conn, WORKER_ID, job_id, video_id, model_size,
                                 stats["audio_duration_secs"],
                                 stats["transcribe_secs"],
                                 stats["word_count"])
                print(f"[worker] Job {job_id} complete — "
                      f"{stats['word_count']} words in {stats['transcribe_secs']}s "
                      f"({round(stats['word_count'] / stats['transcribe_secs'] * 60) if stats['transcribe_secs'] else 0} WPM).\n",
                      flush=True)
            except Exception as exc:
                print(f"[worker] Job {job_id} failed: {exc}", flush=True)
                fail_job(conn, job_id, str(exc))

    except KeyboardInterrupt:
        print("\n[worker] Shutting down ...", flush=True)
    finally:
        set_status(conn, "offline")
        conn.close()


if __name__ == "__main__":
    main()
