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
        # Release any jobs this worker owned from a previous crashed session
        cur.execute("""
            UPDATE job_queue
            SET status = 'queued', worker_id = NULL, started_at = NULL
            WHERE worker_id = %s AND status = 'processing'
        """, (WORKER_ID,))
        released = cur.rowcount
    conn.commit()
    if released:
        print(f"[worker] Released {released} stale job(s) from previous session.", flush=True)


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
# Connection helpers
# ---------------------------------------------------------------------------

def _new_conn():
    from pipeline.storage import get_connection
    return get_connection()


def _ensure_conn(conn):
    """Return conn if alive, otherwise reconnect and re-register."""
    try:
        conn.cursor().execute("SELECT 1")
        return conn
    except Exception:
        print("[worker] DB connection lost — reconnecting ...", flush=True)
        try:
            conn.close()
        except Exception:
            pass
        new = _new_conn()
        register_worker(new)
        print("[worker] Reconnected.", flush=True)
        return new


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------

_conn_holder = {}   # mutable dict so the heartbeat thread always uses latest conn


def _heartbeat_loop():
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        conn = _conn_holder.get("conn")
        if conn is None:
            continue
        try:
            ping_heartbeat(conn)
        except Exception:
            pass  # main loop will reconnect


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
    _conn_holder["conn"] = conn

    device = _collect_device_info(model_size)
    update_worker_device_info(conn, WORKER_ID, **device)
    print(f"[worker] Device    : {device['os']} | {device['cpu']} | {device['ram_gb']} GB RAM", flush=True)
    print(f"[worker] Registered in DB.", flush=True)

    # Background heartbeat (uses _conn_holder so it follows reconnects)
    threading.Thread(target=_heartbeat_loop, daemon=True).start()

    print(f"[worker] Loading Whisper model ...", flush=True)
    model = load_model(model_size)

    print(f"[worker] Ready — polling every {POLL_INTERVAL}s. Press Ctrl-C to stop.\n", flush=True)

    try:
        while True:
            conn = _ensure_conn(conn)
            _conn_holder["conn"] = conn

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

            # Fresh connection per job — long jobs (hours) would kill a shared conn
            proc_conn = _new_conn()
            try:
                stats = process_video(proc_conn, model, model_size, meta,
                                      log=lambda *a, **kw: print(*a, flush=True, **kw))
                conn = _ensure_conn(conn)
                _conn_holder["conn"] = conn
                complete_job(conn, job_id)
                record_benchmark(conn, WORKER_ID, job_id, video_id, model_size,
                                 stats["audio_duration_secs"],
                                 stats["transcribe_secs"],
                                 stats["word_count"])
                wpm = round(stats["word_count"] / stats["transcribe_secs"] * 60) if stats["transcribe_secs"] else 0
                print(f"[worker] Job {job_id} complete — "
                      f"{stats['word_count']} words in {stats['transcribe_secs']}s "
                      f"({wpm} WPM).\n", flush=True)
            except Exception as exc:
                print(f"[worker] Job {job_id} failed: {exc}", flush=True)
                conn = _ensure_conn(conn)
                _conn_holder["conn"] = conn
                fail_job(conn, job_id, str(exc))
            finally:
                try:
                    proc_conn.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[worker] Shutting down ...", flush=True)
    finally:
        try:
            set_status(conn, "offline")
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
