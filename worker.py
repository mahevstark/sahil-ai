"""
worker.py — distributed transcription worker.

Launch via run_worker.bat (double-click) or run_worker.sh (Mac/Linux), or:
    python worker.py
"""

import os
import platform
import signal
import socket
import sys
import threading
import time
import traceback

import psutil
from dotenv import load_dotenv

load_dotenv()

from pipeline.logger import get_logger

log = get_logger("worker")

WORKER_ID          = os.environ.get("WORKER_ID", socket.gethostname())
HEARTBEAT_INTERVAL = 30    # seconds between DB heartbeat pings
STALE_TIMEOUT      = 300   # seconds before a silent worker's jobs are re-queued
POLL_INTERVAL      = 15    # seconds between job polls when idle
MAX_RETRIES        = 3

# Shutdown event — set by SIGTERM or Ctrl-C for graceful exit
_shutdown = threading.Event()


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
        log.info("Released %d stale job(s) from previous session.", released)


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
        log.info("Re-queued %d stale job(s) from dead workers.", count)


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
        log.warning("DB connection lost — reconnecting ...")
        try:
            conn.close()
        except Exception:
            pass
        new = _new_conn()
        register_worker(new)
        log.info("Reconnected.")
        return new


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------

_conn_holder = {}   # mutable dict so the heartbeat thread always uses latest conn


def _heartbeat_loop():
    while not _shutdown.is_set():
        _shutdown.wait(HEARTBEAT_INTERVAL)
        if _shutdown.is_set():
            break
        conn = _conn_holder.get("conn")
        if conn is None:
            continue
        try:
            ping_heartbeat(conn)
        except Exception as exc:
            log.debug("Heartbeat failed: %s", exc)


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

    # Handle SIGTERM (from systemd, Docker, etc.) for graceful shutdown
    def _handle_signal(signum, frame):
        log.info("Received signal %d — shutting down ...", signum)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)

    # Sanity-check env
    if not os.environ.get("DATABASE_URL"):
        log.error("DATABASE_URL not set in .env — cannot connect to database.")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set in .env — cannot generate embeddings.")
        sys.exit(1)

    model_size = os.environ.get("WHISPER_MODEL", "large-v3")

    log.info("Worker ID  : %s", WORKER_ID)
    log.info("Model      : %s", model_size)
    log.info("Connecting to database ...")

    try:
        conn = get_connection()
    except Exception as exc:
        log.error("Cannot connect to database: %s", exc)
        log.error("Check DATABASE_URL and ensure PostgreSQL is running with pgvector installed.")
        sys.exit(1)

    register_worker(conn)
    _conn_holder["conn"] = conn

    device = _collect_device_info(model_size)
    update_worker_device_info(conn, WORKER_ID, **device)
    log.info("Device     : %s | %s | %s GB RAM", device['os'], device['cpu'], device['ram_gb'])
    log.info("Registered in DB.")

    # Background heartbeat (uses _conn_holder so it follows reconnects)
    threading.Thread(target=_heartbeat_loop, daemon=True).start()

    log.info("Loading Whisper model ...")
    model = load_model(model_size)

    log.info("Ready — polling every %ds. Press Ctrl-C to stop.\n", POLL_INTERVAL)

    try:
        while not _shutdown.is_set():
            conn = _ensure_conn(conn)
            _conn_holder["conn"] = conn

            requeue_stale(conn)
            job = claim_job(conn)

            if job is None:
                set_status(conn, "idle")
                _shutdown.wait(POLL_INTERVAL)
                continue

            job_id, video_id = job
            set_status(conn, "busy")
            log.info("Claimed job %d: %s", job_id, video_id)

            meta = fetch_video_meta(video_id)
            log.info("Title: %s", meta.get('title') or video_id)

            # Fresh connection per job — long jobs (hours) would kill a shared conn
            proc_conn = _new_conn()
            try:
                def _log_fn(*args, **kwargs):
                    kwargs.pop("flush", None)
                    kwargs.pop("end", None)
                    msg = " ".join(str(a) for a in args)
                    log.info(msg)

                stats = process_video(proc_conn, model, model_size, meta, log=_log_fn)
                conn = _ensure_conn(conn)
                _conn_holder["conn"] = conn
                complete_job(conn, job_id)
                record_benchmark(conn, WORKER_ID, job_id, video_id, model_size,
                                 stats["audio_duration_secs"],
                                 stats["transcribe_secs"],
                                 stats["word_count"])
                wpm = round(stats["word_count"] / stats["transcribe_secs"] * 60) if stats["transcribe_secs"] else 0
                log.info("Job %d complete — %d words in %ds (%d WPM).",
                         job_id, stats['word_count'], stats['transcribe_secs'], wpm)
            except Exception as exc:
                log.error("Job %d failed: %s", job_id, exc)
                log.debug("Traceback:\n%s", traceback.format_exc())
                conn = _ensure_conn(conn)
                _conn_holder["conn"] = conn
                fail_job(conn, job_id, str(exc))
            finally:
                try:
                    proc_conn.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        log.info("Shutting down ...")
    finally:
        try:
            set_status(conn, "offline")
            conn.close()
        except Exception:
            pass
        log.info("Worker stopped.")


if __name__ == "__main__":
    main()
