import json
import os
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


def get_connection():
    database_url = os.environ["DATABASE_URL"]
    # Bootstrap: run schema on plain connection first (vector type may not exist yet)
    bootstrap = psycopg2.connect(database_url)
    schema_sql = (Path(__file__).parent.parent / "schema.sql").read_text(encoding="utf-8")
    with bootstrap.cursor() as cur:
        cur.execute(schema_sql)
    bootstrap.commit()
    bootstrap.close()
    # Fresh connection with pgvector registered
    conn = psycopg2.connect(database_url)
    register_vector(conn)
    return conn


def init_db(conn):
    schema_path = Path(__file__).parent.parent / "schema.sql"
    with conn.cursor() as cur:
        cur.execute(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def video_exists(conn, video_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM videos WHERE video_id = %s", (video_id,))
        return cur.fetchone() is not None


def upsert_video(conn, meta: dict):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO videos (video_id, title, channel, url, duration, uploaded_at)
            VALUES (%(video_id)s, %(title)s, %(channel)s, %(url)s, %(duration)s, %(uploaded_at)s)
            ON CONFLICT (video_id) DO UPDATE SET
                title       = EXCLUDED.title,
                channel     = EXCLUDED.channel,
                url         = EXCLUDED.url,
                duration    = EXCLUDED.duration,
                uploaded_at = EXCLUDED.uploaded_at
            """,
            meta,
        )
    conn.commit()


def mark_processed(conn, video_id: str):
    with conn.cursor() as cur:
        cur.execute("UPDATE videos SET processed = TRUE WHERE video_id = %s", (video_id,))
    conn.commit()


def upsert_chunks(conn, video_id: str, chunks: list):
    rows = [
        (video_id, c["chunk_index"], c["text"], c["start_sec"], c["end_sec"],
         np.array(c["embedding"], dtype=np.float32))
        for c in chunks
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO chunks (video_id, chunk_index, text, start_sec, end_sec, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id, chunk_index) DO UPDATE SET
                text = EXCLUDED.text, start_sec = EXCLUDED.start_sec,
                end_sec = EXCLUDED.end_sec, embedding = EXCLUDED.embedding
            """,
            rows,
        )
    conn.commit()


def search_chunks(conn, embedding: list, top_k: int = 5) -> list:
    vec = np.array(embedding, dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.video_id, v.title, c.text, c.start_sec, c.end_sec,
                   1 - (c.embedding <=> %s) AS similarity
            FROM chunks c JOIN videos v ON v.video_id = c.video_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (vec, vec, top_k),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Transcription job / chunk resume
# ---------------------------------------------------------------------------

def upsert_transcription_job(conn, video_id: str, model_size: str,
                              title: str, total_chunks: int):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO transcription_jobs (video_id, model_size, title, total_chunks, status)
            VALUES (%s, %s, %s, %s, 'in_progress')
            ON CONFLICT (video_id, model_size) DO UPDATE SET
                title = EXCLUDED.title, total_chunks = EXCLUDED.total_chunks,
                status = 'in_progress', updated_at = NOW()
        """, (video_id, model_size, title, total_chunks))
    conn.commit()


def mark_transcription_job_done(conn, video_id: str, model_size: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE transcription_jobs SET status = 'completed', updated_at = NOW()
            WHERE video_id = %s AND model_size = %s
        """, (video_id, model_size))
    conn.commit()


def get_completed_chunk_indices(conn, video_id: str, model_size: str) -> set:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_index FROM transcription_chunks
            WHERE video_id = %s AND model_size = %s AND status = 'completed'
        """, (video_id, model_size))
        return {row[0] for row in cur.fetchall()}


def save_transcription_chunk(conn, video_id: str, model_size: str,
                              chunk_index: int, offset_secs: float, segments: list):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO transcription_chunks
                (video_id, model_size, chunk_index, offset_secs, status, text)
            VALUES (%s, %s, %s, %s, 'completed', %s)
            ON CONFLICT (video_id, model_size, chunk_index) DO UPDATE SET
                status = 'completed', text = EXCLUDED.text, updated_at = NOW()
        """, (video_id, model_size, chunk_index, offset_secs, json.dumps(segments)))
    conn.commit()


def load_transcription_chunks(conn, video_id: str, model_size: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_index, text FROM transcription_chunks
            WHERE video_id = %s AND model_size = %s AND status = 'completed'
            ORDER BY chunk_index
        """, (video_id, model_size))
        return {row[0]: json.loads(row[1]) for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Distributed job queue
# ---------------------------------------------------------------------------

def queue_videos(conn, video_ids: list) -> int:
    """Insert video IDs into job_queue in one statement. Returns count newly added."""
    if not video_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("""
            WITH inserted AS (
                INSERT INTO job_queue (video_id)
                SELECT unnest(%s::text[])
                ON CONFLICT (video_id) DO NOTHING
                RETURNING video_id
            )
            SELECT COUNT(*) FROM inserted
        """, (list(video_ids),))
        count = cur.fetchone()[0]
    conn.commit()
    return count


def get_queue_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT status, COUNT(*) FROM job_queue GROUP BY status ORDER BY status
        """)
        return {row[0]: row[1] for row in cur.fetchall()}


def get_worker_nodes(conn) -> list:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT worker_id, hostname, status, last_heartbeat
            FROM worker_nodes ORDER BY last_heartbeat DESC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
