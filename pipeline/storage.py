import os
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


def get_connection():
    database_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(database_url)
    register_vector(conn)
    return conn


def init_db(conn):
    schema_path = Path(__file__).parent.parent / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
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
        cur.execute(
            "UPDATE videos SET processed = TRUE WHERE video_id = %s", (video_id,)
        )
    conn.commit()


def upsert_chunks(conn, video_id: str, chunks: list):
    rows = [
        (
            video_id,
            chunk["chunk_index"],
            chunk["text"],
            chunk["start_sec"],
            chunk["end_sec"],
            np.array(chunk["embedding"], dtype=np.float32),
        )
        for chunk in chunks
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO chunks (video_id, chunk_index, text, start_sec, end_sec, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id, chunk_index) DO UPDATE SET
                text      = EXCLUDED.text,
                start_sec = EXCLUDED.start_sec,
                end_sec   = EXCLUDED.end_sec,
                embedding = EXCLUDED.embedding
            """,
            rows,
        )
    conn.commit()


def search_chunks(conn, embedding: list, top_k: int = 5) -> list:
    vec = np.array(embedding, dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                c.video_id,
                v.title,
                c.text,
                c.start_sec,
                c.end_sec,
                1 - (c.embedding <=> %s) AS similarity
            FROM chunks c
            JOIN videos v ON v.video_id = c.video_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (vec, vec, top_k),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
