CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS videos (
    video_id    TEXT PRIMARY KEY,
    title       TEXT,
    channel     TEXT,
    url         TEXT,
    duration    INT,
    uploaded_at DATE,
    processed   BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    video_id    TEXT REFERENCES videos(video_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text        TEXT NOT NULL,
    start_sec   FLOAT,
    end_sec     FLOAT,
    embedding   vector(1536),
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (video_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
