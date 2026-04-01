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

-- Note: ivfflat index is NOT created here on purpose.
-- ivfflat requires at least `lists` rows to build correctly.
-- Schema runs on every startup against an initially-empty table,
-- which would produce an index with 0 cluster centers → zero search results.
-- pgvector's sequential scan (<=> operator) is exact and fast up to ~100k rows.
-- To add an approximate index later, run manually after data is loaded:
--   CREATE INDEX chunks_embedding_idx ON chunks
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Tracks per-video transcription jobs (one row per video+model combination)
CREATE TABLE IF NOT EXISTS transcription_jobs (
    video_id     TEXT  NOT NULL,
    model_size   TEXT  NOT NULL,
    title        TEXT,
    status       TEXT  DEFAULT 'in_progress',  -- in_progress | completed | failed
    total_chunks INT,
    created_at   TIMESTAMP DEFAULT NOW(),
    updated_at   TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (video_id, model_size)
);

-- Tracks individual 3-min audio chunks and their transcribed text (stored as JSON segments)
CREATE TABLE IF NOT EXISTS transcription_chunks (
    id           SERIAL PRIMARY KEY,
    video_id     TEXT  NOT NULL,
    model_size   TEXT  NOT NULL,
    chunk_index  INT   NOT NULL,
    offset_secs  FLOAT NOT NULL,
    status       TEXT  DEFAULT 'pending',  -- pending | completed
    text         TEXT,                     -- JSON array of {start, end, text} segments
    created_at   TIMESTAMP DEFAULT NOW(),
    updated_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE (video_id, model_size, chunk_index)
);

-- Registered worker PCs
CREATE TABLE IF NOT EXISTS worker_nodes (
    worker_id      TEXT PRIMARY KEY,   -- hostname (stable across restarts)
    hostname       TEXT NOT NULL,
    status         TEXT DEFAULT 'idle', -- idle | busy | offline
    last_heartbeat TIMESTAMP DEFAULT NOW(),
    registered_at  TIMESTAMP DEFAULT NOW(),
    os             TEXT,
    cpu            TEXT,
    ram_gb         FLOAT,
    whisper_model  TEXT
);
-- Add device columns to existing deployments
ALTER TABLE worker_nodes ADD COLUMN IF NOT EXISTS os            TEXT;
ALTER TABLE worker_nodes ADD COLUMN IF NOT EXISTS cpu           TEXT;
ALTER TABLE worker_nodes ADD COLUMN IF NOT EXISTS ram_gb        FLOAT;
ALTER TABLE worker_nodes ADD COLUMN IF NOT EXISTS whisper_model TEXT;

-- Per-job transcription speed benchmarks
CREATE TABLE IF NOT EXISTS worker_benchmarks (
    id                  SERIAL PRIMARY KEY,
    worker_id           TEXT REFERENCES worker_nodes(worker_id),
    job_id              INT  REFERENCES job_queue(id),
    video_id            TEXT NOT NULL,
    model_size          TEXT,
    audio_duration_secs FLOAT,
    transcribe_secs     FLOAT,
    word_count          INT,
    words_per_minute    FLOAT,
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Distributed job queue (admin queues, workers claim)
CREATE TABLE IF NOT EXISTS job_queue (
    id           SERIAL PRIMARY KEY,
    video_id     TEXT NOT NULL UNIQUE,
    status       TEXT DEFAULT 'queued', -- queued | processing | completed | failed
    worker_id    TEXT REFERENCES worker_nodes(worker_id),
    queued_at    TIMESTAMP DEFAULT NOW(),
    started_at   TIMESTAMP,
    completed_at TIMESTAMP,
    error_msg    TEXT,
    retries      INT DEFAULT 0
);
