# Sahil AI

Turn YouTube videos into a searchable, queryable knowledge base.

Point it at channels, it transcribes everything across distributed workers, and you ask questions in natural language — getting answers with exact video links and timestamps.

```
YouTube Channels  -->  Distributed Whisper  -->  Embeddings  -->  Q&A Knowledge Base
                       Transcription (GPU/CPU)    (pgvector)       (RAG + GPT)
```

## Features

- **Bulk channel ingestion** — feed entire YouTube channels, playlists, or individual videos
- **Distributed workers** — run on multiple PCs (gaming rigs, cloud VMs, laptops). Workers auto-claim jobs from a shared queue
- **Crash resilient** — if a machine shuts down, progress is saved per-chunk. On restart, it picks up where it left off
- **Semantic search** — search across all videos by meaning, not keywords
- **Q&A with memory** — ask follow-up questions with conversation context. Answers cite exact video timestamps
- **Live dashboard** — monitor workers, queue progress, and per-channel completion in real time

## Quick Start

### Option 1: Docker (recommended)

```bash
git clone https://github.com/yourname/sahil-ai.git && cd sahil-ai
cp .env.example .env          # fill in OPENAI_API_KEY
docker compose up -d           # starts PostgreSQL + 1 worker
docker compose up -d --scale worker=4   # scale to 4 workers
```

### Option 2: Manual Setup

**Prerequisites:** Python 3.10+, ffmpeg, PostgreSQL with [pgvector](https://github.com/pgvector/pgvector)

```bash
git clone https://github.com/yourname/sahil-ai.git && cd sahil-ai
pip install -r requirements.txt
cp .env.example .env          # fill in DATABASE_URL + OPENAI_API_KEY

# Interactive console
python main.py

# Or run a worker
python worker.py
```

## How It Works

### 1. Queue Videos

```bash
# Interactive
python main.py   # select "Queue videos" -> enter channel URL

# CLI
python main.py queue --channel "https://youtube.com/@channel"
python main.py queue --videos "video_id_1,video_id_2"
```

### 2. Process with Distributed Workers

Start workers on any machine connected to the same database:

```bash
# Mac/Linux (auto-installs ffmpeg, auto-restarts on crash)
./run_worker.sh

# Windows (double-click)
run_worker.bat

# Or directly
python worker.py
```

Each worker:
- Registers itself in the database
- Claims the oldest queued video (atomic, no conflicts)
- Downloads audio, transcribes with Whisper, generates embeddings
- Saves progress per 3-minute chunk (crash-safe)
- Marks job complete, picks up the next one

### 3. Search & Ask Questions

```bash
# Interactive (with conversation memory)
python main.py   # select "Ask a question"

# CLI
python main.py search "authentication architecture"
python main.py ask "What did they say about database design?"
python main.py summarize VIDEO_ID
```

## Adding More Workers

Any machine with Python, ffmpeg, and network access to the database can run a worker:

1. Clone the repo
2. Copy `.env` with your `DATABASE_URL` and `OPENAI_API_KEY`
3. Run `./run_worker.sh` (or `run_worker.bat` on Windows)

Multiple workers on the same machine:
```bash
WORKER_ID=worker-1 python worker.py &
WORKER_ID=worker-2 python worker.py &
```

Auto-start on boot: see [deploy/DEPLOY.md](deploy/DEPLOY.md) for systemd, launchd, and Windows Task Scheduler instructions.

## Architecture

```
                          +------------------+
                          |   PostgreSQL     |
                          |   + pgvector     |
                          |                  |
                          |  videos          |
                          |  chunks (embed)  |
                          |  job_queue       |
                          |  worker_nodes    |
                          +--------+---------+
                                   |
                 +-----------------+-----------------+
                 |                 |                 |
          +------+------+  +------+------+  +------+------+
          |   Worker 1  |  |   Worker 2  |  |   Worker 3  |
          |  (your PC)  |  | (gaming PC) |  | (cloud VM)  |
          +------+------+  +------+------+  +------+------+
                 |                 |                 |
           Whisper GPU/CPU   Whisper GPU       Whisper CPU
```

**Processing pipeline per video:**
1. Claim job from queue (atomic `FOR UPDATE SKIP LOCKED`)
2. Download audio via yt-dlp
3. Split into 3-minute chunks
4. Transcribe each chunk with Whisper (resume-safe)
5. Group segments into semantic chunks (~1500 chars)
6. Generate embeddings via OpenAI
7. Store in PostgreSQL with pgvector

**Search & Q&A pipeline:**
1. Embed user query via OpenAI
2. Cosine similarity search across all chunks (pgvector)
3. Feed top results + question to GPT-4o-mini
4. Return answer with timestamped source citations

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `OPENAI_API_KEY` | Yes | — | For embeddings + Q&A |
| `WHISPER_MODEL` | No | `large-v3` | Model size: `base`, `small`, `medium`, `large-v3` |
| `WORKER_ID` | No | hostname | Unique worker identifier |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | No | stdout | Path to log file |

## CLI Reference

```bash
python main.py                          # interactive menu
python main.py queue --channel URL      # queue entire channel
python main.py queue --videos ID1,ID2   # queue specific videos
python main.py status                   # show queue + worker status
python main.py search "query"           # semantic search
python main.py ask "question"           # RAG Q&A
python main.py summarize VIDEO_ID       # summarize a video
python main.py run --channel URL        # process directly (no queue)
```

## License

MIT
