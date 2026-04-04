# Deploying Sahil AI Workers

Run workers on any machine to help process the video queue. Each worker connects to the same PostgreSQL database, claims jobs, and processes them independently.

## Quick Start (Any OS)

1. Clone the repo and set up `.env`:
   ```bash
   git clone <repo-url> && cd sahil-ai
   cp .env.example .env   # fill in DATABASE_URL and OPENAI_API_KEY
   ```

2. Run the worker:
   ```bash
   # Mac/Linux
   ./run_worker.sh

   # Windows (double-click or run in terminal)
   run_worker.bat
   ```

The script auto-installs ffmpeg (Mac/Linux), installs Python dependencies, and starts the worker with auto-restart on crash.

## Auto-Start on Boot

### Linux (systemd)

```bash
# Edit paths in the service file to match your install location
sudo cp deploy/sahil-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sahil-worker
sudo systemctl start sahil-worker

# Check logs
journalctl -u sahil-worker -f
```

### macOS (launchd)

```bash
# Edit paths in the plist file to match your install location
cp deploy/com.sahilai.worker.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.sahilai.worker.plist

# Check logs
tail -f /tmp/sahil-worker.log

# To stop
launchctl unload ~/Library/LaunchAgents/com.sahilai.worker.plist
```

### Windows (Task Scheduler)

1. Open Task Scheduler → Create Task
2. **General**: Name = "Sahil AI Worker", check "Run whether user is logged on or not"
3. **Triggers**: At startup, or at log on
4. **Actions**: Start a program → `run_worker.bat`, Start in → your sahil-ai folder
5. **Settings**: Check "If the task fails, restart every 1 minute", attempts = 999

## Running Multiple Workers on One Machine

Set unique `WORKER_ID` for each:

```bash
WORKER_ID=worker-gpu-1 python worker.py &
WORKER_ID=worker-gpu-2 python worker.py &
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `OPENAI_API_KEY` | Yes | — | For embeddings generation |
| `WHISPER_MODEL` | No | `large-v3` | Whisper model size (base/small/medium/large-v3) |
| `WORKER_ID` | No | hostname | Unique worker identifier |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `LOG_FILE` | No | — | Path to log file (stdout only if unset) |
