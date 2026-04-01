#!/usr/bin/env bash
# Sahil AI - Transcription Worker (Mac / Linux)
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  Sahil AI - Distributed Transcription"
echo "============================================"
echo

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    echo "Install via Homebrew:  brew install python"
    exit 1
fi

# Install / update dependencies
echo "[Setup] Installing dependencies..."
pip3 install -r requirements.txt -q
echo "[Setup] Dependencies OK."
echo

# Launch worker
python3 worker.py

echo
echo "Worker stopped."
