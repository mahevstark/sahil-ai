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

# Install ffmpeg if missing
if ! command -v ffmpeg &>/dev/null; then
    echo "[Setup] ffmpeg not found. Installing..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y ffmpeg
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg
    else
        echo "ERROR: Cannot install ffmpeg automatically."
        echo "Please install it manually: https://ffmpeg.org/download.html"
        exit 1
    fi
    echo "[Setup] ffmpeg installed."
else
    echo "[Setup] ffmpeg OK."
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
