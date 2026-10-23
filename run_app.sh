#!/usr/bin/env bash
# Spider Vision HUD — Launch Script
# Usage: bash run_app.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/SpiderVision"
VENV="$APP_DIR/.venv"

echo "╔══════════════════════════════════════════════╗"
echo "║   Spider Vision HUD — Cinematic Edition      ║"
echo "╚══════════════════════════════════════════════╝"

# Check venv
if [ ! -d "$VENV" ]; then
    echo "[INFO] Creating virtual environment with Python 3.12..."
    python3.12 -m venv "$VENV"
fi

# Activate venv
source "$VENV/bin/activate"

# Install only when missing; never upgrade working packages on every launch.
if ! "$VENV/bin/python" -c 'import moderngl, glfw, glm, cv2, mediapipe, pygame, psutil, PIL, scipy, edge_tts, vosk, sounddevice; assert hasattr(cv2, "face") and hasattr(mediapipe, "solutions")' >/dev/null 2>&1; then
    echo "[INFO] Installing SpiderVision dependencies..."
    "$VENV/bin/pip" install 'moderngl>=5.10' 'glfw>=2.6' 'PyGLM>=2.7' \
        'opencv-contrib-python>=4.9,<6' 'mediapipe==0.10.14' \
        'pygame>=2.5' 'psutil>=5.9' 'Pillow>=10' 'scipy>=1.12' 'edge-tts>=6.1' \
        'vosk==0.3.44' 'sounddevice>=0.5,<1'
fi

# Launch
echo "[INFO] Starting Spider Vision HUD..."
cd "$APP_DIR"
"$VENV/bin/python" main.py
