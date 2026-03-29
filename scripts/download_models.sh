#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODELS_DIR"

echo "=== Downloading ONNX models ==="

# --- YOLO11s ---
YOLO_OUT="$MODELS_DIR/yolo11s.onnx"
if [ ! -f "$YOLO_OUT" ]; then
    echo "[1/3] Exporting YOLO11s to ONNX via ultralytics..."
    WORK_DIR=$(mktemp -d)
    python3 -c "
import os, sys
os.chdir('$WORK_DIR')
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
model.export(format='onnx', imgsz=640, simplify=True)
"
    # ultralytics puts the .onnx next to the .pt
    find "$WORK_DIR" -name "yolo11s.onnx" -exec mv {} "$YOLO_OUT" \;
    rm -rf "$WORK_DIR"
    echo "      -> $YOLO_OUT"
else
    echo "[1/3] YOLO11s already exists, skipping."
fi

# --- InsightFace models (SCRFD + ArcFace) ---
SCRFD_OUT="$MODELS_DIR/scrfd_10g.onnx"
ARCFACE_OUT="$MODELS_DIR/w600k_r50.onnx"

INSIGHTFACE_URL="https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l"

if [ ! -f "$SCRFD_OUT" ]; then
    echo "[2/3] Downloading SCRFD 10g..."
    curl -L -o "$SCRFD_OUT" "$INSIGHTFACE_URL/det_10g.onnx"
    echo "      -> $SCRFD_OUT"
else
    echo "[2/3] SCRFD already exists, skipping."
fi

if [ ! -f "$ARCFACE_OUT" ]; then
    echo "[3/3] Downloading ArcFace w600k_r50..."
    curl -L -o "$ARCFACE_OUT" "$INSIGHTFACE_URL/w600k_r50.onnx"
    echo "      -> $ARCFACE_OUT"
else
    echo "[3/3] ArcFace already exists, skipping."
fi

echo ""
echo "=== All models ready in $MODELS_DIR ==="
ls -lh "$MODELS_DIR"/*.onnx
