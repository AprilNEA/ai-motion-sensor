#!/usr/bin/env bash
set -euo pipefail

# Model hosting — change this to your own GitHub Release URL after uploading.
# To upload: gh release create models-v1 models/*.onnx --repo arcboxlabs/ai-motion-sensor
GITHUB_BASE="${MODEL_BASE_URL:-https://github.com/arcboxlabs/ai-motion-sensor/releases/download/models-v1}"
INSIGHTFACE_BASE="https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l"

MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODELS_DIR"

download() {
    local name="$1" url="$2" out="$3"
    if [ -f "$out" ]; then
        echo "  ✓ $name already exists, skipping."
        return
    fi
    echo "  ↓ $name ..."
    curl -L --progress-bar -o "$out" "$url"
}

echo "=== Downloading ONNX models ==="
echo ""

download "YOLO11s (36 MB)" \
    "$GITHUB_BASE/yolo11s.onnx" \
    "$MODELS_DIR/yolo11s.onnx"

download "SCRFD face detection (16 MB)" \
    "$INSIGHTFACE_BASE/det_10g.onnx" \
    "$MODELS_DIR/scrfd_10g.onnx"

download "ArcFace recognition (166 MB)" \
    "$INSIGHTFACE_BASE/w600k_r50.onnx" \
    "$MODELS_DIR/w600k_r50.onnx"

echo ""
echo "=== All models ready ==="
ls -lh "$MODELS_DIR"/*.onnx
