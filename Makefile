.PHONY: setup deps models build test run clean help

MODELS_DIR := models
YOLO_MODEL := $(MODELS_DIR)/yolo11s.onnx
SCRFD_MODEL := $(MODELS_DIR)/scrfd_10g.onnx
ARCFACE_MODEL := $(MODELS_DIR)/w600k_r50.onnx
INSIGHTFACE_BASE := https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# One-shot setup
# ---------------------------------------------------------------------------

setup: deps models build ## Full one-click setup: install deps + download models + build
	@echo "\n✅ Setup complete. Run 'make run INPUT=<video>' to start."

deps: ## Install system & Python dependencies
	@echo "==> Checking system dependencies..."
	@command -v ffmpeg >/dev/null 2>&1 || { echo "Installing ffmpeg..."; brew install ffmpeg; }
	@command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
	@python3 -c "import ultralytics" 2>/dev/null || { echo "Installing ultralytics..."; pip3 install ultralytics; }
	@echo "    All dependencies OK."

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

models: $(YOLO_MODEL) $(SCRFD_MODEL) $(ARCFACE_MODEL) ## Download all ONNX models
	@echo "==> All models ready:"
	@ls -lh $(MODELS_DIR)/*.onnx

$(YOLO_MODEL):
	@echo "==> Exporting YOLO11s to ONNX..."
	@mkdir -p $(MODELS_DIR)
	@python3 -c "\
from ultralytics import YOLO; \
YOLO('yolo11s.pt').export(format='onnx', imgsz=640, simplify=True)"
	@mv yolo11s.onnx $@
	@echo "    -> $@"

$(SCRFD_MODEL):
	@echo "==> Downloading SCRFD 10g..."
	@mkdir -p $(MODELS_DIR)
	@curl -L --progress-bar -o $@ "$(INSIGHTFACE_BASE)/det_10g.onnx"
	@echo "    -> $@"

$(ARCFACE_MODEL):
	@echo "==> Downloading ArcFace w600k_r50..."
	@mkdir -p $(MODELS_DIR)
	@curl -L --progress-bar -o $@ "$(INSIGHTFACE_BASE)/w600k_r50.onnx"
	@echo "    -> $@"

# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------

build: ## Build in release mode
	cargo build --release

test: ## Run tests
	cargo test

run: ## Run pipeline (usage: make run INPUT=<file|dir|rtsp_url> [FPS=30])
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage:"; \
		echo "  make run INPUT=video.mp4           # video file"; \
		echo "  make run INPUT=frames/             # image directory"; \
		echo "  make run INPUT=rtsps://user:pass@ip:554/stream  # RTSP stream"; \
		exit 1; \
	fi
	cargo run --release -- \
		-i "$(INPUT)" \
		$(if $(FPS),--fps $(FPS),)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

mark-door: ## Mark door zone interactively (usage: make mark-door INPUT=video.mp4 [FRAME=0] [NAME=front_door])
	@if [ -z "$(INPUT)" ]; then echo "Usage: make mark-door INPUT=<video_or_image> [FRAME=0] [NAME=front_door]"; exit 1; fi
	python3 scripts/mark_door.py "$(INPUT)" --frame $(or $(FRAME),0) --name "$(or $(NAME),front_door)"

clean: ## Remove build artifacts
	cargo clean

clean-models: ## Remove downloaded models
	rm -f $(MODELS_DIR)/*.onnx
