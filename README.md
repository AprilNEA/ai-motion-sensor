<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/ONNX_Runtime-007808?style=flat&logo=onnx&logoColor=white" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/CoreML-000000?style=flat&logo=apple&logoColor=white" alt="CoreML">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</p>

# AI Motion Sensor

Real-time exit-intent detection system built in Rust. It watches a camera feed, tracks people, recognises faces, and determines when someone intends to walk out the door — then optionally unlocks it for them.

```mermaid
flowchart TD
    subgraph Input
        CAM["Camera / Video / RTSP"]
    end

    subgraph ORT["ort crate &lt;ONNX Runtime Rust&gt;"]
        direction TB
        EP["CoreML EP → ANE / GPU / CPU"]
    end

    subgraph Detection["Detection"]
        YOLO["YOLO11s.onnx\nPerson Detection"]
        SCRFD["SCRFD.onnx\nFace Detection + Landmarks"]
    end

    subgraph FaceID["Face Recognition"]
        ALIGN["Crop + Affine Alignment"]
        ARCFACE["ArcFace.onnx\n512D Embedding"]
        FACEDB[("Face Database\nCosine Similarity")]
    end

    subgraph Tracking
        BT["ByteTrack\nKalman Filter + IoU Association"]
    end

    subgraph Analysis["Intent Analysis"]
        SPATIAL["Spatial Signals\nDirection / Distance / Zone / Speed"]
        SCORER["Intent Scorer\nWeighted Fusion + Sliding Window"]
    end

    subgraph Action["Door Control"]
        CTRL["DoorController Trait"]
        UNIFI["UniFi Access API"]
    end

    CAM --> ORT
    ORT --> YOLO
    ORT --> SCRFD
    YOLO -- "BBox + Confidence" --> BT
    SCRFD -- "5-Point Landmarks" --> ALIGN --> ARCFACE
    ARCFACE -- "512D Vector" --> FACEDB
    FACEDB -- "Identity Label" --> BT
    BT -- "Tracks + Trajectory" --> SPATIAL
    SPATIAL --> SCORER
    SCORER -- "Alert 🔔" --> CTRL
    CTRL --> UNIFI

    style Input fill:#f5f5f5,stroke:#bbb
    style ORT fill:#e8eaf6,stroke:#7986cb
    style YOLO fill:#e0f2f1,stroke:#4db6ac
    style SCRFD fill:#fce4ec,stroke:#ef9a9a
    style ARCFACE fill:#fce4ec,stroke:#ef9a9a
    style FACEDB fill:#fff3e0,stroke:#ffb74d
    style BT fill:#e0f2f1,stroke:#4db6ac
    style SCORER fill:#e8eaf6,stroke:#7986cb
    style UNIFI fill:#fff3e0,stroke:#ffb74d
```

## Features

- **Person detection** — YOLO11 with configurable confidence and NMS
- **Multi-object tracking** — ByteTrack with Kalman filter, handles occlusions and re-identification
- **Face recognition** — SCRFD face detection + ArcFace 512D embeddings with similarity matching
- **Exit intent scoring** — Weighted multi-signal fusion (direction, distance, zone, speed) with sliding-window confirmation and hysteresis debouncing
- **Door control** — Pluggable `DoorController` trait; ships with UniFi Access integration
- **Multiple video sources** — Video files, image directories, RTSP/RTSPS streams with auto-reconnect
- **Apple Silicon optimised** — CoreML execution provider for hardware-accelerated inference on macOS

## Quick Start

### Prerequisites

- Rust 1.75+
- FFmpeg (`brew install ffmpeg` on macOS)
- Python 3 (only for `mark_door.py` helper)

### Setup

```bash
git clone https://github.com/arcboxlabs/ai-motion-sensor.git
cd ai-motion-sensor
make setup   # installs deps, downloads models, builds release
```

### Run

```bash
# Video file
make run INPUT=video.mp4

# RTSP camera stream
make run INPUT=rtsps://admin:password@192.168.1.100:554/stream

# Image directory (for testing)
make run INPUT=frames/ FPS=30
```

Or directly:

```bash
cargo run --release -- -i <input> [-c config/default.toml] [--fps 30]
```

## Configuration

Copy the example config and customise:

```bash
cp config/default.example.toml config/default.toml
```

### Door Zones

Define polygons in normalised coordinates (0–1) where doors are located. The `direction` vector points from inside toward outside.

```toml
[[door_zones]]
name = "front_door"
polygon = [[0.35, 0.37], [0.33, 0.01], [0.44, 0.01], [0.45, 0.43]]
direction = [0.115, -0.993]
```

Use the interactive tool to mark zones visually:

```bash
make mark-door INPUT=video.mp4
```

### Intent Scoring

The scorer fuses five spatial signals with configurable weights:

| Signal | Default Weight | Description |
|--------|---------------|-------------|
| `direction` | 0.30 | Movement direction toward door |
| `distance` | 0.25 | Proximity to door center |
| `in_zone` | 0.20 | Whether person is inside door polygon |
| `facing` | 0.15 | Body orientation (placeholder) |
| `walking` | 0.10 | Movement speed |

An alert fires when the sliding window confirms sustained intent:

```toml
[intent]
alert_threshold = 0.60    # minimum score
confirm_frames = 8         # window size
confirm_ratio = 0.75       # fraction above threshold
cooldown_secs = 30.0       # debounce after alert
```

### Door Control (Optional)

Connect to a door access system to auto-unlock on exit intent. Currently supports [UniFi Access](https://ui.com/door-access).

```toml
[door_control]
backend = "unifi_access"
host = "https://192.168.1.1:12445"
token = "YOUR_API_TOKEN"

[door_control.door_name_map]
front_door = "Door 5c6a"    # maps zone name → UniFi door name
```

The `DoorController` trait is designed for extension — implement it for any access control system:

```rust
pub trait DoorController: Send + Sync {
    fn unlock(&self, door_name: &str) -> Result<()>;
    fn lock_state(&self, door_name: &str) -> Result<DoorLockState>;
}
```

## Architecture

```
src/
├── main.rs              # CLI entry point
├── pipeline.rs          # Central processing orchestration
├── config.rs            # TOML configuration structures
├── geometry.rs          # Primitives: BBox, Point2D, IoU, NMS, point-in-polygon
├── inference/
│   ├── engine.rs        # ONNX model loader (CoreML on macOS, CPU elsewhere)
│   ├── yolo.rs          # YOLO11 person detection
│   ├── scrfd.rs         # SCRFD multi-scale face detection + 5-point landmarks
│   └── arcface.rs       # ArcFace 512D face embedding with affine alignment
├── tracking/
│   ├── byte_track.rs    # ByteTrack two-stage IoU association
│   ├── kalman.rs        # 8D linear Kalman filter
│   └── track.rs         # Track state machine (Tentative → Active → Lost)
├── analysis/
│   ├── intent.rs        # Weighted intent scorer with rolling confirmation
│   ├── spatial.rs       # Spatial signal extraction
│   └── face_db.rs       # In-memory face embedding database
├── door/
│   ├── mod.rs           # DoorController trait
│   └── unifi_access.rs  # UniFi Access Developer API client
└── video/
    └── source.rs        # FrameSource trait + Ffmpeg/RTSP/ImageDir backends
```

### Pipeline Flow

Each frame goes through:

1. **Detection** — YOLO11 locates persons (runs every N frames; tracking predicts on skipped frames)
2. **Tracking** — ByteTrack associates detections across frames using Kalman-predicted IoU
3. **Face recognition** — SCRFD detects faces, ArcFace extracts embeddings, matched against known identities
4. **Spatial analysis** — Computes direction, distance, zone membership, and speed per track per door zone
5. **Intent scoring** — Weighted fusion → rolling window confirmation → alert with cooldown
6. **Door unlock** — On confirmed alert, sends unlock command via configured backend

## Models

Downloaded automatically via `make models`:

| Model | Purpose | Source |
|-------|---------|--------|
| YOLO11s | Person detection | Ultralytics |
| SCRFD-10G | Face detection + landmarks | InsightFace |
| w600k_r50 | Face embedding (512D) | InsightFace |

## License

MIT
