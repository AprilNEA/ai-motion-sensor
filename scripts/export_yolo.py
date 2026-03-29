#!/usr/bin/env python3
"""Export YOLO11s to ONNX format.

Usage:
    pip install ultralytics
    python3 scripts/export_yolo.py [--model yolo11s] [--output models/yolo11s.onnx]
"""

import argparse
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", default="yolo11s", help="Model name (default: yolo11s)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size (default: 640)")
    parser.add_argument("--output", default="models/yolo11s.onnx", help="Output path")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        raise SystemExit(1)

    work_dir = Path(tempfile.mkdtemp())
    try:
        model = YOLO(f"{args.model}.pt")
        model.export(format="onnx", imgsz=args.imgsz, simplify=True)

        # ultralytics puts the .onnx next to the .pt
        exported = next(Path(".").glob(f"{args.model}.onnx"), None)
        if exported is None:
            exported = next(work_dir.glob(f"{args.model}.onnx"), None)
        if exported is None:
            print("ERROR: exported .onnx not found")
            raise SystemExit(1)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(out))
        print(f"Saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        # Clean up .pt file if downloaded to cwd
        pt_file = Path(f"{args.model}.pt")
        if pt_file.exists():
            pt_file.unlink()


if __name__ == "__main__":
    main()
