#!/usr/bin/env python3
"""
Interactive door zone annotation tool.

Usage:
    python3 scripts/mark_door.py <video_or_image> [--frame N]

Instructions:
    1. Left-click to place polygon vertices around the door area
    2. Right-click (or press Enter) to close the polygon
    3. Then left-click INSIDE the room, and left-click OUTSIDE (through the door)
       to define the exit direction vector
    4. Press 'q' to finish — the TOML config is printed to stdout

    Press 'z' to undo the last point, 'r' to reset all points.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import matplotlib
    # Prefer macOS native backend; fall back to any available GUI backend.
    for _backend in ("macosx", "TkAgg", "Qt5Agg", "GTK3Agg"):
        try:
            matplotlib.use(_backend)
            break
        except ImportError:
            continue
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
except ImportError:
    print("ERROR: matplotlib is required.  Install with:  pip3 install matplotlib")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.  Install with:  pip3 install Pillow")
    sys.exit(1)


def extract_frame(video_path: str, frame_num: int = 0) -> Image.Image:
    """Extract a single frame from a video using ffmpeg."""
    suffix = Path(video_path).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return Image.open(video_path)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    # Seek to the requested frame and grab one png.
    time_sec = frame_num / 30.0  # assume 30 fps for seek
    cmd = [
        "ffmpeg", "-y", "-ss", str(time_sec), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", tmp_path,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    return Image.open(tmp_path)


class DoorAnnotator:
    def __init__(self, img: Image.Image):
        self.img = np.array(img)
        self.h, self.w = self.img.shape[:2]

        # State
        self.polygon_pts: list[tuple[float, float]] = []
        self.polygon_closed = False
        self.direction_pts: list[tuple[float, float]] = []
        self.phase = "polygon"  # "polygon" → "direction" → "done"

        # Set up figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.ax.imshow(self.img)
        self.ax.set_title(
            "Left-click: add door polygon vertex  |  Right-click/Enter: close polygon",
            fontsize=11,
        )
        self.ax.axis("off")

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._redraw()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata

        if self.phase == "polygon":
            if event.button == 1:  # left click
                self.polygon_pts.append((x, y))
            elif event.button == 3:  # right click → close
                self._close_polygon()
        elif self.phase == "direction":
            if event.button == 1:
                self.direction_pts.append((x, y))
                if len(self.direction_pts) == 2:
                    self.phase = "done"
                    self.ax.set_title(
                        "Done!  Press 'q' to output config.",
                        fontsize=11, color="green",
                    )
        self._redraw()

    def _on_key(self, event):
        if event.key == "enter" and self.phase == "polygon":
            self._close_polygon()
            self._redraw()
        elif event.key == "z":  # undo
            if self.phase == "polygon" and self.polygon_pts:
                self.polygon_pts.pop()
                self._redraw()
            elif self.phase == "direction" and self.direction_pts:
                self.direction_pts.pop()
                self._redraw()
        elif event.key == "r":  # reset
            self.polygon_pts.clear()
            self.direction_pts.clear()
            self.polygon_closed = False
            self.phase = "polygon"
            self.ax.set_title(
                "Left-click: add door polygon vertex  |  Right-click/Enter: close polygon",
                fontsize=11, color="black",
            )
            self._redraw()
        elif event.key == "q":
            plt.close(self.fig)

    def _close_polygon(self):
        if len(self.polygon_pts) < 3:
            return
        self.polygon_closed = True
        self.phase = "direction"
        self.ax.set_title(
            "Click 1: inside the room  →  Click 2: outside (exit direction)",
            fontsize=11, color="blue",
        )

    def _redraw(self):
        # Remove old annotations.
        while len(self.ax.patches) > 0:
            self.ax.patches[-1].remove()
        while len(self.ax.lines) > 0:
            self.ax.lines[-1].remove()
        # Re-plot scatter via collections — simpler to just clear & redraw.
        for coll in list(self.ax.collections):
            coll.remove()

        # Polygon vertices
        if self.polygon_pts:
            xs, ys = zip(*self.polygon_pts)
            self.ax.scatter(xs, ys, c="lime", s=60, zorder=5, edgecolors="black")
            if len(self.polygon_pts) > 1:
                self.ax.plot(
                    list(xs) + ([xs[0]] if self.polygon_closed else []),
                    list(ys) + ([ys[0]] if self.polygon_closed else []),
                    "lime", linewidth=2, zorder=4,
                )

        # Filled polygon
        if self.polygon_closed:
            poly = MplPolygon(
                self.polygon_pts, closed=True,
                facecolor=(0, 1, 0, 0.15), edgecolor="lime", linewidth=2,
            )
            self.ax.add_patch(poly)

        # Direction arrow
        if len(self.direction_pts) >= 1:
            dx, dy = self.direction_pts[0]
            self.ax.scatter([dx], [dy], c="cyan", s=80, zorder=6,
                            marker="o", edgecolors="black")
        if len(self.direction_pts) == 2:
            x0, y0 = self.direction_pts[0]
            x1, y1 = self.direction_pts[1]
            self.ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                zorder=6,
            )

        self.fig.canvas.draw_idle()

    def run(self) -> tuple[list[list[float]], list[float]] | None:
        plt.show()

        if not self.polygon_closed or len(self.direction_pts) < 2:
            return None

        # Normalise coordinates to [0, 1], converting numpy floats to plain Python floats.
        norm_poly = [[round(float(x / self.w), 4), round(float(y / self.h), 4)]
                     for x, y in self.polygon_pts]

        x0, y0 = self.direction_pts[0]
        x1, y1 = self.direction_pts[1]
        dx, dy = x1 - x0, y1 - y0
        length = max((dx**2 + dy**2) ** 0.5, 1e-8)
        norm_dir = [round(float(dx / length), 4), round(float(dy / length), 4)]

        return norm_poly, norm_dir


def format_toml(name: str, polygon: list[list[float]], direction: list[float]) -> str:
    poly_str = ", ".join(str(p) for p in polygon)
    return (
        f'[[door_zones]]\n'
        f'name = "{name}"\n'
        f'polygon = [{poly_str}]\n'
        f'direction = {direction}\n'
    )


def main():
    parser = argparse.ArgumentParser(description="Mark door zone on a video/image.")
    parser.add_argument("input", help="Video file or image")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame number to extract (default: 0)")
    parser.add_argument("--name", default="front_door",
                        help="Door zone name (default: front_door)")
    args = parser.parse_args()

    print(f"Extracting frame {args.frame} from '{args.input}'...")
    img = extract_frame(args.input, args.frame)
    print(f"Image size: {img.size[0]}x{img.size[1]}")
    print()
    print("Instructions:")
    print("  1. Left-click around the door to place polygon vertices")
    print("  2. Right-click or Enter to close the polygon")
    print("  3. Click INSIDE the room, then click OUTSIDE to set exit direction")
    print("  4. Press 'q' to finish")
    print("  (z = undo, r = reset)")
    print()

    annotator = DoorAnnotator(img)
    result = annotator.run()

    if result is None:
        print("Cancelled — no polygon was completed.")
        sys.exit(1)

    polygon, direction = result
    toml_block = format_toml(args.name, polygon, direction)

    print()
    print("=" * 60)
    print("Copy this into config/default.toml (replace the existing")
    print("[[door_zones]] section):")
    print("=" * 60)
    print()
    print(toml_block)


if __name__ == "__main__":
    main()
