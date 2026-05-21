"""Create a homing video from Blender homing frames and the trajectory summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _draw_trajectory_panel(
    trajectory: list[list[float]],
    step_index: int,
    successes: list[bool],
    run_index: int,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    img = Image.new("RGB", size, (248, 249, 247))
    draw = ImageDraw.Draw(img)
    title_font = _font(26)
    label_font = _font(18)
    small_font = _font(15)

    xy = np.asarray([[p[0], p[1]] for p in trajectory], dtype=float)
    home = np.asarray([0.0, 0.0])
    shown = xy[: min(step_index + 2, len(xy))]
    all_points = np.vstack([xy, home[None, :]])
    pad = 1.3
    min_xy = all_points.min(axis=0) - pad
    max_xy = all_points.max(axis=0) + pad
    span = np.maximum(max_xy - min_xy, 1.0)

    left, top, right, bottom = 54, 80, width - 34, height - 72
    plot_w = right - left
    plot_h = bottom - top

    def to_px(point):
        nx = (point[0] - min_xy[0]) / span[0]
        ny = (point[1] - min_xy[1]) / span[1]
        return int(left + nx * plot_w), int(bottom - ny * plot_h)

    draw.text((28, 24), f"Bee-Nav Homing Run {run_index:02d}", fill=(25, 32, 35), font=title_font)
    status = "success" if successes[run_index] else "not reached"
    draw.text((30, height - 42), f"step {min(step_index + 1, len(xy) - 1)} / {len(xy) - 1}   {status}", fill=(50, 58, 62), font=label_font)

    for i in range(6):
        x = left + i * plot_w / 5
        y = top + i * plot_h / 5
        draw.line((x, top, x, bottom), fill=(220, 224, 220), width=1)
        draw.line((left, y, right, y), fill=(220, 224, 220), width=1)

    hx, hy = to_px(home)
    draw.line((hx - 12, hy - 12, hx + 12, hy + 12), fill=(210, 40, 40), width=4)
    draw.line((hx - 12, hy + 12, hx + 12, hy - 12), fill=(210, 40, 40), width=4)
    draw.text((hx + 10, hy - 30), "home", fill=(160, 25, 25), font=small_font)

    if len(shown) > 1:
        points = [to_px(p) for p in shown]
        draw.line(points, fill=(42, 120, 195), width=5)
        for p in points[:-1]:
            draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(42, 120, 195))
        current = points[-1]
        draw.ellipse((current[0] - 9, current[1] - 9, current[0] + 9, current[1] + 9), fill=(255, 150, 35))

    start = to_px(xy[0])
    draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7), fill=(35, 150, 80))
    draw.text((start[0] + 8, start[1] + 8), "start", fill=(30, 110, 65), font=small_font)
    return img


def make_video(run_dir: Path, run_index: int | None, fps: int) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    successes = summary["successes"]
    if run_index is None:
        run_index = next((idx for idx, ok in enumerate(successes) if ok), 0)

    trajectory = summary["trajectories"][run_index]
    frame_dir = run_dir / "homing_frames"
    frame_paths = sorted(frame_dir.glob(f"run_{run_index:02d}_step_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No homing frames found for run {run_index:02d} in {frame_dir}")

    out_path = run_dir / f"kenney_homing_run_{run_index:02d}.mp4"
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open OpenCV VideoWriter for MP4 output.")

    for step, frame_path in enumerate(frame_paths):
        pov = Image.open(frame_path).convert("RGB").resize((720, 720), Image.Resampling.BICUBIC)
        panel = _draw_trajectory_panel(trajectory, step, successes, run_index, (560, 720))
        composed = Image.new("RGB", (width, height), (255, 255, 255))
        composed.paste(pov, (0, 0))
        composed.paste(panel, (720, 0))
        arr = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)
        for _ in range(8):
            writer.write(arr)

    writer.release()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(ROOT / "windows_blender_kenney_run"))
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    out = make_video(Path(args.run_dir), args.run_index, args.fps)
    print(out)


if __name__ == "__main__":
    main()
