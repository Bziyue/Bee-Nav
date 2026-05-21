"""Lightweight Blender rendering backend for Bee-Nav.

The backend keeps the original training dataset contract:

    output_folder/
      dataset_navigation.csv
      Replicator/rgb/rgb_0000.png
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from insect_utils.flight_path_functions import (
    calculate_target_vectors,
    generate_bee_path,
    generate_circle_centered_on_home,
    generate_circle_centered_on_landmarks,
    generate_grid_path,
    generate_noisy_spiral_allan,
    generate_spiral_path,
    generate_uniform_area,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe")


def find_blender(blender_path: str | None = None) -> Path:
    candidates = []
    if blender_path:
        candidates.append(Path(blender_path))
    env_path = os.environ.get("BLENDER_EXE")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            DEFAULT_BLENDER,
            Path(r"C:\Program Files\Blender Foundation\Blender 4.2 LTS\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender\blender.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find blender.exe. Set BLENDER_EXE or pass blender_path in the config.")


def get_locations_path(map_path: str) -> str | None:
    area_index = map_path.find("_area")
    if area_index == -1:
        return None
    locations_path = map_path[: area_index + len("_area")] + "_locations" + map_path[area_index + len("_area") :]
    return locations_path[:-4] + ".csv"


def load_landmark_locations(locations_path: str | None) -> np.ndarray | None:
    if not locations_path or not os.path.exists(locations_path):
        return None
    points = []
    with open(locations_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                points.append([float(row[0]), float(row[1]), float(row[2])])
    return np.asarray(points, dtype=float) if points else None


def default_landmarks() -> np.ndarray:
    return np.asarray(
        [
            [1.2, 0.7, 0.0],
            [-1.4, -0.8, 0.0],
            [-0.6, 1.5, 0.0],
            [1.8, -1.5, 0.0],
        ],
        dtype=float,
    )


def generate_positions_and_rotations(config: dict):
    home_position = np.asarray(config["home_position"], dtype=float)
    shape_flight = config["shape_flight"]
    shape_params = config["shape_params"]
    noisy_path = None

    if shape_flight == "spiral":
        positions = generate_spiral_path(*shape_params, home_position)
        if config.get("use_noisy_spiral", False):
            noisy_path = generate_noisy_spiral_allan(*shape_params, home_position, *config["noise_params"])
    elif shape_flight == "bee":
        if len(shape_params) == 1:
            shape_params = [4, int(config.get("n_positions", 40)) + 3, 0.08]
        positions = generate_bee_path(*shape_params, home_position)
        if config.get("use_noisy_spiral", False):
            noisy_path = generate_bee_path(*shape_params, home_position, add_noise=True, noise_params=config["noise_params"])
    elif shape_flight == "grid":
        positions = generate_grid_path(config)
    elif shape_flight == "area":
        positions = generate_uniform_area(config["n_positions"], home_position, area=shape_params)
    elif shape_flight == "home_circle":
        positions = generate_circle_centered_on_home(config["n_positions"], shape_params[0], home_position)
    elif shape_flight == "landmark_circle":
        positions = generate_circle_centered_on_landmarks(config["n_positions"], shape_params[0], home_position)
    elif shape_flight == "debug":
        positions = [(shape_params[0], shape_params[1], shape_params[2])] * int(config.get("n_positions", 12))
    else:
        raise ValueError(f"Unknown flight shape: {shape_flight}")

    n_positions = len(positions)
    if config.get("only_point_north", False):
        rotations = [(0.0, 0.0, 0.0) for _ in range(n_positions)]
    else:
        rotations = [(0.0, 0.0, float(np.random.rand() * 360.0)) for _ in range(n_positions)]

    return positions, rotations, noisy_path


def _write_csv(config: dict, positions, rotations, targets, noisy_positions=None, noisy_targets=None) -> None:
    output_folder = Path(config["output_folder"])
    csv_path = output_folder / config.get("csv_filename", "dataset_navigation.csv")
    if config.get("normalize_targets", False):
        norms = np.linalg.norm(targets, axis=1, keepdims=True)
        targets = targets / np.maximum(norms, 1e-8)
        if noisy_targets is not None:
            noisy_norms = np.linalg.norm(noisy_targets, axis=1, keepdims=True)
            noisy_targets = noisy_targets / np.maximum(noisy_norms, 1e-8)
    target_scale = float(config.get("target_scale", 0.0))
    if target_scale > 0.0:
        targets = targets / target_scale
        if noisy_targets is not None:
            noisy_targets = noisy_targets / target_scale
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if noisy_positions is not None:
            writer.writerow(["Filename", "Position", "Rotation", "Target", "Noisy Position", "Noisy Rotation", "Noisy Target"])
        else:
            writer.writerow(["Filename", "Position", "Rotation", "Target"])
        for idx, (position, rotation, target) in enumerate(zip(positions, rotations, targets)):
            filename = f"rgb_{idx:04d}.png"
            if noisy_positions is not None:
                writer.writerow([filename, position, rotation, target, noisy_positions[idx], rotation, noisy_targets[idx]])
            else:
                writer.writerow([filename, position, rotation, target])


def _run_blender_job(job: dict, blender_path: str | None = None, log_path: str | Path | None = None) -> None:
    blender = find_blender(blender_path)
    worker = ROOT / "blender_render_worker.py"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as temp:
        json.dump(job, temp)
        temp_path = temp.name
    command = [str(blender), "-b", "--python", str(worker), "--", temp_path]
    try:
        result = subprocess.run(command, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if log_path:
            Path(log_path).write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"Blender render failed with exit code {result.returncode}. See {log_path or 'captured output'}.")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def build_scene_job(config: dict, frames: list[dict]) -> dict:
    map_path = config.get("map", "")
    landmarks = config.get("landmarks")
    if landmarks is None:
        landmarks_array = load_landmark_locations(get_locations_path(map_path))
        if landmarks_array is None:
            landmarks_array = default_landmarks()
        landmarks = landmarks_array.tolist()

    area = config.get("area", [-4.0, 4.0, -4.0, 4.0])
    image_size = int(config.get("image_size", 160))
    return {
        "width": int(config.get("image_width", image_size)),
        "height": int(config.get("image_height", image_size)),
        "samples": int(config.get("samples", 12)),
        "area": area,
        "landmarks": landmarks,
        "scene_style": config.get("scene_style", "simple"),
        "kenney_asset_dir": config.get("kenney_asset_dir"),
        "kenney_seed": int(config.get("kenney_seed", 42)),
        "kenney_tree_positions": config.get("kenney_tree_positions", []),
        "kenney_detail_positions": config.get("kenney_detail_positions", []),
        "blocks": config.get("blocks", []),
        "landmark_radius": float(config.get("landmark_radius", 0.24)),
        "landmark_height": float(config.get("landmark_height", 1.7)),
        "landmark_style": config.get("landmark_style", "cylinder"),
        "fov_degrees": float(config.get("fov_degrees", 105.0)),
        "lens_mm": float(config.get("lens_mm", 12.0)),
        "frames": frames,
    }


def generate_dataset(config: dict) -> None:
    output_folder = Path(config["output_folder"])
    image_folder = output_folder / "Replicator" / "rgb"
    if config.get("clean", True) and output_folder.exists():
        shutil.rmtree(output_folder)
    image_folder.mkdir(parents=True, exist_ok=True)

    positions, rotations, noisy_path = generate_positions_and_rotations(config)
    render_positions = noisy_path if noisy_path is not None else positions
    render_positions = [tuple(float(v) for v in pos) for pos in render_positions]
    rotations = [tuple(float(v) for v in rot) for rot in rotations]

    frames = [
        {
            "position": render_positions[idx],
            "yaw_degrees": rotations[idx][2],
            "output": str((image_folder / f"rgb_{idx:04d}.png").resolve()),
            "pitch_degrees": config.get("pitch_degrees", -4.0),
        }
        for idx in range(len(render_positions))
    ]
    job = build_scene_job(config, frames)
    _run_blender_job(job, config.get("blender_path"), output_folder / "blender_render.log")

    targets = calculate_target_vectors(positions, rotations, np.asarray(config["home_position"], dtype=float))
    if noisy_path is not None:
        noisy_targets = calculate_target_vectors(noisy_path, rotations, np.asarray(config["home_position"], dtype=float))
        _write_csv(config, positions, rotations, targets, noisy_path, noisy_targets)
    else:
        _write_csv(config, positions, rotations, targets)


def render_single_image(config: dict, position, yaw_degrees: float, output_path: str | Path) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = {
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "yaw_degrees": float(yaw_degrees),
        "output": str(output_path),
        "pitch_degrees": config.get("pitch_degrees", -4.0),
    }
    job = build_scene_job(config, [frame])
    _run_blender_job(job, config.get("blender_path"), output_path.with_suffix(".log"))
    return output_path
