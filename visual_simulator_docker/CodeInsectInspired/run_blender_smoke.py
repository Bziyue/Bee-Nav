"""Run a lightweight Bee-Nav end-to-end smoke test with Blender.

The flow mirrors the Isaac Sim version, but Blender is only used as an RGB
camera renderer:

1. render a tiny visual memory dataset;
2. train the visual homing CNN briefly;
3. run a few closed-loop homing steps;
4. write a summary and trajectory plot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import blender_backend
import train as train_module
from insect_utils.flight_path_functions import calculate_absolute_angular_error, calculate_distance, normalize_vectors, rotate_vector_by_yaw
from simple_network import SimpleCNN


ROOT = Path(__file__).resolve().parent


def _as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def _load_model(model_path: str) -> SimpleCNN:
    model = SimpleCNN()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _image_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return TF.to_tensor(image).unsqueeze(0)


def _plot_trajectories(trajectories, home, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    for idx, trajectory in enumerate(trajectories):
        xy = np.asarray([[p[0], p[1]] for p in trajectory], dtype=float)
        if len(xy):
            plt.plot(xy[:, 0], xy[:, 1], marker="o", linewidth=1.2, markersize=3, label=f"run {idx + 1}")
    plt.scatter([home[0]], [home[1]], color="red", marker="x", s=80, label="home")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Blender Bee-Nav homing smoke test")
    plt.legend(loc="best", fontsize=8)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def _make_kenney_scene(seed: int = 42):
    rng = random.Random(seed)
    tree_positions = []
    detail_positions = []
    while len(tree_positions) < 42:
        x = rng.uniform(-14.0, 14.0)
        y = rng.uniform(-14.0, 14.0)
        if abs(x + 2.0) < 1.8 or (abs(x) < 2.2 and abs(y) < 2.2):
            continue
        tree_positions.append([x, y, rng.uniform(0.65, 1.2), rng.uniform(0, 360)])

    for idx in range(34):
        x = rng.uniform(-13.0, 13.0)
        y = rng.uniform(-13.0, 13.0)
        if abs(x + 2.0) < 1.6:
            continue
        kind = "rock" if idx % 4 == 0 else ("flower" if idx % 5 == 0 else "bush")
        detail_positions.append([x, y, rng.uniform(0.45, 0.85), rng.uniform(0, 360), kind])

    landmarks = [[x, y, 0.0] for x, y, *_ in tree_positions]
    return landmarks, tree_positions, detail_positions


def _run_homing(config: dict, model_path: str, output_root: Path):
    model = _load_model(model_path)
    home = np.asarray(config["home_position"], dtype=float)
    starts = config.get(
        "homing_starts",
        [
            [1.9, 0.0, home[2]],
            [0.0, 1.9, home[2]],
        ],
    )
    yaw_values = [0.0] * len(starts) if config.get("only_point_north", False) else [180.0, -90.0] * len(starts)
    threshold = float(config.get("homing_threshold", 0.75))
    max_steps = int(config.get("homing_max_steps", 8))
    d_lim = float(config.get("avoidance_distance", 1.0))
    max_step = float(config.get("homing_max_step", 0.65))
    landmarks = np.asarray(config["landmarks"], dtype=float)
    trajectories = []
    successes = []
    angular_errors = []

    for run_index, start in enumerate(starts):
        current_position = np.asarray(start, dtype=float)
        yaw = yaw_values[run_index]
        trajectory = [current_position.tolist()]
        success = False

        for step in range(max_steps):
            if calculate_distance(current_position, home) <= threshold:
                success = True
                break

            image_path = output_root / "homing_frames" / f"run_{run_index:02d}_step_{step:02d}.png"
            blender_backend.render_single_image(config, current_position, yaw, image_path)
            image_tensor = _image_tensor(image_path)

            with torch.no_grad():
                prediction = model(image_tensor).squeeze().cpu().numpy()

            if config.get("homing_use_scaled_vector", False):
                target_scale = float(config.get("target_scale", 1.0))
                move = np.asarray(prediction, dtype=float) * target_scale
                if config.get("homing_invert_prediction", False):
                    move = -move
            else:
                norm_prediction = normalize_vectors(np.asarray([prediction]))[0]
                step_size = float(config.get("homing_fixed_step", 0.0))
                if step_size <= 0.0:
                    step_size = 0.2 + 0.35 * min(np.linalg.norm(prediction) / 3.0, 1.0)
                if config.get("only_point_north", False):
                    yaw = 0.0
                    sign = -1.0 if config.get("homing_invert_prediction", False) else 1.0
                    move = sign * step_size * norm_prediction
                else:
                    move_body = step_size * norm_prediction
                    desired_world = -np.asarray(rotate_vector_by_yaw(move_body, yaw))
                    yaw = math.degrees(math.atan2(desired_world[1], desired_world[0]))
                    move = step_size * np.asarray([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])

            for landmark in landmarks:
                dist = calculate_distance(current_position, landmark)
                if 1e-6 < dist < d_lim:
                    direction_away = current_position[:2] - landmark[:2]
                    direction_away = direction_away / np.linalg.norm(direction_away)
                    repulsive = 0.2 * (1.0 / dist - 1.0 / d_lim) / (dist**2) * direction_away
                    move += repulsive

            move_norm = np.linalg.norm(move)
            if move_norm > max_step:
                move = move / move_norm * max_step

            target_vector = home[:2] - current_position[:2]
            angular_errors.append(calculate_absolute_angular_error(target_vector, move))
            current_position = np.asarray([current_position[0] + move[0], current_position[1] + move[1], current_position[2]])
            trajectory.append(current_position.tolist())

        if calculate_distance(current_position, home) <= threshold:
            success = True
        trajectories.append(trajectory)
        successes.append(bool(success))

    return trajectories, successes, angular_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="windows_blender_outputs", help="Output directory.")
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--n-positions", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--forest15", action="store_true", help="Use the bundled 15-tree forest map locations.")
    parser.add_argument("--kenney-forest", action="store_true", help="Use downloaded Kenney Nature Kit models.")
    parser.add_argument("--flight-radius", type=float, default=None, help="Learning-flight circle radius.")
    parser.add_argument("--homing-max-steps", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    output_root = (ROOT / args.output_dir).resolve()
    dataset_dir = output_root / "dataset"
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.kenney_forest:
        asset_dir = ROOT / "kenney_forest_assets" / "nature-kit" / "Models" / "DAE format"
        if not asset_dir.exists():
            raise FileNotFoundError(f"Kenney assets not found: {asset_dir}")
        landmarks, tree_positions, detail_positions = _make_kenney_scene()
        area = [-16.0, 16.0, -16.0, 16.0]
        flight_radius = args.flight_radius or 7.5
        landmark_radius = 0.45
        landmark_height = 2.2
        landmark_style = "kenney"
        homing_starts = [[6.5, 0.0, 1.2], [0.0, 6.5, 1.2], [-5.2, -4.4, 1.2]]
        homing_max_steps = args.homing_max_steps or 24
        output_map = "kenney_nature_kit_forest"
        scene_style = "kenney_forest"
        kenney_asset_dir = _as_posix(asset_dir)
    elif args.forest15:
        map_path = ROOT / "maps" / "forest_15_trees_50x50_area_0.usd"
        locations_path = ROOT / "maps" / "forest_15_trees_50x50_area_locations_0.csv"
        landmarks_array = blender_backend.load_landmark_locations(str(locations_path))
        if landmarks_array is None:
            raise FileNotFoundError(f"Could not load forest locations: {locations_path}")
        landmarks = landmarks_array.tolist()
        area = [-50.0, 50.0, -50.0, 50.0]
        flight_radius = args.flight_radius or 12.0
        landmark_radius = 0.32
        landmark_height = 7.0
        landmark_style = "tree"
        homing_starts = [[10.5, 0.0, 1.2], [0.0, 10.5, 1.2], [-8.5, -7.0, 1.2]]
        homing_max_steps = args.homing_max_steps or 28
        output_map = _as_posix(map_path)
        scene_style = "simple"
        kenney_asset_dir = None
        tree_positions = []
        detail_positions = []
    else:
        landmarks = [
            [1.2, 0.7, 0.0],
            [-1.4, -0.8, 0.0],
            [-0.6, 1.5, 0.0],
            [1.8, -1.4, 0.0],
        ]
        area = [-3.2, 3.2, -3.2, 3.2]
        flight_radius = args.flight_radius or 2.2
        landmark_radius = 0.22
        landmark_height = 1.5
        landmark_style = "cylinder"
        homing_starts = [[1.9, 0.0, 1.2], [0.0, 1.9, 1.2]]
        homing_max_steps = args.homing_max_steps or 8
        output_map = "procedural_blender_smoke"
        scene_style = "simple"
        kenney_asset_dir = None
        tree_positions = []
        detail_positions = []

    render_config = {
        "backend": "blender",
        "map": output_map,
        "scene_style": scene_style,
        "kenney_asset_dir": kenney_asset_dir,
        "kenney_seed": 42,
        "kenney_tree_positions": tree_positions,
        "kenney_detail_positions": detail_positions,
        "output_folder": _as_posix(dataset_dir),
        "csv_filename": "dataset_navigation.csv",
        "home_position": [0.0, 0.0, 1.2],
        "shape_flight": "home_circle",
        "shape_params": [flight_radius],
        "n_positions": args.n_positions,
        "only_point_north": True,
        "use_noisy_spiral": False,
        "image_size": args.image_size,
        "image_width": args.image_size,
        "image_height": args.image_size,
        "area": area,
        "landmarks": landmarks,
        "landmark_radius": landmark_radius,
        "landmark_height": landmark_height,
        "landmark_style": landmark_style,
        "fov_degrees": 105,
        "samples": 12,
        "homing_starts": homing_starts,
        "homing_threshold": 1.0 if (args.forest15 or args.kenney_forest) else 0.75,
        "homing_max_steps": homing_max_steps,
        "homing_max_step": 0.75 if (args.forest15 or args.kenney_forest) else 0.65,
        "homing_fixed_step": 0.62 if args.kenney_forest else (0.65 if args.forest15 else 0.0),
        "homing_invert_prediction": False,
        "homing_use_scaled_vector": bool(args.kenney_forest),
        "avoidance_distance": 1.2 if args.kenney_forest else (2.2 if args.forest15 else 1.0),
        "normalize_targets": bool(args.forest15),
        "target_scale": flight_radius if args.kenney_forest else 0.0,
    }

    print("[1/4] Rendering Blender learning dataset...")
    blender_backend.generate_dataset(render_config)

    print("[2/4] Training visual homing network...")
    training_config = train_module.load_config("config/config_training.json")
    training_config.update(
        {
            "model_suffix": "_blender_kenney" if args.kenney_forest else ("_blender_forest15" if args.forest15 else "_blender_smoke"),
            "epochs": args.epochs,
            "batch_size": 4,
            "batch_size_val": 2,
            "batch_size_eval": 2,
            "learning_rate": 5e-4,
            "dataset_folder": _as_posix(dataset_dir),
            "image_folder": _as_posix(dataset_dir / "Replicator" / "rgb") + "/",
            "csv_filename": "dataset_navigation.csv",
            "split_ratio": 0.8,
            "augment": False,
            "evaluation_dataset_folder": _as_posix(dataset_dir),
            "evaluation_image_folder": _as_posix(dataset_dir / "Replicator" / "rgb") + "/",
            "evaluation_csv_filename": "dataset_navigation.csv",
            "model_type": "simple",
            "num_workers": 0,
        }
    )
    model_path = train_module.train_model(training_config, only_training=True)

    print("[3/4] Running Blender closed-loop homing...")
    trajectories, successes, angular_errors = _run_homing(render_config, model_path, output_root)

    print("[4/4] Saving summary...")
    plot_path = output_root / "homing_trajectories.png"
    _plot_trajectories(trajectories, render_config["home_position"], plot_path)

    summary = {
        "backend": "blender",
        "map": render_config["map"],
        "landmark_count": len(landmarks),
        "dataset": str(dataset_dir),
        "model": str(Path(model_path).resolve()),
        "successes": successes,
        "success_count": int(sum(successes)),
        "num_runs": len(successes),
        "mean_angular_error": float(np.mean(angular_errors)) if angular_errors else None,
        "trajectories": trajectories,
        "trajectory_plot": str(plot_path),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
