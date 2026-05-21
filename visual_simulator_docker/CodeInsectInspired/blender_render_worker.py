"""Blender-side renderer for the lightweight Bee-Nav backend.

This file is executed by Blender, not by the normal Python interpreter:

    blender -b --python blender_render_worker.py -- job.json
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _args_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _make_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.78
    return material


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _terrain_height(x: float, y: float) -> float:
    return 0.16 * math.sin(x * 0.34) * math.cos(y * 0.26) - 0.12 * math.exp(-(x + 2.0) ** 2 / 5.5)


def _make_kenney_terrain(area, ground_mat, water_mat) -> None:
    size = max(float(area[1]) - float(area[0]), float(area[3]) - float(area[2]))
    resolution = 48
    x_min, x_max, y_min, y_max = [float(v) for v in area]
    verts = []
    faces = []
    for yi in range(resolution + 1):
        y = y_min + (y_max - y_min) * yi / resolution
        for xi in range(resolution + 1):
            x = x_min + (x_max - x_min) * xi / resolution
            verts.append((x, y, _terrain_height(x, y)))
    for yi in range(resolution):
        for xi in range(resolution):
            a = yi * (resolution + 1) + xi
            faces.append((a, a + 1, a + resolution + 2, a + resolution + 1))
    mesh = bpy.data.meshes.new("BeeNavKenneyTerrainMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    terrain = bpy.data.objects.new("BeeNavKenneyTerrain", mesh)
    bpy.context.collection.objects.link(terrain)
    terrain.data.materials.append(ground_mat)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-2.0, 0.0, -0.08))
    river = bpy.context.object
    river.name = "BeeNavKenneyRiver"
    river.dimensions = (2.1, size + 4.0, 0.035)
    river.rotation_euler[2] = math.radians(-7.0)
    river.data.materials.append(water_mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _set_camera_view(camera, position, yaw_degrees: float, pitch_degrees: float = -4.0) -> None:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    direction = Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
    camera.location = (float(position[0]), float(position[1]), float(position[2]))
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_cube(name: str, location, scale, material) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)


def _add_cylinder(name: str, location, radius: float, depth: float, material) -> None:
    bpy.ops.mesh.primitive_cylinder_add(vertices=18, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)


def _add_cone(name: str, location, radius1: float, depth: float, material) -> None:
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=radius1, radius2=0.05, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)


def _add_tree(name: str, x: float, y: float, z: float, height: float, radius: float, trunk_mat, foliage_mat) -> None:
    trunk_height = height * 0.45
    crown_height = height * 0.75
    _add_cylinder(f"{name}_trunk", (x, y, z + trunk_height / 2.0), radius, trunk_height, trunk_mat)
    _add_cone(
        f"{name}_crown",
        (x, y, z + trunk_height + crown_height / 2.0 - 0.15),
        radius * 5.0,
        crown_height,
        foliage_mat,
    )


def _import_template(asset_dir: str, filename: str) -> list:
    path = Path(asset_dir) / filename
    before = set(bpy.data.objects)
    bpy.ops.wm.collada_import(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    for obj in imported:
        obj.hide_render = True
        obj.hide_viewport = True
    return imported


def _place_template(template: list, name: str, location, scale: float, yaw: float) -> None:
    angle = math.radians(yaw)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for obj in template:
        new = obj.copy()
        new.data = obj.data
        new.hide_render = False
        new.hide_viewport = False
        new.name = name
        rel = Vector(obj.location)
        new.location = (
            float(location[0]) + scale * (rel.x * cos_a - rel.y * sin_a),
            float(location[1]) + scale * (rel.x * sin_a + rel.y * cos_a),
            float(location[2]) + scale * rel.z,
        )
        new.rotation_euler = obj.rotation_euler.copy()
        new.rotation_euler[2] += angle
        new.scale = (scale, scale, scale)
        bpy.context.collection.objects.link(new)


def _populate_kenney_forest(job: dict) -> None:
    asset_dir = job.get("kenney_asset_dir")
    if not asset_dir:
        raise RuntimeError("kenney_forest scene requires kenney_asset_dir")

    templates = {
        "pine_a": _import_template(asset_dir, "tree_pineTallA.dae"),
        "pine_b": _import_template(asset_dir, "tree_pineRoundB.dae"),
        "oak": _import_template(asset_dir, "tree_oak.dae"),
        "tree": _import_template(asset_dir, "tree_default.dae"),
        "bush": _import_template(asset_dir, "plant_bushLarge.dae"),
        "rock": _import_template(asset_dir, "rock_largeB.dae"),
        "flower": _import_template(asset_dir, "flower_yellowB.dae"),
    }
    tree_keys = ["pine_a", "pine_b", "oak", "tree"]

    rng = random.Random(int(job.get("kenney_seed", 42)))
    for idx, position in enumerate(job.get("kenney_tree_positions", [])):
        x, y = float(position[0]), float(position[1])
        z = _terrain_height(x, y)
        _place_template(
            templates[tree_keys[idx % len(tree_keys)]],
            f"KenneyTree_{idx:03d}",
            (x, y, z),
            float(position[2]) if len(position) > 2 else rng.uniform(0.7, 1.25),
            float(position[3]) if len(position) > 3 else rng.uniform(0, 360),
        )

    for idx, position in enumerate(job.get("kenney_detail_positions", [])):
        x, y = float(position[0]), float(position[1])
        z = _terrain_height(x, y) + 0.02
        kind = position[4] if len(position) > 4 else ("bush" if idx % 3 else "rock")
        _place_template(
            templates.get(kind, templates["bush"]),
            f"KenneyDetail_{idx:03d}",
            (x, y, z),
            float(position[2]) if len(position) > 2 else 0.7,
            float(position[3]) if len(position) > 3 else rng.uniform(0, 360),
        )


def _build_scene(job: dict):
    _clear_scene()

    scene = bpy.context.scene
    scene.render.engine = job.get("render_engine", "BLENDER_EEVEE_NEXT")
    scene.render.resolution_x = int(job.get("width", 256))
    scene.render.resolution_y = int(job.get("height", 128))
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = bpy.data.worlds.new("BeeNavWorld")
    scene.world.color = (0.55, 0.70, 0.95)

    try:
        scene.eevee.taa_render_samples = int(job.get("samples", 16))
    except Exception:
        pass

    ground_mat = _make_material("ground_green", (0.30, 0.43, 0.26, 1.0))
    red_mat = _make_material("red_landmark", (0.85, 0.12, 0.08, 1.0))
    blue_mat = _make_material("blue_landmark", (0.08, 0.20, 0.85, 1.0))
    yellow_mat = _make_material("yellow_landmark", (0.95, 0.70, 0.08, 1.0))
    trunk_mat = _make_material("trunk", (0.35, 0.20, 0.10, 1.0))
    foliage_mats = [
        _make_material("foliage_dark", (0.07, 0.26, 0.09, 1.0)),
        _make_material("foliage_mid", (0.12, 0.38, 0.12, 1.0)),
        _make_material("foliage_warm", (0.20, 0.34, 0.10, 1.0)),
    ]

    area = job.get("area", [-5.0, 5.0, -5.0, 5.0])
    materials = [red_mat, blue_mat, yellow_mat, trunk_mat]
    if job.get("scene_style") == "kenney_forest":
        water_mat = _make_material("water_blue", (0.08, 0.30, 0.42, 0.72))
        _make_kenney_terrain(area, ground_mat, water_mat)
        _populate_kenney_forest(job)
    else:
        width = max(4.0, float(area[1]) - float(area[0]) + 4.0)
        depth = max(4.0, float(area[3]) - float(area[2]) + 4.0)
        _add_cube("Ground", (0.0, 0.0, -0.03), (width, depth, 0.06), ground_mat)

        landmark_style = job.get("landmark_style", "cylinder")
        for idx, landmark in enumerate(job.get("landmarks", [])):
            x, y, z = float(landmark[0]), float(landmark[1]), float(landmark[2])
            height = float(job.get("landmark_height", 1.6 + 0.2 * (idx % 3)))
            radius = float(job.get("landmark_radius", 0.22 + 0.04 * (idx % 2)))
            if landmark_style == "tree":
                _add_tree(f"Tree_{idx:02d}", x, y, z, height, radius, trunk_mat, foliage_mats[idx % len(foliage_mats)])
            else:
                _add_cylinder(f"Landmark_{idx:02d}", (x, y, z + height / 2.0), radius, height, materials[idx % len(materials)])

    for idx, block in enumerate(job.get("blocks", [])):
        material = materials[idx % len(materials)]
        _add_cube(f"Block_{idx:02d}", block["location"], block["scale"], material)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 5.0), rotation=(math.radians(45), 0, math.radians(35)))
    bpy.context.object.name = "Sun"
    bpy.context.object.data.energy = 2.2

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 5.0))
    bpy.context.object.name = "Softbox"
    bpy.context.object.data.energy = 350
    bpy.context.object.data.size = 6

    bpy.ops.object.camera_add(location=(0.0, -2.0, 1.5))
    camera = bpy.context.object
    camera.name = "BeeNavCamera"
    camera.data.lens = float(job.get("lens_mm", 12.0))
    camera.data.angle = math.radians(float(job.get("fov_degrees", 105.0)))
    scene.camera = camera
    return camera


def main() -> None:
    args = _args_after_double_dash()
    if not args:
        raise SystemExit("Missing job JSON path after --")
    job_path = Path(args[0])
    job = json.loads(job_path.read_text(encoding="utf-8"))

    camera = _build_scene(job)

    for frame in job["frames"]:
        output = Path(frame["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        _set_camera_view(camera, frame["position"], float(frame["yaw_degrees"]), float(frame.get("pitch_degrees", -4.0)))
        bpy.context.scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        print(f"[blender-worker] rendered {output}")


if __name__ == "__main__":
    main()
