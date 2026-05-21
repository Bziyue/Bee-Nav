"""Render a small forest fly-through using Kenney Nature Kit assets.

Run with Blender:
    blender -b --python render_kenney_forest_video.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "kenney_forest_assets" / "nature-kit" / "Models" / "DAE format"
OUTPUT_DIR = ROOT / "kenney_forest_video"
VIDEO_PATH = OUTPUT_DIR / "kenney_forest_flythrough.mp4"
STILL_PATH = OUTPUT_DIR / "preview_frame.png"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.78
    return mat


def make_terrain(size: float = 36.0, resolution: int = 80) -> None:
    grass_mat = make_material("soft_moss_grass", (0.18, 0.42, 0.20, 1.0))
    verts = []
    faces = []
    half = size / 2.0
    for y in range(resolution + 1):
        fy = -half + size * y / resolution
        for x in range(resolution + 1):
            fx = -half + size * x / resolution
            river = math.exp(-(fx + 2.0) ** 2 / 5.5)
            height = 0.18 * math.sin(fx * 0.35) * math.cos(fy * 0.27) - 0.18 * river
            verts.append((fx, fy, height))
    for y in range(resolution):
        for x in range(resolution):
            a = y * (resolution + 1) + x
            faces.append((a, a + 1, a + resolution + 2, a + resolution + 1))
    mesh = bpy.data.meshes.new("TerrainMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    terrain = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(terrain)
    terrain.data.materials.append(grass_mat)

    water_mat = make_material("quiet_river", (0.08, 0.30, 0.42, 0.72))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-2.0, 0.0, -0.08))
    river = bpy.context.object
    river.name = "NarrowRiver"
    river.dimensions = (2.6, size + 2.0, 0.035)
    river.rotation_euler[2] = math.radians(-7.0)
    river.data.materials.append(water_mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def import_template(filename: str) -> list[bpy.types.Object]:
    path = ASSET_DIR / filename
    before = set(bpy.data.objects)
    bpy.ops.wm.collada_import(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    for obj in imported:
        obj.name = f"template_{filename}_{obj.name}"
        obj.hide_render = True
        obj.hide_viewport = True
    return imported


def place_template(
    template: list[bpy.types.Object],
    name: str,
    location: tuple[float, float, float],
    scale: float,
    yaw: float,
) -> None:
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
            location[0] + scale * (rel.x * cos_a - rel.y * sin_a),
            location[1] + scale * (rel.x * sin_a + rel.y * cos_a),
            location[2] + scale * rel.z,
        )
        new.rotation_euler = obj.rotation_euler.copy()
        new.rotation_euler[2] += angle
        new.scale = (scale, scale, scale)
        bpy.context.collection.objects.link(new)


def populate_forest() -> None:
    random.seed(42)
    templates = {
        "pine_a": import_template("tree_pineTallA.dae"),
        "pine_b": import_template("tree_pineRoundB.dae"),
        "oak": import_template("tree_oak.dae"),
        "default": import_template("tree_default.dae"),
        "bush": import_template("plant_bushLarge.dae"),
        "rock": import_template("rock_largeB.dae"),
        "flower": import_template("flower_yellowB.dae"),
    }

    tree_files = ["pine_a", "pine_b", "oak", "default"]
    for idx in range(95):
        x = random.uniform(-16.0, 16.0)
        y = random.uniform(-16.0, 16.0)
        if abs(x + 2.0) < 1.7 or (abs(x) < 2.8 and y < -5.0):
            continue
        scale = random.uniform(0.65, 1.35)
        place_template(templates[random.choice(tree_files)], f"Tree_{idx:03d}", (x, y, 0.0), scale, random.uniform(0, 360))

    for idx in range(65):
        x = random.uniform(-15.0, 15.0)
        y = random.uniform(-15.0, 15.0)
        if abs(x + 2.0) < 1.5:
            continue
        template = templates["bush"] if idx % 3 else templates["rock"]
        place_template(template, f"Detail_{idx:03d}", (x, y, 0.02), random.uniform(0.45, 0.9), random.uniform(0, 360))

    for idx in range(35):
        place_template(
            templates["flower"],
            f"Flower_{idx:03d}",
            (random.uniform(-10.0, 13.0), random.uniform(-12.0, 8.0), 0.03),
            random.uniform(0.5, 0.9),
            random.uniform(0, 360),
        )


def point_camera_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera_and_lighting() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 32
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 24
    scene.frame_start = 1
    scene.frame_end = 96
    scene.world = bpy.data.worlds.new("ForestWorld")
    scene.world.color = (0.50, 0.68, 0.92)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10), rotation=(math.radians(45), 0, math.radians(-35)))
    sun = bpy.context.object
    sun.name = "WarmSun"
    sun.data.energy = 2.4

    bpy.ops.object.light_add(type="AREA", location=(0, -6, 8))
    fill = bpy.context.object
    fill.name = "ForestFill"
    fill.data.energy = 120
    fill.data.size = 10

    bpy.ops.object.camera_add(location=(-11.0, -12.0, 3.2))
    camera = bpy.context.object
    camera.name = "FlythroughCamera"
    camera.data.lens = 24
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 12.0
    camera.data.dof.aperture_fstop = 5.6
    scene.camera = camera

    path = [
        (-11.0, -12.0, 3.2, (-2.0, -2.0, 1.7)),
        (-7.0, -4.0, 2.4, (2.0, 2.0, 1.6)),
        (-2.5, 2.5, 2.1, (4.5, 6.0, 1.4)),
        (4.0, 8.5, 2.7, (-2.0, 12.0, 1.6)),
    ]
    for frame, (x, y, z, target) in zip((1, 32, 64, 96), path):
        scene.frame_set(frame)
        camera.location = (x, y, z)
        point_camera_at(camera, target)
        camera.keyframe_insert(data_path="location")
        camera.keyframe_insert(data_path="rotation_euler")

    for fcurve in camera.animation_data.action.fcurves:
        for key in fcurve.keyframe_points:
            key.interpolation = "BEZIER"


def render_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(42)
    scene.render.filepath = str(STILL_PATH)
    bpy.ops.render.render(write_still=True)

    scene.render.filepath = str(VIDEO_PATH)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    bpy.ops.render.render(animation=True)


def main() -> None:
    clear_scene()
    make_terrain()
    populate_forest()
    setup_camera_and_lighting()
    render_outputs()
    print(f"Rendered still: {STILL_PATH}")
    print(f"Rendered video: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
