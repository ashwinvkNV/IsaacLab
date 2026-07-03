# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a minimal NVlabs/GraspGenX gripper descriptor from an Isaac USD.

This is an offline helper. It reads a gripper USD, exports merged visual and
collision meshes, writes a GraspGenX ``config.json``, and copies the source USD
into ``assets/x_grippers/<name>/``. GraspGenX release checkpoints that use the
``sweep_volume_v2`` gripper backbone can load this descriptor without the
optional pointcloud, TSDF, or VAE files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


@dataclass
class MeshRecord:
    """A USD mesh loaded into the selected gripper root frame."""

    path: str
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gripper_usd", type=Path, required=True, help="Isaac USD containing the gripper geometry.")
    parser.add_argument("--name", type=str, required=True, help="Name of the GraspGenX gripper descriptor to create.")
    parser.add_argument(
        "--graspgenx_root",
        type=Path,
        default=None,
        help="NVlabs/GraspGenX checkout. Used to default --output_dir to <root>/assets/x_grippers.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory that contains GraspGenX x_grippers. Defaults to <graspgenx_root>/assets/x_grippers.",
    )
    parser.add_argument(
        "--root_prim",
        type=str,
        default=None,
        help="USD prim to use as the gripper root frame. Defaults to the stage default prim.",
    )
    parser.add_argument(
        "--include_mesh_regex",
        type=str,
        default=None,
        help="Optional regex. Only mesh prim paths matching this regex are considered.",
    )
    parser.add_argument(
        "--exclude_mesh_regex",
        type=str,
        default=None,
        help="Optional regex. Mesh prim paths matching this regex are ignored.",
    )
    parser.add_argument("--visual_regex", type=str, default="visual", help="Regex for visual mesh prim paths.")
    parser.add_argument("--collision_regex", type=str, default="collision", help="Regex for collision mesh prim paths.")
    parser.add_argument(
        "--finger_regex",
        type=str,
        default="finger_tip|finger_pad|finger",
        help="Regex for meshes used to estimate the inner grasp/sweep volume.",
    )
    parser.add_argument(
        "--closing_axis",
        type=str,
        default="auto",
        help="Source USD axis that separates the fingers: auto, x, y, z, -x, -y, or -z.",
    )
    parser.add_argument(
        "--approach_axis",
        type=str,
        default="auto",
        help="Source USD axis that points along the gripper approach/depth: auto, x, y, z, -x, -y, or -z.",
    )
    parser.add_argument("--unit_scale", type=float, default=None, help="Scale USD coordinates to meters.")
    parser.add_argument(
        "--open_joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override or add an open joint value in the output config. Can be repeated.",
    )
    parser.add_argument(
        "--close_joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override or add a close joint value in the output config. Can be repeated.",
    )
    parser.add_argument(
        "--sweep_volume_extents",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override open sweep-volume extents in the canonical GraspGenX frame [m].",
    )
    parser.add_argument(
        "--sweep_volume_offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override open sweep-volume center in the canonical GraspGenX frame [m].",
    )
    parser.add_argument(
        "--sweep_volume_mid_extents",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override half-open sweep-volume extents in the canonical GraspGenX frame [m].",
    )
    parser.add_argument(
        "--sweep_volume_mid_offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override half-open sweep-volume center in the canonical GraspGenX frame [m].",
    )
    parser.add_argument(
        "--fingertip",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override fingertip/TCP point in the canonical GraspGenX frame [m].",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override GraspGenX standoff values [m].",
    )
    parser.add_argument(
        "--gripper_type",
        type=str,
        default="revolute_2f",
        choices=("parallel_2f", "revolute_2f", "revolute_3f"),
        help="GraspGenX gripper type written to config.json.",
    )
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to mark the descriptor as symmetric.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing descriptor directory.")
    args = parser.parse_args()

    if args.output_dir is None and args.graspgenx_root is None:
        parser.error("Provide --output_dir or --graspgenx_root.")
    if (args.sweep_volume_extents is None) != (args.sweep_volume_offset is None):
        parser.error("--sweep_volume_extents and --sweep_volume_offset must be provided together.")
    if (args.sweep_volume_mid_extents is None) != (args.sweep_volume_mid_offset is None):
        parser.error("--sweep_volume_mid_extents and --sweep_volume_mid_offset must be provided together.")
    return args


def _import_usd_and_mesh_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import trimesh

        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise ImportError(
            "Run this script with a Python environment that includes USD Python bindings and trimesh, "
            "for example `/path/to/GraspGenX/.venv/bin/python scripts/tools/create_graspgenx_gripper_from_usd.py ...`."
        ) from exc

    return Gf, Usd, UsdGeom, trimesh


def _parse_joint_overrides(entries: list[str]) -> dict[str, float]:
    overrides = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Joint override must be NAME=VALUE, got: {entry}")
        name, value = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Joint override has an empty name: {entry}")
        overrides[name] = float(value)
    return overrides


def _parse_axis(value: str) -> np.ndarray | None:
    value = value.strip().lower()
    if value == "auto":
        return None
    sign = -1.0 if value.startswith("-") else 1.0
    axis = value[1:] if value.startswith("-") else value
    if axis not in AXIS_VECTORS:
        raise ValueError(f"Unsupported axis '{value}'. Expected auto, x, y, z, -x, -y, or -z.")
    return sign * AXIS_VECTORS[axis]


def _axis_name(axis: np.ndarray) -> str:
    index = int(np.argmax(np.abs(axis)))
    prefix = "-" if axis[index] < 0.0 else ""
    return f"{prefix}{('x', 'y', 'z')[index]}"


def _path_matches(path: str, pattern: str | None) -> bool:
    return pattern is None or re.search(pattern, path) is not None


def _path_excluded(path: str, pattern: str | None) -> bool:
    return pattern is not None and re.search(pattern, path) is not None


def _is_under_root(path: str, root_path: str) -> bool:
    return path == root_path or path.startswith(f"{root_path}/")


def _triangulate(face_counts: Any, face_indices: Any) -> np.ndarray:
    faces = []
    cursor = 0
    for count in face_counts:
        count = int(count)
        indices = [int(index) for index in face_indices[cursor : cursor + count]]
        cursor += count
        if count < 3:
            continue
        for index in range(1, count - 1):
            faces.append([indices[0], indices[index], indices[index + 1]])
    return np.asarray(faces, dtype=np.int64)


def _load_mesh_records(args: argparse.Namespace, stage: Any, root_prim: Any, UsdGeom: Any) -> list[MeshRecord]:
    cache = UsdGeom.XformCache()
    root_path = str(root_prim.GetPath())
    root_inverse = cache.GetLocalToWorldTransform(root_prim).GetInverse()
    unit_scale = args.unit_scale
    if unit_scale is None:
        unit_scale = float(UsdGeom.GetStageMetersPerUnit(stage))

    records = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetTypeName() != "Mesh" or not _is_under_root(path, root_path):
            continue
        if not _path_matches(path, args.include_mesh_regex) or _path_excluded(path, args.exclude_mesh_regex):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_counts or not face_indices:
            continue

        vertices = np.asarray(points, dtype=np.float64)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) * root_inverse, dtype=np.float64)
        vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
        vertices = (vertices_h @ transform)[:, :3] * unit_scale
        faces = _triangulate(face_counts, face_indices)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        records.append(MeshRecord(path=path, vertices=vertices, faces=faces))

    if not records:
        raise RuntimeError(f"No mesh prims found under {root_path} in {args.gripper_usd}.")
    return records


def _detect_closing_axis(finger_records: list[MeshRecord]) -> np.ndarray:
    if len(finger_records) < 2:
        raise RuntimeError("Need at least two finger meshes to infer the closing axis. Pass --closing_axis explicitly.")
    centroids = np.asarray([record.centroid for record in finger_records], dtype=np.float64)
    spread = centroids.max(axis=0) - centroids.min(axis=0)
    if float(np.max(spread)) <= 1.0e-9:
        raise RuntimeError("Finger mesh centroids do not separate enough to infer the closing axis.")
    return AXIS_VECTORS[("x", "y", "z")[int(np.argmax(spread))]]


def _detect_approach_axis(records: list[MeshRecord], closing_axis: np.ndarray) -> np.ndarray:
    mins, maxs = _union_bounds(records)
    extents = maxs - mins
    closing_index = int(np.argmax(np.abs(closing_axis)))
    extents[closing_index] = -np.inf
    if not np.isfinite(extents).any():
        raise RuntimeError("Could not infer approach axis.")
    return AXIS_VECTORS[("x", "y", "z")[int(np.argmax(extents))]]


def _canonical_rotation(closing_axis: np.ndarray, approach_axis: np.ndarray) -> np.ndarray:
    closing_axis = closing_axis / np.linalg.norm(closing_axis)
    approach_axis = approach_axis / np.linalg.norm(approach_axis)
    if abs(float(np.dot(closing_axis, approach_axis))) > 1.0e-6:
        raise ValueError("closing_axis and approach_axis must be orthogonal.")
    side_axis = np.cross(approach_axis, closing_axis)
    side_axis = side_axis / np.linalg.norm(side_axis)
    return np.vstack([closing_axis, side_axis, approach_axis])


def _transform_records(records: list[MeshRecord], rotation: np.ndarray) -> list[MeshRecord]:
    return [
        MeshRecord(path=record.path, vertices=record.vertices @ rotation.T, faces=record.faces.copy())
        for record in records
    ]


def _union_bounds(records: list[MeshRecord]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.asarray([record.bounds[0] for record in records], dtype=np.float64)
    maxs = np.asarray([record.bounds[1] for record in records], dtype=np.float64)
    return mins.min(axis=0), maxs.max(axis=0)


def _estimate_sweep_volume(finger_records: list[MeshRecord]) -> tuple[np.ndarray, np.ndarray]:
    if len(finger_records) < 2:
        raise RuntimeError("Need at least two finger meshes to estimate the sweep volume.")

    bboxes = [record.bounds for record in finger_records]
    centroids = np.asarray([(bbox_min + bbox_max) / 2.0 for bbox_min, bbox_max in bboxes], dtype=np.float64)
    order = np.argsort(centroids[:, 0])
    left_bbox = bboxes[int(order[0])]
    right_bbox = bboxes[int(order[-1])]
    inner_min = float(left_bbox[1][0])
    inner_max = float(right_bbox[0][0])
    if inner_max <= inner_min:
        inner_min = float(centroids[int(order[0]), 0])
        inner_max = float(centroids[int(order[-1]), 0])

    union_min, union_max = _union_bounds(finger_records)
    sweep_min = union_min.copy()
    sweep_max = union_max.copy()
    sweep_min[0] = inner_min
    sweep_max[0] = inner_max
    return sweep_max - sweep_min, (sweep_min + sweep_max) / 2.0


def _merge_mesh_records(records: list[MeshRecord], trimesh: Any) -> Any:
    meshes = [trimesh.Trimesh(vertices=record.vertices, faces=record.faces, process=False) for record in records]
    return trimesh.util.concatenate(meshes)


def _discover_usd_joints(stage: Any, root_prim: Any) -> tuple[dict[str, float], dict[str, float], list[str]]:
    root_path = str(root_prim.GetPath())
    open_joints = {}
    close_joints = {}
    joint_names = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not _is_under_root(path, root_path):
            continue
        if prim.GetTypeName() not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            continue
        name = prim.GetName()
        lower_attr = prim.GetAttribute("physics:lowerLimit")
        upper_attr = prim.GetAttribute("physics:upperLimit")
        lower = lower_attr.Get() if lower_attr and lower_attr.HasAuthoredValueOpinion() else None
        upper = upper_attr.Get() if upper_attr and upper_attr.HasAuthoredValueOpinion() else None
        if lower is None or upper is None:
            continue
        joint_names.append(name)
        open_joints[name] = float(upper)
        close_joints[name] = float(lower)
    return open_joints, close_joints, joint_names


def _round_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _round_nested(value.tolist())
    if isinstance(value, list | tuple):
        return [_round_nested(item) for item in value]
    if isinstance(value, float | np.floating):
        return round(float(value), 9)
    return value


def _write_descriptor(
    args: argparse.Namespace,
    root_prim: Any,
    config: dict[str, Any],
    visual_records: list[MeshRecord],
    collision_records: list[MeshRecord],
    trimesh: Any,
) -> Path:
    output_root = args.output_dir
    if output_root is None:
        output_root = args.graspgenx_root.expanduser().resolve() / "assets" / "x_grippers"
    output_root = output_root.expanduser().resolve()
    descriptor_dir = output_root / args.name

    if descriptor_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{descriptor_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(descriptor_dir)
    descriptor_dir.mkdir(parents=True)

    config_path = descriptor_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(_round_nested(config), file, indent=4)
        file.write("\n")

    visual_mesh = _merge_mesh_records(visual_records, trimesh)
    visual_mesh.export(descriptor_dir / "vis_mesh.obj")

    collision_mesh = _merge_mesh_records(collision_records, trimesh)
    collision_mesh.export(descriptor_dir / "coll_mesh.obj")

    shutil.copy2(args.gripper_usd.expanduser().resolve(), descriptor_dir / "gripper.usd")
    metadata = {
        "source_usd": str(args.gripper_usd.expanduser().resolve()),
        "root_prim": str(root_prim.GetPath()),
        "note": "Generated by scripts/tools/create_graspgenx_gripper_from_usd.py.",
    }
    with (descriptor_dir / "README.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
        file.write("\n")

    return descriptor_dir


def main() -> None:
    """Create the descriptor folder."""
    args = parse_args()
    _Gf, Usd, UsdGeom, trimesh = _import_usd_and_mesh_modules()

    stage = Usd.Stage.Open(str(args.gripper_usd.expanduser().resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {args.gripper_usd}")

    root_prim = stage.GetPrimAtPath(args.root_prim) if args.root_prim else stage.GetDefaultPrim()
    if root_prim is None or not root_prim.IsValid():
        raise RuntimeError(f"Could not resolve root prim {args.root_prim!r} in {args.gripper_usd}.")

    records = _load_mesh_records(args, stage, root_prim, UsdGeom)
    visual_records = [record for record in records if re.search(args.visual_regex, record.path)]
    if not visual_records:
        visual_records = [record for record in records if not re.search(args.collision_regex, record.path)]
    if not visual_records:
        visual_records = records

    collision_records = [record for record in records if re.search(args.collision_regex, record.path)]
    if not collision_records:
        collision_records = visual_records

    finger_records = [record for record in visual_records if re.search(args.finger_regex, record.path)]
    if len(finger_records) < 2:
        print(
            f"[warn] Found {len(finger_records)} finger meshes with --finger_regex={args.finger_regex!r}; "
            "falling back to all visual meshes for sweep-volume estimation.",
            file=sys.stderr,
        )
        finger_records = visual_records

    closing_axis = _parse_axis(args.closing_axis) or _detect_closing_axis(finger_records)
    approach_axis = _parse_axis(args.approach_axis) or _detect_approach_axis(finger_records, closing_axis)
    canonical_rotation = _canonical_rotation(closing_axis, approach_axis)

    visual_records = _transform_records(visual_records, canonical_rotation)
    collision_records = _transform_records(collision_records, canonical_rotation)
    finger_records = _transform_records(finger_records, canonical_rotation)

    bbox_min, bbox_max = _union_bounds(visual_records)
    sweep_extents, sweep_offset = _estimate_sweep_volume(finger_records)
    if args.sweep_volume_extents is not None:
        sweep_extents = np.asarray(args.sweep_volume_extents, dtype=np.float64)
        sweep_offset = np.asarray(args.sweep_volume_offset, dtype=np.float64)
    else:
        bbox_width = float(bbox_max[0] - bbox_min[0])
        if bbox_width > 0.0 and float(sweep_extents[0]) < 0.5 * bbox_width:
            print(
                "[warn] Auto-estimated sweep width is much smaller than the gripper bbox width. "
                "The USD may be authored in a partially closed pose; pass --sweep_volume_extents "
                "and --sweep_volume_offset to describe the fully-open grasp volume.",
                file=sys.stderr,
            )

    if args.sweep_volume_mid_extents is not None:
        sweep_mid_extents = np.asarray(args.sweep_volume_mid_extents, dtype=np.float64)
        sweep_mid_offset = np.asarray(args.sweep_volume_mid_offset, dtype=np.float64)
    else:
        sweep_mid_extents = sweep_extents.copy()
        sweep_mid_extents[0] *= 0.55
        sweep_mid_offset = sweep_offset.copy()

    if args.fingertip is not None:
        fingertip = np.asarray(args.fingertip, dtype=np.float64)
    else:
        fingertip = sweep_offset + np.array([0.0, 0.0, sweep_extents[2] / 2.0])

    standoff = np.asarray(args.standoff if args.standoff is not None else [0.0, sweep_extents[2] / 2.0])
    open_joints, close_joints, discovered_joint_names = _discover_usd_joints(stage, root_prim)
    open_joints.update(_parse_joint_overrides(args.open_joint))
    close_joints.update(_parse_joint_overrides(args.close_joint))

    link_names = sorted(
        {str(record.path).split("/")[-2] for record in visual_records if len(str(record.path).split("/")) >= 2}
    )

    config = {
        "open": open_joints,
        "close": close_joints,
        "fingertip": fingertip,
        "sweep_volume": {
            "extents": sweep_extents,
            "offset": sweep_offset,
            "extents2": sweep_mid_extents,
            "offset2": sweep_mid_offset,
        },
        "links": link_names,
        "standoff": standoff,
        "bbox": [bbox_min, bbox_max],
        "symmetric": bool(args.symmetric),
        "type": args.gripper_type,
        "base_rotation": canonical_rotation,
        "source_usd": str(args.gripper_usd.expanduser().resolve()),
        "source_root_prim": str(root_prim.GetPath()),
    }

    descriptor_dir = _write_descriptor(args, root_prim, config, visual_records, collision_records, trimesh)

    print(f"Created GraspGenX descriptor: {descriptor_dir}")
    print(f"source root prim = {root_prim.GetPath()}")
    print(f"canonical closing axis = {_axis_name(closing_axis)} -> +x")
    print(f"canonical approach axis = {_axis_name(approach_axis)} -> +z")
    print(f"visual mesh prims = {len(visual_records)}")
    print(f"collision mesh prims = {len(collision_records)}")
    print(f"discovered joints = {discovered_joint_names}")
    print(f"sweep_volume.extents = {_round_nested(sweep_extents)}")
    print(f"sweep_volume.offset = {_round_nested(sweep_offset)}")
    print(f"bbox = {_round_nested([bbox_min, bbox_max])}")


if __name__ == "__main__":
    main()
