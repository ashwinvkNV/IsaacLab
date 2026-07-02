#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate gear assembly grasp-pose artifacts with NVlabs/GraspGenX.

Run this script with the GraspGenX Python environment, not the Isaac Lab
environment. The output JSON is intentionally lightweight so Isaac Lab can
consume generated poses without depending on GraspGenX at training time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

GEAR_TYPES = ("gear_small", "gear_medium", "gear_large")
GRASP_POSE_FORMAT = "isaaclab_gear_grasp_pose"
GRASP_POSE_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graspgenx_root",
        type=Path,
        default=None,
        help="Path to a GraspGenX checkout. Adds the checkout to PYTHONPATH before importing GraspGenX.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help=(
            "Path to a GraspGenX checkpoint root containing gen/ and dis/. "
            "If omitted, GraspGenX resolves its default release checkpoint."
        ),
    )
    parser.add_argument("--gen_pth", type=str, default=None, help="Generator .pth filename.")
    parser.add_argument("--dis_pth", type=str, default=None, help="Discriminator .pth filename.")
    parser.add_argument("--gripper_name", type=str, required=True, help="GraspGenX gripper descriptor name.")
    parser.add_argument(
        "--target_gripper_name",
        type=str,
        default=None,
        help="Optional real gripper name if --gripper_name is a descriptor surrogate.",
    )
    parser.add_argument(
        "--assets_dir",
        type=Path,
        default=None,
        help="GraspGenX assets directory. Defaults to <graspgenx_root>/assets when --graspgenx_root is set.",
    )
    parser.add_argument("--gear_small_mesh", type=Path, required=True, help="Mesh for the small gear.")
    parser.add_argument("--gear_medium_mesh", type=Path, required=True, help="Mesh for the medium gear.")
    parser.add_argument("--gear_large_mesh", type=Path, required=True, help="Mesh for the large gear.")
    parser.add_argument("--mesh_scale", type=float, default=1.0, help="Scale factor applied before sampling.")
    parser.add_argument("--num_sample_points", type=int, default=3500, help="Surface points sampled from each mesh.")
    parser.add_argument("--num_grasps", type=int, default=100, help="Grasps generated per inference attempt.")
    parser.add_argument("--topk_num_grasps", type=int, default=-1, help="Top-k grasps kept per inference attempt.")
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=-1.0,
        help="Grasp confidence threshold. Use -1.0 to rank by confidence and keep top-k.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="graspmoe",
        choices=["diffusion", "graspmoe"],
        help="Grasp planner. GraspMoE unions diffusion samples with OBB-swept candidates.",
    )
    parser.add_argument("--moe_num_yaws", type=int, default=144, help="[graspmoe] Number of yaw samples.")
    parser.add_argument(
        "--moe_z_offsets_cm",
        type=str,
        default="-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3",
        help="[graspmoe] Comma-separated Z offsets in cm relative to the OBB top.",
    )
    parser.add_argument("--moe_outlier_threshold", type=float, default=0.014, help="[graspmoe] Outlier threshold.")
    parser.add_argument("--moe_outlier_k", type=int, default=20, help="[graspmoe] Outlier k-NN.")
    parser.add_argument("--moe_obb_mode", type=str, default="advanced", choices=["advanced", "pca"])
    parser.add_argument("--moe_skip_obb_rule", type=str, default="never", choices=["auto", "never"])
    parser.add_argument(
        "--moe_obb_density",
        type=str,
        default="dense-topandside",
        choices=["sparse", "dense", "dense-topandside"],
    )
    parser.add_argument("--moe_obb_position_spacing_cm", type=float, default=0.25)
    parser.add_argument(
        "--ee_from_grasp_pos",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Translation from the GraspGenX gripper frame to the Isaac Lab task EE frame.",
    )
    parser.add_argument(
        "--ee_from_grasp_quat_xyzw",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z", "W"),
        help="Rotation from the GraspGenX gripper frame to the Isaac Lab task EE frame.",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="task_frame",
        choices=["top_confidence", "task_frame"],
        help="How to select one candidate per gear.",
    )
    parser.add_argument(
        "--task_grasp_offset_z",
        type=float,
        default=None,
        help="Expected task-frame grasp offset Z [m], required by --selection_mode task_frame.",
    )
    parser.add_argument(
        "--task_grasp_quat_xyzw",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "W"),
        help="Expected object-to-task-EE rotation, required by --selection_mode task_frame.",
    )
    parser.add_argument("--task_xy_weight", type=float, default=45.0)
    parser.add_argument("--task_z_weight", type=float, default=45.0)
    parser.add_argument("--task_rot_weight", type=float, default=3.0)
    parser.add_argument("--task_confidence_weight", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mesh sampling and GraspGenX.")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to write the Isaac Lab JSON artifact.")
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional note embedded in the artifact, for example when a surrogate gripper descriptor is used.",
    )
    args = parser.parse_args()
    if args.selection_mode == "task_frame":
        if args.task_grasp_offset_z is None:
            parser.error("--task_grasp_offset_z is required when --selection_mode task_frame.")
        if args.task_grasp_quat_xyzw is None:
            parser.error("--task_grasp_quat_xyzw is required when --selection_mode task_frame.")
    return args


def _prepare_graspgenx_imports(graspgenx_root: Path | None):
    """Import GraspGenX after applying the optional checkout path."""
    if graspgenx_root is not None:
        graspgenx_root = graspgenx_root.expanduser().resolve()
        sys.path.insert(0, str(graspgenx_root))

    try:
        import torch
        import trimesh
        import trimesh.transformations as tra
        from graspgenx import get_checkpoints_version_dir
        from graspgenx.dataset.dataset_utils import sample_points
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.samplers import run_planner_on_object
        from graspgenx.utils.checkpoint_io import load_model_cfg
    except ImportError as exc:
        raise ImportError(
            "GraspGenX dependencies are required. Run this script with the GraspGenX uv environment, e.g. "
            "`/path/to/GraspGenX/.venv/bin/python scripts/tools/generate_gear_grasp_poses_graspgenx.py ...`."
        ) from exc

    return torch, trimesh, tra, sample_points, get_checkpoints_version_dir, GraspGenXSampler, load_model_cfg, run_planner_on_object


def _load_mesh_data(mesh_file: Path, scale: float, num_sample_points: int, trimesh: Any, tra: Any, sample_points: Any):
    """Load a mesh or point cloud and return centered points, restore transform, and mesh centroid."""
    mesh_file = mesh_file.expanduser().resolve()
    suffix = mesh_file.suffix.lower()

    if suffix == ".ply":
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(mesh_file))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        mesh_centroid = xyz.mean(axis=0).astype(np.float64)
        point_indices = sample_points(xyz, num_sample_points)
        xyz = xyz[point_indices]
        obj = None
    elif suffix in (".usd", ".usda", ".usdc", ".usdz"):
        import scene_synthesizer as synth

        asset = synth.Asset(str(mesh_file))
        obj = asset.mesh()
        obj.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.asarray(xyz, dtype=np.float32)
        mesh_centroid = np.asarray(obj.centroid, dtype=np.float64)
    else:
        obj = trimesh.load(str(mesh_file))
        obj.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.asarray(xyz, dtype=np.float32)
        mesh_centroid = np.asarray(obj.centroid, dtype=np.float64)

    transform_subtract_mean = tra.translation_matrix(-xyz.mean(axis=0))
    xyz = tra.transform_points(xyz, transform_subtract_mean)
    if obj is not None:
        obj.apply_transform(transform_subtract_mean)

    return xyz, tra.inverse_matrix(transform_subtract_mean), mesh_centroid


def _round_list(values: np.ndarray, digits: int = 9) -> list[float]:
    return [round(float(value), digits) for value in values]


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _matrix_from_pos_quat_xyzw(pos: tuple[float, ...] | list[float], quat_xyzw: tuple[float, ...] | list[float], tra: Any):
    quat_xyzw_array = np.asarray(quat_xyzw, dtype=np.float64)
    quat_xyzw_array = quat_xyzw_array / np.linalg.norm(quat_xyzw_array)
    transform = tra.quaternion_matrix(
        [quat_xyzw_array[3], quat_xyzw_array[0], quat_xyzw_array[1], quat_xyzw_array[2]]
    )
    transform[:3, 3] = np.asarray(pos, dtype=np.float64)
    return transform


def _quat_xyzw_from_matrix(transform: np.ndarray, tra: Any) -> np.ndarray:
    quat_wxyz = np.asarray(tra.quaternion_from_matrix(transform), dtype=np.float64)
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    if quat_wxyz[0] < 0.0:
        quat_wxyz = -quat_wxyz
    return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _quat_angle_rad(quat_xyzw_1: np.ndarray, quat_xyzw_2: np.ndarray) -> float:
    quat_xyzw_1 = quat_xyzw_1 / np.linalg.norm(quat_xyzw_1)
    quat_xyzw_2 = quat_xyzw_2 / np.linalg.norm(quat_xyzw_2)
    dot = abs(float(np.dot(quat_xyzw_1, quat_xyzw_2)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _pose_to_isaaclab_grasp_fields(transform_object_gripper: np.ndarray, tra: Any) -> tuple[list[float], list[float], list[float]]:
    """Convert a GraspGenX object-frame pose to Isaac Lab grasp offset fields.

    GraspGenX returns the gripper pose in the object frame. Isaac Lab stores the
    rotation as ``gear_rot_offsets_grasp`` in xyzw order, and stores
    ``gear_offsets_grasp`` in the rotated gripper frame.
    """
    rotation_object_gripper = transform_object_gripper[:3, :3]
    position_object_gripper = transform_object_gripper[:3, 3]
    position_gripper_frame = rotation_object_gripper.T @ position_object_gripper

    quat_wxyz = np.asarray(tra.quaternion_from_matrix(transform_object_gripper), dtype=np.float64)
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    if quat_wxyz[0] < 0.0:
        quat_wxyz = -quat_wxyz
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)

    return _round_list(position_gripper_frame), _round_list(quat_xyzw), _round_list(quat_wxyz)


def _select_candidate(
    candidate_poses: np.ndarray,
    confidence: np.ndarray,
    branch_tags: list[str],
    mesh_centroid: np.ndarray,
    args: argparse.Namespace,
    tra: Any,
) -> tuple[int, dict[str, Any]]:
    """Select one generated candidate to export."""
    if args.selection_mode == "top_confidence":
        best_idx = int(confidence.argmax())
        return best_idx, {"rank": "top_confidence", "branch": branch_tags[best_idx]}

    task_quat = np.asarray(args.task_grasp_quat_xyzw, dtype=np.float64)
    task_quat = task_quat / np.linalg.norm(task_quat)
    task_transform = _matrix_from_pos_quat_xyzw((0.0, 0.0, 0.0), tuple(task_quat), tra)
    target_offset = task_transform[:3, :3].T @ np.asarray([mesh_centroid[0], mesh_centroid[1], 0.0])
    target_offset[2] = float(args.task_grasp_offset_z)

    best_idx = 0
    best_score = float("inf")
    best_meta: dict[str, Any] = {}
    for index, transform in enumerate(candidate_poses):
        rotation_object_ee = transform[:3, :3]
        offset = rotation_object_ee.T @ transform[:3, 3]
        quat_xyzw = _quat_xyzw_from_matrix(transform, tra)
        xy_error = float(np.linalg.norm((offset - target_offset)[:2]))
        z_error = abs(float(offset[2] - target_offset[2]))
        rot_error = _quat_angle_rad(quat_xyzw, task_quat)
        score = (
            args.task_xy_weight * xy_error
            + args.task_z_weight * z_error
            + args.task_rot_weight * rot_error
            - args.task_confidence_weight * float(confidence[index])
        )
        if score < best_score:
            best_idx = index
            best_score = score
            best_meta = {
                "rank": "task_frame",
                "branch": branch_tags[index],
                "target_offset": _round_list(target_offset),
                "score": round(float(score), 9),
                "xy_error_m": round(xy_error, 9),
                "z_error_m": round(z_error, 9),
                "rotation_error_deg": round(float(np.rad2deg(rot_error)), 6),
            }

    return best_idx, best_meta


def _generate_one_pose(
    mesh_file: Path,
    args: argparse.Namespace,
    grasp_sampler: Any,
    trimesh: Any,
    tra: Any,
    sample_points: Any,
    GraspGenXSampler: Any,
    run_planner_on_object: Any,
) -> tuple[list[float], list[float], list[float], float, dict[str, Any]]:
    points, transform_restore_object_frame, mesh_centroid = _load_mesh_data(
        mesh_file, args.mesh_scale, args.num_sample_points, trimesh, tra, sample_points
    )
    if args.planner == "graspmoe":
        grasps, grasp_conf, branch_tags, _ = run_planner_on_object(
            points,
            grasp_sampler,
            planner=args.planner,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
            moe_num_yaws=args.moe_num_yaws,
            moe_z_offsets_cm=_parse_float_csv(args.moe_z_offsets_cm),
            moe_outlier_threshold=args.moe_outlier_threshold,
            moe_outlier_k=args.moe_outlier_k,
            moe_obb_mode=args.moe_obb_mode,
            moe_skip_obb_rule=args.moe_skip_obb_rule,
            moe_obb_density=args.moe_obb_density,
            moe_obb_position_spacing_cm=args.moe_obb_position_spacing_cm,
        )
    else:
        grasps, grasp_conf = GraspGenXSampler.run_inference(
            points,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
            remove_outliers=False,
        )
        branch_tags = ["diff"] * len(grasps)
    if len(grasps) == 0:
        raise RuntimeError(f"GraspGenX did not return any grasps for {mesh_file}.")

    grasp_conf = grasp_conf.detach().cpu().numpy() if hasattr(grasp_conf, "detach") else np.asarray(grasp_conf)
    grasps = grasps.detach().cpu().numpy() if hasattr(grasps, "detach") else np.asarray(grasps)
    grasps[:, 3, 3] = 1.0

    transform_grasp_to_ee = _matrix_from_pos_quat_xyzw(args.ee_from_grasp_pos, args.ee_from_grasp_quat_xyzw, tra)
    candidate_poses = np.asarray(
        [transform_restore_object_frame @ grasp @ transform_grasp_to_ee for grasp in grasps], dtype=np.float64
    )
    best_idx, selection = _select_candidate(candidate_poses, grasp_conf, branch_tags, mesh_centroid, args, tra)
    best_pose_object_frame = candidate_poses[best_idx]
    offset, quat_xyzw, quat_wxyz = _pose_to_isaaclab_grasp_fields(best_pose_object_frame, tra)
    if args.selection_mode == "task_frame":
        task_quat = np.asarray(args.task_grasp_quat_xyzw, dtype=np.float64)
        task_quat = task_quat / np.linalg.norm(task_quat)
        quat_array = np.asarray(quat_xyzw, dtype=np.float64)
        if float(np.dot(quat_array, task_quat)) < 0.0:
            quat_xyzw = _round_list(-quat_array)
            quat_wxyz = _round_list(-np.asarray(quat_wxyz, dtype=np.float64))
    selection["candidate_index"] = int(best_idx)
    selection["num_candidates"] = int(len(grasps))
    selection["mesh_centroid"] = _round_list(mesh_centroid)
    return offset, quat_xyzw, quat_wxyz, round(float(grasp_conf[best_idx]), 9), selection


def main() -> None:
    args = parse_args()

    torch, trimesh, tra, sample_points, get_checkpoints_version_dir, GraspGenXSampler, load_model_cfg, run_planner_on_object = (
        _prepare_graspgenx_imports(args.graspgenx_root)
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint_root = args.checkpoints or Path(get_checkpoints_version_dir())
    model_cfg = load_model_cfg(
        os.path.join(checkpoint_root, "gen"),
        os.path.join(checkpoint_root, "dis"),
        args.gen_pth,
        args.dis_pth,
    )

    assets_dir = args.assets_dir
    if assets_dir is None and args.graspgenx_root is not None:
        assets_dir = args.graspgenx_root.expanduser().resolve() / "assets"

    grasp_sampler = GraspGenXSampler(
        model_cfg,
        args.gripper_name,
        assets_dir=None if assets_dir is None else str(assets_dir.expanduser().resolve()),
    )

    meshes = {
        "gear_small": args.gear_small_mesh,
        "gear_medium": args.gear_medium_mesh,
        "gear_large": args.gear_large_mesh,
    }

    gear_offsets_grasp = {}
    gear_rot_offsets_grasp = {}
    confidence = {}
    pose_object_frame_quat_wxyz = {}
    per_gear_selection = {}

    for gear_type in GEAR_TYPES:
        print(f"Generating GraspGenX grasp for {gear_type}: {meshes[gear_type]}")
        offset, quat_xyzw, quat_wxyz, score, selection = _generate_one_pose(
            meshes[gear_type],
            args,
            grasp_sampler,
            trimesh,
            tra,
            sample_points,
            GraspGenXSampler,
            run_planner_on_object,
        )
        gear_offsets_grasp[gear_type] = offset
        gear_rot_offsets_grasp[gear_type] = quat_xyzw
        confidence[gear_type] = score
        pose_object_frame_quat_wxyz[gear_type] = quat_wxyz
        per_gear_selection[gear_type] = selection
        print(f"  confidence={score:.3f} offset={offset} quat_xyzw={quat_xyzw}")

    artifact = {
        "format": GRASP_POSE_FORMAT,
        "format_version": GRASP_POSE_FORMAT_VERSION,
        "source": "NVlabs/GraspGenX",
        "gripper_name": args.gripper_name,
        "object_meshes": {gear_type: meshes[gear_type].name for gear_type in GEAR_TYPES},
        "selection": {
            "rank": args.selection_mode,
            "planner": args.planner,
            "num_grasps": args.num_grasps,
            "topk_num_grasps": args.topk_num_grasps,
            "num_sample_points": args.num_sample_points,
            "grasp_threshold": args.grasp_threshold,
            "seed": args.seed,
            "mesh_scale": args.mesh_scale,
            "per_gear": per_gear_selection,
        },
        "confidence": confidence,
        "gear_offsets_grasp": gear_offsets_grasp,
        "gear_rot_offsets_grasp": gear_rot_offsets_grasp,
        "pose_object_frame_quat_wxyz": pose_object_frame_quat_wxyz,
        "ee_from_grasp": {
            "pos": _round_list(np.asarray(args.ee_from_grasp_pos, dtype=np.float64)),
            "quat_xyzw": _round_list(np.asarray(args.ee_from_grasp_quat_xyzw, dtype=np.float64)),
        },
    }
    if args.planner == "graspmoe":
        artifact["selection"]["graspmoe"] = {
            "moe_num_yaws": args.moe_num_yaws,
            "moe_z_offsets_cm": list(_parse_float_csv(args.moe_z_offsets_cm)),
            "moe_outlier_threshold": args.moe_outlier_threshold,
            "moe_outlier_k": args.moe_outlier_k,
            "moe_obb_mode": args.moe_obb_mode,
            "moe_skip_obb_rule": args.moe_skip_obb_rule,
            "moe_obb_density": args.moe_obb_density,
            "moe_obb_position_spacing_cm": args.moe_obb_position_spacing_cm,
        }
    if args.selection_mode == "task_frame":
        artifact["selection"]["task_frame"] = {
            "task_grasp_offset_z": args.task_grasp_offset_z,
            "task_grasp_quat_xyzw": _round_list(np.asarray(args.task_grasp_quat_xyzw, dtype=np.float64)),
            "task_xy_weight": args.task_xy_weight,
            "task_z_weight": args.task_z_weight,
            "task_rot_weight": args.task_rot_weight,
            "task_confidence_weight": args.task_confidence_weight,
        }
    if args.target_gripper_name is not None:
        artifact["target_gripper_name"] = args.target_gripper_name
    if args.notes is not None:
        artifact["notes"] = args.notes

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
