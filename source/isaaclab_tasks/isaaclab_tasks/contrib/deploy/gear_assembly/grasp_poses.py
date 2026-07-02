# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for loading generated gear assembly grasp-pose artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

GEAR_TYPES = ("gear_small", "gear_medium", "gear_large")
GRASP_POSE_FORMAT = "isaaclab_gear_grasp_pose"
GRASP_POSE_FORMAT_VERSION = 1


def resolve_grasp_pose_file(default_path: str | Path, env_var: str) -> Path:
    """Resolve a grasp-pose artifact path from an environment variable.

    Args:
        default_path: Repository default artifact path.
        env_var: Environment variable that can override ``default_path``.

    Returns:
        Resolved path to the artifact.
    """
    return Path(os.environ.get(env_var, default_path)).expanduser()


def load_gear_grasp_pose_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a generated gear assembly grasp-pose artifact.

    Args:
        path: Path to a JSON artifact produced by the GraspGenX generation script.

    Returns:
        Validated artifact fields consumed by gear assembly configs.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("format") != GRASP_POSE_FORMAT:
        raise ValueError(f"{path} has unsupported format {data.get('format')!r}. Expected {GRASP_POSE_FORMAT!r}.")
    if int(data.get("format_version", -1)) != GRASP_POSE_FORMAT_VERSION:
        raise ValueError(
            f"{path} has unsupported format_version {data.get('format_version')!r}. "
            f"Expected {GRASP_POSE_FORMAT_VERSION}."
        )

    gear_offsets_grasp = _validate_vector_map(path, data, "gear_offsets_grasp", 3)
    gear_rot_offsets_grasp = _validate_vector_map(path, data, "gear_rot_offsets_grasp", 4)
    gear_rot_offsets_grasp = {
        gear_type: _normalize_quat_xyzw(values) for gear_type, values in gear_rot_offsets_grasp.items()
    }

    result = {
        "gear_offsets_grasp": gear_offsets_grasp,
        "gear_rot_offsets_grasp": gear_rot_offsets_grasp,
    }

    for optional_key in ("hand_grasp_width", "hand_close_width"):
        if optional_key in data:
            result[optional_key] = _validate_scalar_map(path, data, optional_key)

    return result


def _validate_vector_map(path: Path, data: dict[str, Any], key: str, length: int) -> dict[str, list[float]]:
    if key not in data:
        raise ValueError(f"{path} is missing required key {key!r}.")
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{path} key {key!r} must be a mapping.")

    result: dict[str, list[float]] = {}
    for gear_type in GEAR_TYPES:
        if gear_type not in value:
            raise ValueError(f"{path} key {key!r} is missing {gear_type!r}.")
        vector = value[gear_type]
        if not isinstance(vector, list) or len(vector) != length:
            raise ValueError(f"{path} key {key!r}.{gear_type} must be a list with {length} values.")
        result[gear_type] = [float(entry) for entry in vector]
    return result


def _validate_scalar_map(path: Path, data: dict[str, Any], key: str) -> dict[str, float]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{path} key {key!r} must be a mapping.")

    result: dict[str, float] = {}
    for gear_type in GEAR_TYPES:
        if gear_type not in value:
            raise ValueError(f"{path} key {key!r} is missing {gear_type!r}.")
        result[gear_type] = float(value[gear_type])
    return result


def _normalize_quat_xyzw(quat: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 0.0:
        raise ValueError("Grasp rotation quaternion must have non-zero norm.")
    return [value / norm for value in quat]
