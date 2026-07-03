# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for gear assembly grasp-pose parameters."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

GEAR_TYPES = ("gear_small", "gear_medium", "gear_large")


def build_gear_grasp_offsets(gear_offsets_grasp: dict[str, list[float]], device: str | torch.device) -> torch.Tensor:
    """Build stacked per-gear grasp offsets [m] in gripper frame.

    Args:
        gear_offsets_grasp: Mapping from gear type to ``[x, y, z]`` offset [m].
        device: Torch device for the output tensor.

    Returns:
        Tensor of shape ``(3, 3)`` ordered as ``gear_small``, ``gear_medium``, ``gear_large``.
    """
    if not isinstance(gear_offsets_grasp, dict):
        raise TypeError(f"'gear_offsets_grasp' parameter must be a dict, got {type(gear_offsets_grasp).__name__}.")

    offsets = []
    for gear_type in GEAR_TYPES:
        if gear_type not in gear_offsets_grasp:
            raise ValueError(
                f"'{gear_type}' offset is required in 'gear_offsets_grasp'. "
                f"Found keys: {list(gear_offsets_grasp.keys())}"
            )
        if len(gear_offsets_grasp[gear_type]) != 3:
            raise ValueError(f"'{gear_type}' grasp offset must have 3 values.")
        offsets.append(torch.tensor(gear_offsets_grasp[gear_type], device=device, dtype=torch.float32))

    return torch.stack(offsets, dim=0)


def build_gear_grasp_rot_offsets(cfg_params: dict, device: str | torch.device) -> torch.Tensor:
    """Build stacked per-gear grasp rotation offsets in ``xyzw`` order.

    The legacy configuration uses one ``grasp_rot_offset`` for every gear type.
    GraspGenX can generate a distinct rotation for each object, represented by
    ``gear_rot_offsets_grasp``.

    Args:
        cfg_params: Manager term parameters containing either ``gear_rot_offsets_grasp`` or
            the legacy ``grasp_rot_offset``.
        device: Torch device for the output tensor.

    Returns:
        Tensor of shape ``(3, 4)`` ordered as ``gear_small``, ``gear_medium``, ``gear_large``.
    """
    gear_rot_offsets_grasp = cfg_params.get("gear_rot_offsets_grasp")
    if gear_rot_offsets_grasp is None:
        if "grasp_rot_offset" not in cfg_params:
            raise ValueError("Either 'gear_rot_offsets_grasp' or 'grasp_rot_offset' must be provided.")
        grasp_rot_offset = cfg_params["grasp_rot_offset"]
        if len(grasp_rot_offset) != 4:
            raise ValueError("'grasp_rot_offset' must have 4 values in xyzw order.")
        rot_offsets = torch.tensor([grasp_rot_offset] * len(GEAR_TYPES), device=device, dtype=torch.float32)
    else:
        if not isinstance(gear_rot_offsets_grasp, dict):
            raise TypeError(
                f"'gear_rot_offsets_grasp' parameter must be a dict, got {type(gear_rot_offsets_grasp).__name__}."
            )
        rot_offsets_list = []
        for gear_type in GEAR_TYPES:
            if gear_type not in gear_rot_offsets_grasp:
                raise ValueError(
                    f"'{gear_type}' rotation is required in 'gear_rot_offsets_grasp'. "
                    f"Found keys: {list(gear_rot_offsets_grasp.keys())}"
                )
            if len(gear_rot_offsets_grasp[gear_type]) != 4:
                raise ValueError(f"'{gear_type}' grasp rotation offset must have 4 values in xyzw order.")
            rot_offsets_list.append(torch.tensor(gear_rot_offsets_grasp[gear_type], device=device, dtype=torch.float32))
        rot_offsets = torch.stack(rot_offsets_list, dim=0)

    return math_utils.normalize(rot_offsets)
