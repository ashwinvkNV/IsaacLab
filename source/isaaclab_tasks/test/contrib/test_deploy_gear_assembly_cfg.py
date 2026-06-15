# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Deploy GearAssembly environment configuration defaults."""

import pytest

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.deploy.mdp.observations as deploy_observations
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.parametrize(
    "task_name",
    [
        "Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
        "Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
    ],
)
def test_ur10e_gear_assembly_default_num_envs(task_name: str):
    """UR10e GearAssembly training configs should fit on 16 GB GPUs by default."""
    env_cfg = parse_env_cfg(task_name)

    assert env_cfg.scene.num_envs == 2048


def test_rizon_gear_assembly_uses_deploy_joint_observations():
    """Rizon ROS inference should use deploy joint observations, not the generic full-articulation terms."""
    env_cfg = parse_env_cfg("Isaac-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference-v0")

    assert env_cfg.observations.policy.joint_pos.func is deploy_observations.joint_pos
    assert env_cfg.observations.policy.joint_vel.func is deploy_observations.joint_vel
    assert env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
