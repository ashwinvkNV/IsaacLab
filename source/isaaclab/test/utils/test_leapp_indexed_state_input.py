# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for LEAPP indexed state inputs."""

from types import SimpleNamespace

import torch
import warp as wp

import isaaclab.utils.leapp.utils as leapp_utils
from isaaclab.utils.leapp.leapp_semantics import LeappTensorSemantics, joint_names_resolver
from isaaclab.utils.leapp.utils import TracedProxyArray
from isaaclab.utils.warp.proxy_array import ProxyArray


def test_traced_proxy_array_annotates_indexed_joint_subset(monkeypatch):
    """Indexed joint-state reads become indexed LEAPP input tensors."""

    all_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "left_outer_finger_joint",
        "right_outer_finger_joint",
    ]
    arm_joint_ids = list(range(7))
    arm_joint_names = all_joint_names[:7]
    base_tensor = torch.arange(13, dtype=torch.float32).reshape(1, 13)
    proxy_array = ProxyArray(wp.from_torch(base_tensor))
    real_data = SimpleNamespace(joint_names=all_joint_names)
    semantics = LeappTensorSemantics(
        kind="state/joint/position",
        element_names_resolver=joint_names_resolver,
    )
    cache = {}
    captured = []

    def fake_input_tensors(task_name, tensor_semantics):
        captured.append((task_name, tensor_semantics))
        return tensor_semantics.ref

    monkeypatch.setattr(leapp_utils.annotate, "input_tensors", fake_input_tensors)

    traced = TracedProxyArray(
        proxy_array,
        input_name="robot_joint_pos",
        semantics_meta=semantics,
        real_data=real_data,
        entity_name="robot",
        property_name="joint_pos",
        task_name="TestTask-v0",
        cache=cache,
    )

    result = traced.torch[:, arm_joint_ids]

    torch.testing.assert_close(result, base_tensor[:, arm_joint_ids])
    assert len(captured) == 1
    task_name, tensor_semantics = captured[0]
    assert task_name == "TestTask-v0"
    assert tensor_semantics.name == "robot_joint_pos"
    assert tensor_semantics.ref.shape == (1, 7)
    assert tensor_semantics.kind == "state/joint/position"
    assert tensor_semantics.element_names == [arm_joint_names]
    assert tensor_semantics.extra == {"isaaclab_connection": "state:robot:joint_pos"}

    cached = traced.torch[:, arm_joint_ids]

    torch.testing.assert_close(cached, result)
    assert len(captured) == 1
