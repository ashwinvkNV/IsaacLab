# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.envs import mdp
from isaaclab.test.mock_interfaces.assets.mock_articulation import MockArticulationData
from isaaclab.utils import math as math_utils
from isaaclab.utils.leapp import utils as leapp_utils
from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.leapp_semantics import InputKindEnum
from isaaclab.utils.leapp.proxy import _DataProxy, _EnvProxy

from isaaclab_tasks.contrib.deploy.mdp.actions import DeployRelativeJointPositionAction


class _TestScene(dict):
    """Minimal scene mapping for LEAPP proxy tests."""

    sensors = {}


def _make_articulation_data() -> tuple[MockArticulationData, torch.Tensor]:
    """Create mock articulation data with a non-identity root orientation."""
    data = MockArticulationData(num_instances=2, num_joints=0, num_bodies=1, device="cpu")
    root_pose_w = torch.zeros(2, 7, dtype=torch.float32)
    root_pose_w[:, 6] = 1.0
    root_pose_w[1, 3] = math.sin(math.pi / 4.0)
    root_pose_w[1, 6] = math.cos(math.pi / 4.0)
    data.set_root_link_pose_w(root_pose_w)
    return data, root_pose_w


def _capture_leapp_inputs(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture LEAPP input annotations while returning their tensor references."""
    annotated_inputs = []

    def _record_input_tensor(task_name, semantics):
        annotated_inputs.append((task_name, semantics))
        return semantics.ref

    monkeypatch.setattr(leapp_utils.annotate, "input_tensors", _record_input_tensor)
    return annotated_inputs


def test_direct_projected_gravity_b_read_preserves_vector3d_input(monkeypatch: pytest.MonkeyPatch):
    """Test direct data proxy reads keep projected gravity as its own semantic input."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, _ = _make_articulation_data()

    proxy = _DataProxy(
        data,
        entity_name="robot",
        task_name="Isaac-Velocity-Flat-G1-v0",
        property_resolution_cache={},
        cache={},
        input_name_resolver=lambda property_name: f"robot_{property_name}",
    )

    assert proxy.projected_gravity_b.torch.shape == (2, 3)

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1-v0"
    assert semantics.name == "robot_projected_gravity_b"
    assert semantics.kind == InputKindEnum.VECTOR3D
    assert semantics.extra == {"isaaclab_connection": "state:robot:projected_gravity_b"}


def test_projected_gravity_observation_exports_root_quat_w_input(monkeypatch: pytest.MonkeyPatch):
    """Test the projected-gravity observation is export-lowered through root quaternion."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, root_pose_w = _make_articulation_data()
    scene = _TestScene({"robot": SimpleNamespace(data=data)})
    env = SimpleNamespace(scene=scene)
    proxy_env = _EnvProxy(env, "Isaac-Velocity-Flat-G1-v0", {}, {})

    term_cfg = SimpleNamespace(func=mdp.projected_gravity, noise="noise")
    obs_manager = SimpleNamespace(_group_obs_term_cfgs={"policy": [term_cfg]}, compute=lambda *args, **kwargs: None)
    patcher = ExportPatcher(export_method="onnx-dynamo", required_obs_groups={"policy"})
    patcher.task_name = "Isaac-Velocity-Flat-G1-v0"
    patcher._patch_observation_manager(obs_manager, proxy_env)

    projected_gravity_b = term_cfg.func(env)

    expected = math_utils.quat_apply_inverse(
        root_pose_w[:, 3:7],
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32).expand(2, 3),
    )
    assert torch.allclose(projected_gravity_b, expected)
    assert term_cfg.noise is None

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1-v0"
    assert semantics.name == "robot_root_quat_w"
    assert semantics.kind == InputKindEnum.BODY_ROTATION
    assert semantics.extra == {"isaaclab_connection": "state:robot:root_quat_w"}


def test_deploy_relative_joint_position_action_exports_selected_current_joint_pos(monkeypatch: pytest.MonkeyPatch):
    """Test the deploy relative action exposes selected current joint positions as a LEAPP input."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data = MockArticulationData(num_instances=1, num_joints=4, num_bodies=0, device="cpu")
    data.joint_names = ["joint1", "finger_joint", "joint2", "finger_mimic_joint"]
    data.set_joint_pos(torch.tensor([[1.0, 10.0, 2.0, 20.0]], dtype=torch.float32))
    real_asset = SimpleNamespace(data=data)

    class _ArticulationWriteProxy:
        def __init__(self):
            self._real_asset = real_asset
            self.target = None
            self.joint_ids = None

        def set_joint_position_target_index(self, target, joint_ids):
            self.target = target
            self.joint_ids = joint_ids

    asset_proxy = _ArticulationWriteProxy()
    action = DeployRelativeJointPositionAction.__new__(DeployRelativeJointPositionAction)
    action.cfg = SimpleNamespace(asset_name="robot")
    action._env = SimpleNamespace(unwrapped=SimpleNamespace(spec=SimpleNamespace(id="Task-v0")))
    action._asset = asset_proxy
    action._joint_ids = [0, 2]
    action._joint_names = ["joint1", "joint2"]
    action._processed_actions = torch.tensor([[0.1, 0.2]], dtype=torch.float32)

    DeployRelativeJointPositionAction.apply_actions(action)

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Task-v0"
    assert semantics.name == "robot_current_joint_pos"
    assert semantics.kind == InputKindEnum.JOINT_POSITION
    assert semantics.element_names == [["joint1", "joint2"]]
    assert semantics.extra == {"isaaclab_connection": "state:robot:joint_pos"}
    assert torch.allclose(semantics.ref, torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.allclose(asset_proxy.target, torch.tensor([[1.1, 2.2]], dtype=torch.float32))
    assert asset_proxy.joint_ids == [0, 2]
