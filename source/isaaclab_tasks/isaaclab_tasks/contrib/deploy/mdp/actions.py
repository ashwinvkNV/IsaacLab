# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action terms for LEAPP export workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import DeployRelativeJointPositionActionCfg


class DeployRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action that exposes current joint positions as a LEAPP input."""

    def __init__(self, cfg: DeployRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        asset = self._asset
        if type(asset).__name__ == "_ArticulationWriteProxy":
            from leapp import annotate
            from leapp.utils.tensor_description import TensorSemantics

            from isaaclab.utils.leapp import InputKindEnum

            real_asset = object.__getattribute__(asset, "_real_asset")
            current_joint_pos = real_asset.data.joint_pos.torch[:, self._joint_ids]
            current_joint_pos = annotate.input_tensors(
                self._env.unwrapped.spec.id,
                TensorSemantics(
                    name=f"{self.cfg.asset_name}_current_joint_pos",
                    ref=current_joint_pos,
                    kind=InputKindEnum.JOINT_POSITION,
                    element_names=self._joint_names,
                    extra={"isaaclab_connection": f"state:{self.cfg.asset_name}:joint_pos"},
                ),
            )
        else:
            current_joint_pos = asset.data.joint_pos.torch[:, self._joint_ids]

        current_actions = self.processed_actions + current_joint_pos
        self._asset.set_joint_position_target_index(target=current_actions, joint_ids=self._joint_ids)
