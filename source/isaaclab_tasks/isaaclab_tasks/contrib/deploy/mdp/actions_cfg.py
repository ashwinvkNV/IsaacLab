# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action configuration classes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass


@configclass
class DeployRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for deploy relative joint actions with explicit LEAPP current-joint input."""

    class_type: type | str = "{DIR}.actions:DeployRelativeJointPositionAction"
