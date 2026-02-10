# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based observation terms for the gear assembly manipulation environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .events import randomize_gear_type


class gear_shaft_pos_w(ManagerTermBase):
    """Object position in world frame with optional offset applied.

    This class-based term caches offset tensors and identity quaternions for efficient computation
    across all environments. For cable insertion, it returns the plug position with optional offset.

    Args:
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("gb300_socket").
        offset: A 3D offset list [x, y, z] to apply to the object position. Optional, defaults to [0, 0, 0].

    Returns:
        Object position tensor in the environment frame with shape (num_envs, 3).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the object position observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("gb300_socket"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

        # Get offset (optional, defaults to zero)
        offset = cfg.params.get("offset", [0.0, 0.0, 0.0])
        self.offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float32)

        # Pre-allocate buffers
        self.identity_quat = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("gb300_socket"),
        offset: list | None = None,
    ) -> torch.Tensor:
        """Compute object position in world frame with optional offset.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the asset (unused, kept for compatibility)
            offset: Optional offset to apply (unused, kept for compatibility)

        Returns:
            Object position tensor of shape (num_envs, 3)
        """
        # Get object position and orientation
        obj_pos = self.asset.data.root_pos_w
        obj_quat = self.asset.data.root_quat_w

        # Apply offset if non-zero
        if torch.any(self.offset_tensor != 0):
            offset_repeated = self.offset_tensor.unsqueeze(0).repeat(env.num_envs, 1)
            obj_pos, _ = combine_frame_transforms(obj_pos, obj_quat, offset_repeated, self.identity_quat)

        return obj_pos - env.scene.env_origins


class gear_shaft_quat_w(ManagerTermBase):
    """Object orientation in world frame.

    This class-based term returns the orientation of the object. The quaternion is canonicalized
    to ensure the w component is positive, reducing observation variation for the policy.

    Args:
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("gb300_socket").

    Returns:
        Object orientation tensor as a quaternion (w, x, y, z) with shape (num_envs, 4).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the object orientation observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("gb300_socket"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("gb300_socket"),
    ) -> torch.Tensor:
        """Compute object orientation in world frame.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the asset (unused, kept for compatibility)

        Returns:
            Object orientation tensor of shape (num_envs, 4)
        """
        # Get object quaternion
        obj_quat = self.asset.data.root_quat_w

        # Ensure w component is positive (q and -q represent the same rotation)
        # Pick one canonical form to reduce observation variation seen by the policy
        w_negative = obj_quat[:, 0] < 0
        positive_quat = obj_quat.clone()
        positive_quat[w_negative] = -obj_quat[w_negative]

        return positive_quat


class gear_pos_w(ManagerTermBase):
    """Object position in world frame.

    This class-based term returns the position of a specific object (e.g., plug) in the environment.

    Args:
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("gb300_plug").

    Returns:
        Object position tensor in the environment frame with shape (num_envs, 3).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the object position observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("gb300_plug"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("gb300_plug")) -> torch.Tensor:
        """Compute object position in world frame.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the asset (unused, kept for compatibility)

        Returns:
            Object position tensor of shape (num_envs, 3)
        """
        # Get object position
        obj_position = self.asset.data.root_pos_w

        return obj_position - env.scene.env_origins


class gear_quat_w(ManagerTermBase):
    """Object orientation in world frame.

    This class-based term returns the orientation of a specific object (e.g., plug) in the environment.
    The quaternion is canonicalized to ensure the w component is positive, reducing observation
    variation for the policy.

    Args:
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("gb300_plug").

    Returns:
        Object orientation tensor as a quaternion (w, x, y, z) with shape (num_envs, 4).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the object orientation observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("gb300_plug"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("gb300_plug")) -> torch.Tensor:
        """Compute object orientation in world frame.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the asset (unused, kept for compatibility)

        Returns:
            Object orientation tensor of shape (num_envs, 4)
        """
        # Get object quaternion
        obj_quat = self.asset.data.root_quat_w

        # Ensure w component is positive (q and -q represent the same rotation)
        # Pick one canonical form to reduce observation variation seen by the policy
        w_negative = obj_quat[:, 0] < 0
        obj_positive_quat = obj_quat.clone()
        obj_positive_quat[w_negative] = -obj_quat[w_negative]

        return obj_positive_quat
