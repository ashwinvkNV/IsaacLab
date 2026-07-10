# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based event terms specific to the gear assembly manipulation environments."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.contrib.automate import factory_control as fc

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


class randomize_gear_type(ManagerTermBase):
    """Randomize and manage the gear type being used for each environment.

    This class stores the current gear type for each environment and provides a mapping
    from gear type names to indices. It serves as the central manager for gear type state
    that other MDP terms depend on.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the gear type randomization term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Extract gear types from config (required parameter)
        if "gear_types" not in cfg.params:
            raise ValueError("'gear_types' parameter is required in randomize_gear_type configuration")
        self.gear_types: list[str] = cfg.params["gear_types"]

        # Create gear type mapping (shared across all terms)
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}

        # Store current gear type for each environment (as list for easy access)
        # Initialize all to first gear type in the list
        self._current_gear_type = [self.gear_types[0]] * env.num_envs

        # Store current gear type indices as tensor for efficient vectorized access
        # Initialize all to first gear type index
        first_gear_idx = self.gear_type_map[self.gear_types[0]]
        self._current_gear_type_indices = torch.full(
            (env.num_envs,), first_gear_idx, device=env.device, dtype=torch.long
        )

        # Store reference on environment for other terms to access
        env._gear_type_manager = self

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        gear_types: list[str] = ["gear_small", "gear_medium", "gear_large"],
    ):
        """Randomize the gear type for specified environments.

        Args:
            env: The environment containing the assets
            env_ids: Environment IDs to randomize
            gear_types: List of available gear types to choose from
        """
        # Randomly select gear type for each environment
        # Use the parameter passed to __call__ (not self.gear_types) to allow runtime overrides
        for env_id in env_ids.tolist():
            chosen_gear = random.choice(gear_types)
            self._current_gear_type[env_id] = chosen_gear
            self._current_gear_type_indices[env_id] = self.gear_type_map[chosen_gear]

    def get_gear_type(self, env_id: int) -> str:
        """Get the current gear type for a specific environment."""
        return self._current_gear_type[env_id]

    def get_all_gear_types(self) -> list[str]:
        """Get current gear types for all environments."""
        return self._current_gear_type

    def get_all_gear_type_indices(self) -> torch.Tensor:
        """Get current gear type indices for all environments as a tensor.

        Returns:
            Tensor of shape (num_envs,) with gear type indices (0=small, 1=medium, 2=large)
        """
        return self._current_gear_type_indices


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """Resolve optional event env ids into a device tensor."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _compute_active_gear_pose_errors(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute active gear pose errors relative to the gear base."""
    if not hasattr(env, "_gear_type_manager"):
        raise RuntimeError(
            "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
            "before logging gear insertion metrics."
        )

    env_ids = _resolve_env_ids(env, env_ids)
    base_asset = env.scene[asset_cfg.name]
    gear_assets = {
        "gear_small": env.scene["factory_gear_small"],
        "gear_medium": env.scene["factory_gear_medium"],
        "gear_large": env.scene["factory_gear_large"],
    }

    gear_type_indices = env._gear_type_manager.get_all_gear_type_indices()[env_ids]
    all_gear_pos = torch.stack(
        [
            gear_assets["gear_small"].data.root_pos_w.torch[env_ids],
            gear_assets["gear_medium"].data.root_pos_w.torch[env_ids],
            gear_assets["gear_large"].data.root_pos_w.torch[env_ids],
        ],
        dim=1,
    )
    gear_pos = all_gear_pos[torch.arange(len(env_ids), device=env.device), gear_type_indices]
    base_pos = base_asset.data.root_pos_w.torch[env_ids]

    pos_error = gear_pos - base_pos
    xy_error = torch.linalg.norm(pos_error[:, :2], dim=-1)
    z_error = pos_error[:, 2]
    pose_error = torch.maximum(xy_error, torch.abs(z_error))
    distance_error = torch.linalg.norm(pos_error, dim=-1)
    return pose_error, xy_error, z_error, distance_error


def log_gear_insertion_pose_error_metrics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
    pose_error_thresholds: tuple[float, ...] = (0.001, 0.003, 0.005),
) -> None:
    """Log insertion pose-error metrics for the active gear against the gear base.

    The Factory gear assets are authored such that an inserted gear has the same
    root position as the gear base. This metric intentionally logs only to
    ``env.extras["log"]`` and does not contribute to the reward.
    """
    pose_error, xy_error, z_error, distance_error = _compute_active_gear_pose_errors(env, env_ids, asset_cfg)
    if not hasattr(env, "extras"):
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}

    for threshold in pose_error_thresholds:
        threshold_mm = int(round(threshold * 1000.0))
        success = pose_error <= threshold
        env.extras["log"][f"gear_success/pose_error_success_rate_{threshold_mm}mm"] = (
            success.float().mean().item()
        )
    env.extras["log"]["gear_success/pose_error_mean_m"] = pose_error.mean().item()
    env.extras["log"]["gear_success/pose_error_min_m"] = pose_error.min().item()
    env.extras["log"]["gear_success/xy_error_mean_m"] = xy_error.mean().item()
    env.extras["log"]["gear_success/xy_error_min_m"] = xy_error.min().item()
    env.extras["log"]["gear_success/z_error_mean_m"] = z_error.mean().item()
    env.extras["log"]["gear_success/abs_z_error_min_m"] = torch.abs(z_error).min().item()
    env.extras["log"]["gear_success/distance_error_min_m"] = distance_error.min().item()


def log_latched_gear_insertion_success_metrics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
    pose_error_thresholds: tuple[float, ...] = (0.001, 0.003, 0.005),
) -> None:
    """Log episode-level insertion success latched over the full episode.

    A success is latched when the active gear pose error falls below a threshold
    at any point during the episode. The episode only counts as successful if it
    later ends by timeout without any non-timeout termination such as gear drop
    or orientation failure. This term logs metrics only and does not affect
    rewards or termination behavior.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    thresholds = torch.tensor(pose_error_thresholds, device=env.device, dtype=torch.float32)
    state = _get_latched_gear_success_state(env, thresholds)

    if not hasattr(env, "extras"):
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}

    done = env.reset_buf[env_ids].bool()
    done_env_ids = env_ids[done]
    done_count = done_env_ids.numel()
    if done_count > 0:
        timeouts = env.reset_time_outs[done_env_ids].bool()
        failures = env.reset_terminated[done_env_ids].bool()
        clean_timeouts = timeouts & ~failures
        gear_drop_failures = _get_termination_term_values(env, "gear_dropped", done_env_ids) & failures
        gear_orientation_failures = (
            _get_termination_term_values(env, "gear_orientation_exceeded", done_env_ids) & failures
        )
        inserted = state["latched_success"][done_env_ids]
        successful = clean_timeouts.unsqueeze(-1) & state["latched_success"][done_env_ids]

        state["episode_count"] += float(done_count)
        state["timeout_count"] += timeouts.sum().to(torch.float64)
        state["clean_timeout_count"] += clean_timeouts.sum().to(torch.float64)
        state["failure_count"] += failures.sum().to(torch.float64)
        state["inserted_count"] += inserted.sum(dim=0).to(torch.float64)
        state["inserted_timeout_count"] += (inserted & timeouts.unsqueeze(-1)).sum(dim=0).to(torch.float64)
        state["inserted_clean_timeout_count"] += (
            inserted & clean_timeouts.unsqueeze(-1)
        ).sum(dim=0).to(torch.float64)
        state["inserted_failure_count"] += (inserted & failures.unsqueeze(-1)).sum(dim=0).to(torch.float64)
        state["inserted_drop_failure_count"] += (
            inserted & gear_drop_failures.unsqueeze(-1)
        ).sum(dim=0).to(torch.float64)
        state["inserted_orientation_failure_count"] += (
            inserted & gear_orientation_failures.unsqueeze(-1)
        ).sum(dim=0).to(torch.float64)
        state["success_count"] += successful.sum(dim=0).to(torch.float64)
        state["latched_success"][done_env_ids] = False

    active_env_ids = env_ids[~done]
    if active_env_ids.numel() > 0:
        pose_error, _, _, _ = _compute_active_gear_pose_errors(env, active_env_ids, asset_cfg)
        state["latched_success"][active_env_ids] |= pose_error.unsqueeze(-1) <= thresholds

    log = env.extras["log"]
    episode_count = torch.clamp(state["episode_count"], min=1.0)
    clean_timeout_count = torch.clamp(state["clean_timeout_count"], min=1.0)
    log["gear_success/episode_count"] = state["episode_count"].item()
    log["gear_success/episode_timeout_count"] = state["timeout_count"].item()
    log["gear_success/episode_clean_timeout_count"] = state["clean_timeout_count"].item()
    log["gear_success/episode_failure_count"] = state["failure_count"].item()
    log["gear_success/episode_timeout_fraction"] = (state["timeout_count"] / episode_count).item()
    log["gear_success/episode_failure_fraction"] = (state["failure_count"] / episode_count).item()

    for index, threshold in enumerate(pose_error_thresholds):
        threshold_mm = int(round(threshold * 1000.0))
        success_count = state["success_count"][index]
        inserted_count = state["inserted_count"][index]
        inserted_count_clamped = torch.clamp(inserted_count, min=1.0)
        inserted_timeout_count = state["inserted_timeout_count"][index]
        inserted_clean_timeout_count = state["inserted_clean_timeout_count"][index]
        inserted_failure_count = state["inserted_failure_count"][index]
        inserted_drop_failure_count = state["inserted_drop_failure_count"][index]
        inserted_orientation_failure_count = state["inserted_orientation_failure_count"][index]
        log[f"gear_success/episode_success_count_{threshold_mm}mm"] = success_count.item()
        log[f"gear_success/episode_success_rate_{threshold_mm}mm"] = (success_count / episode_count).item()
        log[f"gear_success/clean_timeout_success_rate_{threshold_mm}mm"] = (
            success_count / clean_timeout_count
        ).item()
        log[f"gear_success/episode_inserted_count_{threshold_mm}mm"] = inserted_count.item()
        log[f"gear_success/episode_inserted_rate_{threshold_mm}mm"] = (inserted_count / episode_count).item()
        log[f"gear_success/episode_inserted_then_timeout_rate_{threshold_mm}mm"] = (
            inserted_timeout_count / episode_count
        ).item()
        log[f"gear_success/episode_inserted_then_clean_timeout_rate_{threshold_mm}mm"] = (
            inserted_clean_timeout_count / episode_count
        ).item()
        log[f"gear_success/episode_inserted_then_failure_rate_{threshold_mm}mm"] = (
            inserted_failure_count / episode_count
        ).item()
        log[f"gear_success/episode_inserted_then_drop_failure_rate_{threshold_mm}mm"] = (
            inserted_drop_failure_count / episode_count
        ).item()
        log[f"gear_success/episode_inserted_then_orientation_failure_rate_{threshold_mm}mm"] = (
            inserted_orientation_failure_count / episode_count
        ).item()
        log[f"gear_success/inserted_then_timeout_rate_{threshold_mm}mm"] = (
            inserted_timeout_count / inserted_count_clamped
        ).item()
        log[f"gear_success/inserted_then_clean_timeout_rate_{threshold_mm}mm"] = (
            inserted_clean_timeout_count / inserted_count_clamped
        ).item()
        log[f"gear_success/inserted_then_failure_rate_{threshold_mm}mm"] = (
            inserted_failure_count / inserted_count_clamped
        ).item()
        log[f"gear_success/inserted_then_drop_failure_rate_{threshold_mm}mm"] = (
            inserted_drop_failure_count / inserted_count_clamped
        ).item()
        log[f"gear_success/inserted_then_orientation_failure_rate_{threshold_mm}mm"] = (
            inserted_orientation_failure_count / inserted_count_clamped
        ).item()
        log[f"gear_success/latched_pose_error_rate_{threshold_mm}mm"] = (
            state["latched_success"][:, index].float().mean().item()
        )


def _get_termination_term_values(env: ManagerBasedEnv, term_name: str, env_ids: torch.Tensor) -> torch.Tensor:
    """Return latest values for a termination term, or false if unavailable."""
    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is None:
        return torch.zeros(env_ids.shape, device=env.device, dtype=torch.bool)
    if term_name not in termination_manager.active_terms:
        return torch.zeros(env_ids.shape, device=env.device, dtype=torch.bool)
    return termination_manager.get_term(term_name)[env_ids].bool()


def _get_latched_gear_success_state(env: ManagerBasedEnv, thresholds: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return persistent state for latched episode success metrics."""
    state = getattr(env, "_latched_gear_insertion_success_metrics", None)
    required_keys = (
        "pose_error_thresholds",
        "latched_success",
        "inserted_count",
        "inserted_timeout_count",
        "inserted_clean_timeout_count",
        "inserted_failure_count",
        "inserted_drop_failure_count",
        "inserted_orientation_failure_count",
    )
    needs_init = (
        state is None
        or any(key not in state for key in required_keys)
        or state["pose_error_thresholds"].shape != thresholds.shape
        or not torch.allclose(state["pose_error_thresholds"], thresholds)
        or state["latched_success"].shape[0] != env.num_envs
    )
    if not needs_init:
        return state

    state = {
        "pose_error_thresholds": thresholds,
        "latched_success": torch.zeros((env.num_envs, len(thresholds)), device=env.device, dtype=torch.bool),
        "episode_count": torch.zeros((), device=env.device, dtype=torch.float64),
        "timeout_count": torch.zeros((), device=env.device, dtype=torch.float64),
        "clean_timeout_count": torch.zeros((), device=env.device, dtype=torch.float64),
        "failure_count": torch.zeros((), device=env.device, dtype=torch.float64),
        "inserted_count": torch.zeros(len(thresholds), device=env.device, dtype=torch.float64),
        "inserted_timeout_count": torch.zeros(len(thresholds), device=env.device, dtype=torch.float64),
        "inserted_clean_timeout_count": torch.zeros(len(thresholds), device=env.device, dtype=torch.float64),
        "inserted_failure_count": torch.zeros(len(thresholds), device=env.device, dtype=torch.float64),
        "inserted_drop_failure_count": torch.zeros(len(thresholds), device=env.device, dtype=torch.float64),
        "inserted_orientation_failure_count": torch.zeros(len(thresholds), device=env.device, dtype=torch.float64),
        "success_count": torch.zeros(len(thresholds), device=env.device, dtype=torch.float64),
    }
    env._latched_gear_insertion_success_metrics = state
    return state


class set_robot_to_grasp_pose(ManagerTermBase):
    """Set robot to grasp pose using IK with pre-cached tensors.

    This class-based term caches all required tensors and gear offsets during initialization,
    avoiding repeated allocations and lookups during execution.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the set robot to grasp pose term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get robot asset configuration
        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset: Articulation = env.scene[self.robot_asset_cfg.name]

        # Get robot-specific parameters from environment config (all required)
        # Validate required parameters
        if "end_effector_body_name" not in cfg.params:
            raise ValueError(
                "'end_effector_body_name' parameter is required in set_robot_to_grasp_pose configuration. "
                "Example: 'wrist_3_link'"
            )
        if "num_arm_joints" not in cfg.params:
            raise ValueError(
                "'num_arm_joints' parameter is required in set_robot_to_grasp_pose configuration. Example: 6 for UR10e"
            )
        if "grasp_rot_offset" not in cfg.params:
            raise ValueError(
                "'grasp_rot_offset' parameter is required in set_robot_to_grasp_pose configuration. "
                "It should be a quaternion [x, y, z, w]. Example: [0.707, 0.707, 0.0, 0.0]"
            )
        if "gripper_joint_setter_func" not in cfg.params:
            raise ValueError(
                "'gripper_joint_setter_func' parameter is required in set_robot_to_grasp_pose configuration. "
                "It should be a function to set gripper joint positions."
            )

        self.end_effector_body_name = cfg.params["end_effector_body_name"]
        self.num_arm_joints = cfg.params["num_arm_joints"]
        self.gripper_joint_setter_func = cfg.params["gripper_joint_setter_func"]

        # Pre-cache gear grasp offsets as tensors (required parameter)
        if "gear_offsets_grasp" not in cfg.params:
            raise ValueError(
                "'gear_offsets_grasp' parameter is required in set_robot_to_grasp_pose configuration. "
                "It should be a dict with keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )
        gear_offsets_grasp = cfg.params["gear_offsets_grasp"]
        if not isinstance(gear_offsets_grasp, dict):
            raise TypeError(
                f"'gear_offsets_grasp' parameter must be a dict, got {type(gear_offsets_grasp).__name__}. "
                "It should have keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )

        self.gear_grasp_offset_tensors = {}
        for gear_type in ["gear_small", "gear_medium", "gear_large"]:
            if gear_type not in gear_offsets_grasp:
                raise ValueError(
                    f"'{gear_type}' offset is required in 'gear_offsets_grasp' parameter. "
                    f"Found keys: {list(gear_offsets_grasp.keys())}"
                )
            self.gear_grasp_offset_tensors[gear_type] = torch.tensor(
                gear_offsets_grasp[gear_type], device=env.device, dtype=torch.float32
            )

        # Stack grasp offset tensors for vectorized indexing (shape: 3, 3)
        # Index 0=small, 1=medium, 2=large
        self.gear_grasp_offsets_stacked = torch.stack(
            [
                self.gear_grasp_offset_tensors["gear_small"],
                self.gear_grasp_offset_tensors["gear_medium"],
                self.gear_grasp_offset_tensors["gear_large"],
            ],
            dim=0,
        )

        # Pre-cache grasp rotation offset tensor
        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        # Pre-allocate buffers for batch operations
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.local_env_indices = torch.arange(env.num_envs, device=env.device)
        self.gear_grasp_offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

        # Cache hand grasp/close widths
        self.hand_grasp_width = env.cfg.hand_grasp_width
        self.hand_close_width = env.cfg.hand_close_width

        # Find end effector index once
        eef_indices, _ = self.robot_asset.find_bodies([self.end_effector_body_name])
        if len(eef_indices) == 0:
            raise ValueError(f"End effector body '{self.end_effector_body_name}' not found in robot")
        self.eef_idx = eef_indices[0]

        # Find jacobian body index (for fixed-base robots, subtract 1)
        self.jacobi_body_idx = self.eef_idx - 1

        # Find all joints once
        all_joints, all_joints_names = self.robot_asset.find_joints([".*"])
        self.all_joints = all_joints
        self.finger_joints = all_joints[self.num_arm_joints :]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        pos_threshold: float = 1e-6,
        rot_threshold: float = 1e-6,
        max_iterations: int = 50,
        pos_randomization_range: dict | None = None,
        gear_offsets_grasp: dict | None = None,
        end_effector_body_name: str | None = None,
        num_arm_joints: int | None = None,
        grasp_rot_offset: list | None = None,
        gripper_joint_setter_func: callable | None = None,
    ):
        """Set robot to grasp pose using IK.

        Args:
            env: Environment instance
            env_ids: Environment IDs to reset
            robot_asset_cfg: Robot asset configuration (unused, kept for compatibility)
            pos_threshold: Position convergence threshold
            rot_threshold: Rotation convergence threshold
            max_iterations: Maximum IK iterations
            pos_randomization_range: Optional position randomization range
        """
        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this event term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager

        # Slice buffers for current batch size
        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]
        local_env_indices = self.local_env_indices[:num_reset_envs]
        gear_grasp_offsets = self.gear_grasp_offsets_buffer[:num_reset_envs]
        grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]

        # IK loop
        for i in range(max_iterations):
            # Get current joint state
            joint_pos = self.robot_asset.data.joint_pos.torch[env_ids].clone()
            joint_vel = self.robot_asset.data.joint_vel.torch[env_ids].clone()

            # Stack all gear positions and quaternions
            all_gear_pos = torch.stack(
                [
                    env.scene["factory_gear_small"].data.root_link_pos_w.torch,
                    env.scene["factory_gear_medium"].data.root_link_pos_w.torch,
                    env.scene["factory_gear_large"].data.root_link_pos_w.torch,
                ],
                dim=1,
            )[env_ids]

            all_gear_quat = torch.stack(
                [
                    env.scene["factory_gear_small"].data.root_link_quat_w.torch,
                    env.scene["factory_gear_medium"].data.root_link_quat_w.torch,
                    env.scene["factory_gear_large"].data.root_link_quat_w.torch,
                ],
                dim=1,
            )[env_ids]

            # Get gear type indices directly as tensor
            all_gear_type_indices = gear_type_manager.get_all_gear_type_indices()
            gear_type_indices[:] = all_gear_type_indices[env_ids]

            # Select gear data using advanced indexing
            grasp_object_pos_world = all_gear_pos[local_env_indices, gear_type_indices]
            grasp_object_quat = all_gear_quat[local_env_indices, gear_type_indices]

            # Apply rotation offset
            grasp_object_quat = math_utils.quat_mul(grasp_object_quat, grasp_rot_offset_tensor)

            # Get grasp offsets (vectorized)
            gear_grasp_offsets[:] = self.gear_grasp_offsets_stacked[gear_type_indices]

            # Add position randomization if specified
            if pos_randomization_range is not None:
                pos_keys = ["x", "y", "z"]
                range_list_pos = [pos_randomization_range.get(key, (0.0, 0.0)) for key in pos_keys]
                ranges_pos = torch.tensor(range_list_pos, device=env.device)
                rand_pos_offsets = math_utils.sample_uniform(
                    ranges_pos[:, 0], ranges_pos[:, 1], (len(env_ids), 3), device=env.device
                )
                gear_grasp_offsets = gear_grasp_offsets + rand_pos_offsets

            # Transform offsets from gear frame to world frame
            grasp_object_pos_world = grasp_object_pos_world + math_utils.quat_apply(
                grasp_object_quat, gear_grasp_offsets
            )

            # Get end effector pose
            eef_pos = self.robot_asset.data.body_pos_w.torch[env_ids, self.eef_idx]
            eef_quat = self.robot_asset.data.body_quat_w.torch[env_ids, self.eef_idx]

            # Compute pose error
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=eef_pos,
                fingertip_midpoint_quat=eef_quat,
                ctrl_target_fingertip_midpoint_pos=grasp_object_pos_world,
                ctrl_target_fingertip_midpoint_quat=grasp_object_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Check convergence
            pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
            rot_error_norm = torch.linalg.norm(axis_angle_error, dim=-1)

            if torch.all(pos_error_norm < pos_threshold) and torch.all(rot_error_norm < rot_threshold):
                break

            # Solve IK using jacobian. ``body_link_jacobian_w`` prepends ``num_base_dofs``
            # floating-base columns on the DoF axis (0 for fixed-base, 6 for floating-base);
            # slice past them so the column axis aligns with the actuated-joint state.
            jacobians = self.robot_asset.data.body_link_jacobian_w.torch.clone()
            jacobian = jacobians[env_ids, self.jacobi_body_idx, :, self.robot_asset.num_base_dofs :]

            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=jacobian,
                device=env.device,
            )

            # Update joint positions
            joint_pos = joint_pos + delta_dof_pos

            # Wrap arm joint positions to fall within robot's actual joint limits
            joint_pos_limits = self.robot_asset.data.joint_pos_limits.torch[env_ids, : self.num_arm_joints, :]
            joint_min = joint_pos_limits[:, :, 0]
            joint_max = joint_pos_limits[:, :, 1]
            joint_range = joint_max - joint_min

            # Wrap only the arm joint positions (not gripper joints)
            arm_joint_pos = joint_pos[:, : self.num_arm_joints]
            arm_joint_pos = torch.where(
                joint_range > 0,
                joint_min + torch.remainder(arm_joint_pos - joint_min, joint_range),
                arm_joint_pos,
            )
            joint_pos[:, : self.num_arm_joints] = arm_joint_pos

            joint_vel = torch.zeros_like(joint_pos)

            # Write to sim
            self.robot_asset.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
            self.robot_asset.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
            self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Reset joint velocities to zero after IK convergence
        joint_vel = torch.zeros_like(self.robot_asset.data.joint_vel.torch[env_ids])

        # Set gripper to grasp position
        joint_pos = self.robot_asset.data.joint_pos.torch[env_ids].clone()

        # Get gear types for all environments
        all_gear_types = gear_type_manager.get_all_gear_types()
        for row_idx, env_id in enumerate(env_ids.tolist()):
            gear_key = all_gear_types[env_id]
            hand_grasp_width = self.hand_grasp_width[gear_key]
            self.gripper_joint_setter_func(joint_pos, [row_idx], self.finger_joints, hand_grasp_width)

        self.robot_asset.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)
        self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Set gripper to closed position
        for row_idx, env_id in enumerate(env_ids.tolist()):
            gear_key = all_gear_types[env_id]
            hand_close_width = self.hand_close_width[gear_key]
            self.gripper_joint_setter_func(joint_pos, [row_idx], self.finger_joints, hand_close_width)

        self.robot_asset.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)


class fixed_joint_selected_gear_to_gripper(ManagerTermBase):
    """Attach the active gear to the gripper with a PhysX fixed joint.

    This term authors one fixed joint per gear per environment on first reset,
    updates the joint frames from the current reset poses, and enables only the
    joint for the currently selected gear. It keeps the gear physically dynamic
    and constrained to the gripper instead of overwriting the gear pose every step.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset: Articulation = env.scene[self.robot_asset_cfg.name]

        if "end_effector_body_name" not in cfg.params:
            raise ValueError("'end_effector_body_name' parameter is required for fixed_joint_selected_gear_to_gripper.")
        if "grasp_rot_offset" not in cfg.params:
            raise ValueError("'grasp_rot_offset' parameter is required for fixed_joint_selected_gear_to_gripper.")
        if "gear_offsets_grasp" not in cfg.params:
            raise ValueError("'gear_offsets_grasp' parameter is required for fixed_joint_selected_gear_to_gripper.")

        self.end_effector_body_name = cfg.params["end_effector_body_name"]
        self.gear_asset_names = ["factory_gear_small", "factory_gear_medium", "factory_gear_large"]
        self.gear_keys = ["gear_small", "gear_medium", "gear_large"]

        gear_offsets_grasp = cfg.params["gear_offsets_grasp"]
        self.gear_grasp_offsets_stacked = torch.stack(
            [
                torch.tensor(gear_offsets_grasp[gear_key], device=env.device, dtype=torch.float32)
                for gear_key in self.gear_keys
            ],
            dim=0,
        )
        self.grasp_rot_offset = torch.tensor(cfg.params["grasp_rot_offset"], device=env.device, dtype=torch.float32)

        eef_indices, _ = self.robot_asset.find_bodies([self.end_effector_body_name])
        if len(eef_indices) == 0:
            raise ValueError(f"End effector body '{self.end_effector_body_name}' not found in robot")
        self.eef_idx = eef_indices[0]

        self._joints_authored = False
        self._joint_paths: dict[tuple[int, str], str] = {}
        self._gear_type_to_index = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gear_offsets_grasp: dict | None = None,
        end_effector_body_name: str | None = None,
        grasp_rot_offset: list | None = None,
        operation: str = "select",
    ):
        """Author/select fixed joints and log attach errors.

        Args:
            env: Environment instance.
            env_ids: Environment ids affected by this event.
            operation: ``"select"`` enables only the active gear's joint for reset
                envs, while ``"log"`` only logs the fixed-grasp tracking error.
        """
        if operation not in ("author", "select", "log"):
            raise ValueError(f"Unsupported fixed grasp operation: {operation}")

        if operation == "author":
            self._author_fixed_joints(env)
            return

        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "before fixed_joint_selected_gear_to_gripper is used."
            )

        env_ids = _resolve_env_ids(env, env_ids)
        if not self._joints_authored:
            self._author_fixed_joints(env)

        if operation == "select":
            self._select_fixed_joints(env, env_ids)

        self._log_attach_error_metrics(env, env_ids)

    def _author_fixed_joints(self, env: ManagerBasedEnv) -> None:
        """Create one disabled fixed joint for each gear in each environment."""
        from pxr import Gf, UsdPhysics

        existing_joint_paths = getattr(env, "_fixed_grasp_joint_paths", None)
        if existing_joint_paths is not None:
            self._joint_paths = existing_joint_paths
            self._joints_authored = True
            return

        stage = env.scene.stage
        max_float = 3.4028234663852886e38

        for env_id, env_prim_path in enumerate(env.scene.env_prim_paths):
            eef_prim = self._find_rigid_body_prim(stage, f"{env_prim_path}/Robot", self.end_effector_body_name)
            for gear_key in self.gear_keys:
                gear_prim = self._find_rigid_body_prim(stage, f"{env_prim_path}/{self._gear_prim_name(gear_key)}")
                local_pos_0, local_rot_0 = self._current_joint_frame(env, env_id, gear_key)
                joint_path = f"{env_prim_path}/FixedGraspJoint_{gear_key}"
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateLocalPos0Attr().Set(local_pos_0)
                joint.CreateLocalRot0Attr().Set(local_rot_0)
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
                joint.CreateBreakForceAttr().Set(max_float)
                joint.CreateBreakTorqueAttr().Set(max_float)
                joint.CreateJointEnabledAttr().Set(False)
                joint.CreateBody0Rel().SetTargets([eef_prim.GetPath()])
                joint.CreateBody1Rel().SetTargets([gear_prim.GetPath()])
                self._joint_paths[(env_id, gear_key)] = joint_path

        env._fixed_grasp_joint_paths = self._joint_paths
        self._joints_authored = True

    def _select_fixed_joints(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        """Enable the selected gear joint and disable inactive gear joints."""
        from pxr import UsdPhysics

        stage = env.scene.stage
        gear_type_indices = env._gear_type_manager.get_all_gear_type_indices()
        for env_id in env_ids.tolist():
            selected_idx = int(gear_type_indices[env_id].item())
            for gear_key in self.gear_keys:
                joint_path = self._joint_paths[(env_id, gear_key)]
                joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
                self._set_joint_frame_from_current_pose(env, env_id, gear_key, joint)
                joint.GetJointEnabledAttr().Set(self._gear_type_to_index[gear_key] == selected_idx)

    def _set_joint_frame_from_current_pose(self, env: ManagerBasedEnv, env_id: int, gear_key: str, joint) -> None:
        """Set the joint's body-0 frame so it matches the current gear pose."""
        from pxr import Gf

        local_pos_0, local_rot_0 = self._current_joint_frame(env, env_id, gear_key)
        joint.GetLocalPos0Attr().Set(local_pos_0)
        joint.GetLocalRot0Attr().Set(local_rot_0)
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))

    def _current_joint_frame(self, env: ManagerBasedEnv, env_id: int, gear_key: str):
        """Return the gear root frame expressed in the end-effector body frame."""
        from pxr import Gf

        gear_asset = env.scene[self.gear_asset_names[self._gear_type_to_index[gear_key]]]
        eef_pos = self.robot_asset.data.body_link_pos_w.torch[env_id, self.eef_idx].unsqueeze(0)
        eef_quat = self.robot_asset.data.body_link_quat_w.torch[env_id, self.eef_idx].unsqueeze(0)
        gear_pos = gear_asset.data.root_link_pos_w.torch[env_id].unsqueeze(0)
        gear_quat = gear_asset.data.root_link_quat_w.torch[env_id].unsqueeze(0)

        local_pos_0 = math_utils.quat_apply_inverse(eef_quat, gear_pos - eef_pos)[0]
        local_rot_0 = math_utils.quat_mul(math_utils.quat_inv(eef_quat), gear_quat)[0]
        return Gf.Vec3f(*local_pos_0.detach().cpu().tolist()), self._gf_quat_from_xyzw(local_rot_0)

    def _log_attach_error_metrics(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        """Log how well the fixed-joint grasp relation is being maintained."""
        gear_type_indices = env._gear_type_manager.get_all_gear_type_indices()[env_ids]
        all_gear_pos = torch.stack(
            [env.scene[asset_name].data.root_link_pos_w.torch[env_ids] for asset_name in self.gear_asset_names],
            dim=1,
        )
        all_gear_quat = torch.stack(
            [env.scene[asset_name].data.root_link_quat_w.torch[env_ids] for asset_name in self.gear_asset_names],
            dim=1,
        )

        row_ids = torch.arange(len(env_ids), device=env.device)
        gear_pos = all_gear_pos[row_ids, gear_type_indices]
        gear_quat = all_gear_quat[row_ids, gear_type_indices]
        grasp_offsets = self.gear_grasp_offsets_stacked[gear_type_indices]

        expected_eef_quat = math_utils.quat_mul(gear_quat, self.grasp_rot_offset.unsqueeze(0).expand_as(gear_quat))
        expected_eef_pos = gear_pos + math_utils.quat_apply(expected_eef_quat, grasp_offsets)
        eef_pos = self.robot_asset.data.body_link_pos_w.torch[env_ids, self.eef_idx]
        eef_quat = self.robot_asset.data.body_link_quat_w.torch[env_ids, self.eef_idx]

        pos_error = torch.linalg.norm(expected_eef_pos - eef_pos, dim=-1)
        quat_error = math_utils.quat_mul(eef_quat, math_utils.quat_inv(expected_eef_quat))
        rot_error = torch.linalg.norm(math_utils.axis_angle_from_quat(quat_error), dim=-1)

        if not hasattr(env, "extras"):
            env.extras = {}
        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"]["fixed_grasp/attach_pos_error_mean_m"] = pos_error.mean().item()
        env.extras["log"]["fixed_grasp/attach_pos_error_max_m"] = pos_error.max().item()
        env.extras["log"]["fixed_grasp/attach_rot_error_mean_rad"] = rot_error.mean().item()
        env.extras["log"]["fixed_grasp/attach_rot_error_max_rad"] = rot_error.max().item()

    @staticmethod
    def _gear_prim_name(gear_key: str) -> str:
        return {
            "gear_small": "FactoryGearSmall",
            "gear_medium": "FactoryGearMedium",
            "gear_large": "FactoryGearLarge",
        }[gear_key]

    @staticmethod
    def _gf_quat_from_xyzw(quat_xyzw: torch.Tensor):
        from pxr import Gf

        quat = quat_xyzw.detach().cpu().tolist()
        return Gf.Quatf(float(quat[3]), Gf.Vec3f(float(quat[0]), float(quat[1]), float(quat[2])))

    @staticmethod
    def _find_rigid_body_prim(stage, root_path: str, body_name: str | None = None):
        from pxr import Usd, UsdPhysics

        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise ValueError(f"Could not find prim at path '{root_path}'")

        if body_name is not None:
            for prim in Usd.PrimRange(root_prim):
                if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return prim

        if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return root_prim

        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim

        label = f" named '{body_name}'" if body_name else ""
        raise ValueError(f"Could not find a rigid body prim{label} under '{root_path}'")


class randomize_gears_and_base_pose(ManagerTermBase):
    """Randomize both the gear base pose and individual gear poses.

    This class-based term pre-caches all tensors needed for randomization.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomize gears and base pose term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        # Cache asset names
        self.gear_asset_names = ["factory_gear_small", "factory_gear_medium", "factory_gear_large"]
        self.base_asset_name = "factory_gear_base"

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict = {},
        velocity_range: dict = {},
        gear_pos_range: dict = {},
    ):
        """Randomize gear base and gear poses.

        Args:
            env: Environment instance
            env_ids: Environment IDs to randomize
            pose_range: Pose randomization range for base and all gears
            velocity_range: Velocity randomization range
            gear_pos_range: Additional position randomization for selected gear only
        """
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this event term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        device = env.device

        # Shared pose samples for all assets
        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        range_list_pose = [pose_range.get(key, (0.0, 0.0)) for key in pose_keys]
        ranges_pose = torch.tensor(range_list_pose, device=device)
        rand_pose_samples = math_utils.sample_uniform(
            ranges_pose[:, 0], ranges_pose[:, 1], (len(env_ids), 6), device=device
        )

        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
        )

        # Shared velocity samples
        range_list_vel = [velocity_range.get(key, (0.0, 0.0)) for key in pose_keys]
        ranges_vel = torch.tensor(range_list_vel, device=device)
        rand_vel_samples = math_utils.sample_uniform(
            ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=device
        )

        # Prepare poses for all assets
        positions_by_asset = {}
        orientations_by_asset = {}
        velocities_by_asset = {}

        asset_names_to_process = [self.base_asset_name] + self.gear_asset_names
        for asset_name in asset_names_to_process:
            asset: RigidObject | Articulation = env.scene[asset_name]
            default_root_pose = asset.data.default_root_pose.torch[env_ids].clone()
            default_root_vel = asset.data.default_root_vel.torch[env_ids].clone()
            positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]
            orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)
            velocities = default_root_vel + rand_vel_samples
            positions_by_asset[asset_name] = positions
            orientations_by_asset[asset_name] = orientations
            velocities_by_asset[asset_name] = velocities

        # Per-env gear offset (gear_pos_range) applied only to selected gear
        range_list_gear = [gear_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges_gear = torch.tensor(range_list_gear, device=device)
        rand_gear_offsets = math_utils.sample_uniform(
            ranges_gear[:, 0], ranges_gear[:, 1], (len(env_ids), 3), device=device
        )

        # Get gear type indices directly as tensor
        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]
        all_gear_type_indices = gear_type_manager.get_all_gear_type_indices()
        gear_type_indices[:] = all_gear_type_indices[env_ids]

        # Apply offsets using vectorized operations with masks
        for gear_idx, asset_name in enumerate(self.gear_asset_names):
            if asset_name in positions_by_asset:
                mask = gear_type_indices == gear_idx
                positions_by_asset[asset_name][mask] = positions_by_asset[asset_name][mask] + rand_gear_offsets[mask]

        # Write to sim
        for asset_name in positions_by_asset.keys():
            asset = env.scene[asset_name]
            positions = positions_by_asset[asset_name]
            orientations = orientations_by_asset[asset_name]
            velocities = velocities_by_asset[asset_name]
            asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
