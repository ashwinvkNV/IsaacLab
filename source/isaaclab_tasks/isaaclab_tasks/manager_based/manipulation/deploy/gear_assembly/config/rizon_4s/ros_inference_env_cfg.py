# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .joint_pos_env_cfg import Rizon4sGearAssemblyEnvCfg


@configclass
class Rizon4sGearAssemblyROSInferenceEnvCfg(Rizon4sGearAssemblyEnvCfg):
    """ROS / Isaac Manipulator inference fields plus deployment alignment for NVIDIA Hubble Lab.

    This configuration:
    - Exposes variables needed for ROS inference
    - Overrides robot and gear initial poses for fixed/deterministic setup
    - Aligns robot mounting pose with the Flexiv Rizon 4s installation at NVIDIA Hubble Lab
    """

    # Single source for base + all gear rigid bodies (Rizon: closer to robot, centered)
    ros_inference_factory_gears_init_state: RigidObjectCfg.InitialStateCfg = RigidObjectCfg.InitialStateCfg(
        pos=(0.75, 0.0, -0.2),
        rot=(0.0, 0.0, 0.70711, -0.70711),
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Variables used by Isaac Manipulator for on robot inference
        # These parameters allow the ROS inference node to validate environment configuration,
        # perform checks during inference, and correctly interpret observations and actions.
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "shaft_pos", "shaft_quat"]
        self.policy_action_space = "joint"
        # Use inherited joint names from parent's observation configuration
        self.arm_joint_names = self.observations.policy.joint_pos.params["asset_cfg"].joint_names
        # Use inherited num_arm_joints from parent
        self.action_space = self.num_arm_joints
        # State space and observation space for Rizon 4s with Grav gripper (7 DOF arm + 1 gripper)
        # State: 7 joint pos + 7 joint vel + 3 shaft pos + 4 shaft quat + 3 gear pos + 4 gear quat = 28
        # For critic: additional gear observations
        self.state_space = 28
        # Observation: 7 joint pos + 7 joint vel + 3 shaft pos + 4 shaft quat = 21
        self.observation_space = 21

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        # Dynamically generate action_scale_joint_space based on action_space
        self.action_scale_joint_space = [self.joint_action_scale] * self.action_space

        # --- NVIDIA Hubble Lab: Flexiv Rizon 4s mount ---
        # Remove vertical mount stand since Hubble deployment does not use the sim stand asset
        self.scene.stand = None

        # Lab home joint pose (radians); aligns sim defaults / reset with the physical stand
        self.scene.robot.init_state.joint_pos = {
            "joint1": math.radians(-90.0),
            "joint2": math.radians(90.0),
            "joint3": 0.0,
            "joint4": math.radians(90.0),
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        }

        # Orientation of robot is based on the Flexiv Rizon 4s mount in the Hubble Lab
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.rot = (0.5, 0.5, 0.5, 0.5)

        # Override gear base + all gears from one template (fixed pose for ROS inference)
        _g = self.ros_inference_factory_gears_init_state
        for _name in (
            "factory_gear_base",
            "factory_gear_small",
            "factory_gear_medium",
            "factory_gear_large",
        ):
            getattr(self.scene, _name).init_state = RigidObjectCfg.InitialStateCfg(
                pos=_g.pos,
                rot=_g.rot,
                lin_vel=_g.lin_vel,
                ang_vel=_g.ang_vel,
            )

        # Fixed asset parameters for ROS inference - derived from configuration
        # These parameters are used by the ROS inference node to validate the environment setup
        # and apply appropriate noise models for robust real-world deployment.
        # Derive position center from gear base init state
        self.fixed_asset_init_pos_center = list(self.scene.factory_gear_base.init_state.pos)
        # Derive position range from parent's randomize_gears_and_base_pose event pose_range
        pose_range = self.events.randomize_gears_and_base_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],  # max value
            pose_range["y"][1],  # max value
            pose_range["z"][1],  # max value
        ]
        # Orientation in degrees (quaternion (0.0, 0.0, 0.70711, -0.70711) = -90° around Z)
        self.fixed_asset_init_orn_deg = [0.0, 0.0, -90.0]
        # Derive orientation range from parent's pose_range (radians to degrees)
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range["roll"][1]),  # convert radians to degrees
            math.degrees(pose_range["pitch"][1]),
            math.degrees(pose_range["yaw"][1]),
        ]
        # Derive observation noise level from parent's gear_shaft_pos noise configuration
        gear_shaft_pos_noise = self.observations.policy.gear_shaft_pos.noise.noise_cfg.n_max
        self.fixed_asset_pos_obs_noise_level = [
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
        ]
