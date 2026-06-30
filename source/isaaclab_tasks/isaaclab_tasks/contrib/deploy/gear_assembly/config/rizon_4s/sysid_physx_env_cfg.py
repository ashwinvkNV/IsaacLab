# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.deploy.mdp.delayed_joint_actions_cfg import ShapedDelayedRelativeJointPositionActionCfg

from .ros_inference_env_cfg import Rizon4sGearAssemblyROSInferenceEnvCfg

ISAACLAB_ROOT = Path(__file__).resolve().parents[8]
FLEXIV_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
FLEXIV_ACTION_LATENCY_MS = 20.0
FLEXIV_PHYSX_SYSID_ACTUATOR_YAML = "flexiv/manual/flexiv_pd_only_gravityoff_high_cmdlimits_tuned_physx.yaml"
FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH = f"input/actuator_models/{FLEXIV_PHYSX_SYSID_ACTUATOR_YAML}"
FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ = 200.0
FLEXIV_DEPLOYMENT_DECIMATION = 4
FLEXIV_DEPLOYMENT_CONTROL_FREQ_HZ = FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ / FLEXIV_DEPLOYMENT_DECIMATION
FLEXIV_ROBOT_COLLECTION_COMMAND_VELOCITY_LIMIT = 2.0
FLEXIV_ROBOT_COLLECTION_COMMAND_ACCELERATION_LIMIT = 3.0
FLEXIV_GRAV_ROS_INFERENCE_SETUP = {
    "source_env": "Isaac-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference-v0",
    "robot_usd": "rizon4s_with_grav.usd",
    "gravity_compensation": "robot_rigid_bodies_disable_gravity_true",
    "rigid_body": {
        "disable_gravity": True,
        "max_depenetration_velocity": 5.0,
        "linear_damping": 0.0,
        "angular_damping": 0.0,
        "max_linear_velocity": 1000.0,
        "max_angular_velocity": 3666.0,
        "enable_gyroscopic_forces": True,
        "solver_position_iteration_count": 4,
        "solver_velocity_iteration_count": 1,
        "max_contact_impulse": 1e32,
    },
    "articulation": {
        "enabled_self_collisions": False,
        "solver_position_iteration_count": 4,
        "solver_velocity_iteration_count": 1,
    },
    "collision": {"contact_offset": 0.005, "rest_offset": 0.0},
    "arm_actuator_yaml": FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH,
    "action_latency_ms": FLEXIV_ACTION_LATENCY_MS,
}


def _load_implicit_actuator_cfg(actuator_yaml: str):
    if str(ISAACLAB_ROOT) not in sys.path:
        sys.path.append(str(ISAACLAB_ROOT))
    from input.actuator_models import load_implicit_actuator_cfg

    return load_implicit_actuator_cfg(actuator_yaml, FLEXIV_ARM_JOINT_NAMES)


def _replace_arm_actuator(robot_cfg, actuator_yaml: str) -> None:
    """Install a single tuned arm actuator while preserving gripper actuators."""
    gripper_actuators = {
        actuator_name: actuator_cfg
        for actuator_name, actuator_cfg in robot_cfg.actuators.items()
        if actuator_name.startswith("gripper")
    }
    robot_cfg.actuators = {
        "arm": _load_implicit_actuator_cfg(actuator_yaml),
        **gripper_actuators,
    }


@configclass
class Rizon4sGearAssemblyROSInferenceSysIDPhysXShapedEnvCfg(Rizon4sGearAssemblyROSInferenceEnvCfg):
    """ROS-inference env with final PhysX SysID actuator params and 50 Hz command shaping."""

    def __post_init__(self):
        super().__post_init__()
        _replace_arm_actuator(self.scene.robot, FLEXIV_PHYSX_SYSID_ACTUATOR_YAML)
        self.decimation = FLEXIV_DEPLOYMENT_DECIMATION
        self.sim.dt = 1.0 / FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ
        self.sim.render_interval = self.decimation
        self.flexiv_tuned_actuator_yaml = FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH
        self.flexiv_action_latency_ms = FLEXIV_ACTION_LATENCY_MS
        self.sim_to_real_tuned_config = dict(FLEXIV_GRAV_ROS_INFERENCE_SETUP)
        self.sim_to_real_tuned_config.update(
            {
                "active_env": "Isaac-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference-SysIDPhysX-Shaped-v0",
                "controller": "ShapedDelayedRelativeJointPositionActionCfg",
                "arm_actuator_yaml": FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH,
                "sysid_source": (
                    "output/sysid/flexiv/"
                    "pd_only_gravityoff_high_cmdlimits_seed_default_steps_pm10_physx_n64_g20/best_params.yaml"
                ),
                "physics_freq_hz": FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ,
                "decimation": FLEXIV_DEPLOYMENT_DECIMATION,
                "control_freq_hz": FLEXIV_DEPLOYMENT_CONTROL_FREQ_HZ,
                "command_velocity_limit_rad_s": FLEXIV_ROBOT_COLLECTION_COMMAND_VELOCITY_LIMIT,
                "command_acceleration_limit_rad_s2": FLEXIV_ROBOT_COLLECTION_COMMAND_ACCELERATION_LIMIT,
                "sysid_notes": (
                    "PhysX PD-only step-response SysID plus command shaping matched to "
                    "the 50 Hz Flexiv deployment command loop."
                ),
            }
        )
        self.actions.arm_action = ShapedDelayedRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=FLEXIV_ARM_JOINT_NAMES,
            scale=self.joint_action_scale,
            use_zero_offset=True,
            latency_s=FLEXIV_ACTION_LATENCY_MS / 1000.0,
            command_velocity_limit=FLEXIV_ROBOT_COLLECTION_COMMAND_VELOCITY_LIMIT,
            command_acceleration_limit=FLEXIV_ROBOT_COLLECTION_COMMAND_ACCELERATION_LIMIT,
        )
