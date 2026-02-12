# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion.cable_insertion_env_cfg_flexiv import (
    CableInsertionEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.flexiv import RIZON4S_GRAV_CFG  # isort: skip


##
# Gripper-specific helper functions
##


def set_finger_joint_pos_grav(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Grav gripper.

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of all gripper joint indices (6 joints total)
        finger_joint_position: Target position for main finger joint

    Note:
        Grav gripper joint structure (indices from finger_joints list):
        [0] finger_joint - main controllable joint
        [1] left_inner_knuckle_joint - mimic with -1 gearing
        [2] right_inner_knuckle_joint - mimic with -1 gearing
        [3] right_outer_knuckle_joint - mimic with -1 gearing
        [4] left_outer_finger_joint - mimic with +1 gearing
        [5] right_outer_finger_joint - mimic with +1 gearing
    """
    for idx in reset_ind_joint_pos:
        if len(finger_joints) < 6:
            raise ValueError(f"Grav gripper requires at least 6 finger joints, got {len(finger_joints)}")

        # Main controllable joint
        joint_pos[idx, finger_joints[0]] = finger_joint_position

        # Mimic joints with -1 gearing
        joint_pos[idx, finger_joints[1]] = finger_joint_position  # left_inner_knuckle_joint
        joint_pos[idx, finger_joints[2]] = finger_joint_position  # right_inner_knuckle_joint
        joint_pos[idx, finger_joints[3]] = finger_joint_position  # right_outer_knuckle_joint

        # Mimic joints with +1 gearing
        joint_pos[idx, finger_joints[4]] = -finger_joint_position  # left_outer_finger_joint
        joint_pos[idx, finger_joints[5]] = -finger_joint_position  # right_outer_finger_joint


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events."""

    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("gb300_plug", body_names=".*"),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    socket_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("gb300_socket", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),  # Matches finger_mount and finger_tip bodies
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_plug_and_socket_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # "x": [-0.05, 0.05],
                # "y": [-0.05, 0.05],
                # "z": [0.1, 0.15],
                # "roll": [-math.pi / 90, math.pi / 90],  # 2 degrees
                # "pitch": [-math.pi / 90, math.pi / 90],  # 2 degrees
                # "yaw": [-math.pi / 6, math.pi / 6],  # 30 degrees
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [-0.1, -0.05],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("gb300_socket"),
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=mdp.set_robot_to_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "target_object_name": "gb300_plug",
            "grasp_offset": [0.0, 0.0, 0.0],  # Will be updated in __post_init__
        },
    )


@configclass
class TerminationsCfg:
    """Configuration for termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    plug_dropped = DoneTerm(
        func=mdp.reset_when_plug_dropped,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("gb300_plug"),
            "distance_threshold": 0.1,  # Terminate if end effector is more than 10cm from expected grasp position
            "end_effector_body_name": "link7",  # Will be populated from config
            "grasp_offset": [-0.531, -0.025, -0.383],  # Will be populated from config
            "grasp_rot_offset": [-0.70711, 0.70711, 0.0, 0.0],  # Will be populated from config
        },
    )

    plug_orientation_error = DoneTerm(
        func=mdp.reset_when_plug_orientation_exceeds_threshold,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("gb300_plug"),
            "roll_threshold_deg": 30.0,  # Terminate if roll error exceeds 30 degrees
            "pitch_threshold_deg": 30.0,  # Terminate if pitch error exceeds 30 degrees
            "yaw_threshold_deg": 180.0,  # Terminate if yaw error exceeds 180 degrees
            "end_effector_body_name": "link7",  # Will be populated from config
            "grasp_rot_offset": [-0.70711, 0.70711, 0.0, 0.0],  # Will be populated from config
        },
    )


@configclass
class Rizon4sCableInsertionEnvCfg(CableInsertionEnvCfg):
    """Base configuration for Rizon 4s Cable Insertion Environment.

    This class contains common setup for Rizon 4s robot with Grav gripper.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Robot-specific parameters
        self.end_effector_body_name = "link7"  # End effector body name for IK
        self.num_arm_joints = 7  # Number of arm joints (excluding gripper)
        # self.grasp_offset = [0.03, 0.0023, -0.328]
        self.grasp_offset = [-0.531, -0.025, -0.39]
        self.grasp_rot_offset = [
            -0.70711,
            0.70711,
            0.0,
            0.0,
        ]  # Rotation offset for grasp pose (quaternion [x,y,z,w])
        self.gripper_joint_setter_func = set_finger_joint_pos_grav  # Gripper-specific joint setter function

        # Plug parameters (from GB300Plug config class)
        self.plug_diameter = 0.009
        self.plug_height = 0.050
        self.plug_mass = 0.019
        self.plug_grasp_rotation_pos = [0.03, 0.0023, -0.008 / 1.5]
        self.plug_grasp_rotation_deg = [180.0, 0.0, 0.0]

        # Socket parameters (from GB300Socket config class)
        self.socket_diameter = 0.0081
        self.socket_height = 0.025
        self.socket_base_height = 0.0

        # Common observation configuration
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # override events
        self.events = EventCfg()

        # override terminations
        self.terminations = TerminationsCfg()

        # Populate termination term parameters
        self.terminations.plug_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_dropped.params["grasp_offset"] = self.grasp_offset
        self.terminations.plug_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.plug_orientation_error.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_orientation_error.params["grasp_rot_offset"] = self.grasp_rot_offset

        # override command generator body
        self.joint_action_scale = 0.01
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )


@configclass
class Rizon4sGravCableInsertionEnvCfg(Rizon4sCableInsertionEnvCfg):
    """Configuration for Rizon 4s with Grav gripper for cable insertion."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Rizon 4s with Grav gripper
        self.scene.robot = RIZON4S_GRAV_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=RIZON4S_GRAV_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            # Joint positions based on IK
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint1": 0.0,
                    "joint2": -0.698,
                    "joint3": 0.0,
                    "joint4": 1.571,
                    "joint5": 0.0,
                    "joint6": 0.698,
                    "joint7": 0.0,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # Grav gripper actuator configuration override
        self.scene.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],  # Only main controllable joint
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=2e3,
            damping=1e1,
            friction=0.0,
            armature=0.0,
        )

        # # Override plug and socket initial states for Rizon
        # # Socket (fixed position)
        # self.scene.gb300_socket.init_state = RigidObjectCfg.InitialStateCfg(
        #     pos=(-0.6, 0.0, 0.0),
        #     rot=(0.0, 0.0, 0.70711, 0.70711),
        # )

        # # Plug (will be randomized)
        # self.scene.gb300_plug.init_state = RigidObjectCfg.InitialStateCfg(
        #     pos=(-0.6, 0.0, 0.1),
        #     rot=(0.0, 0.0, 0.70711, 0.70711),
        # )

        # Grasp width for Grav gripper (may need adjustment)
        self.hand_grasp_width = -0.065  # Open width for grasping plug
        self.hand_close_width = -0.15  # Closed width

        # Populate event term parameters
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func


@configclass
class Rizon4sGravCableInsertionEnvCfg_PLAY(Rizon4sGravCableInsertionEnvCfg):
    """Play configuration for Rizon 4s with Grav gripper for cable insertion."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
