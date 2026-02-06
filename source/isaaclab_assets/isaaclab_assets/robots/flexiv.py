# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Flexiv Robots.

The following configuration parameters are available:

* :obj:`RIZON4S_CFG`: The Rizon 4s arm without a gripper.
* :obj:`RIZON4S_GRAV_CFG`: The Rizon 4s arm with Grav gripper.

Reference: https://github.com/flexivrobotics/isaac_sim_ws
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

RIZON4S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_ros_gear_insertion/Rizon4s_with_Grav.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
        activate_contact_sensors=False,
    ),
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
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            stiffness={"joint[1-2]": 6000.0, "joint[3-4]": 4200.0, "joint[5-7]": 1500.0},
            damping={"joint[1-2]": 108.5, "joint[3-4]": 90.7, "joint[5-7]": 54.2},
            friction=0.0,
            armature=0.0,
        ),
    },
)

"""Configuration of Rizon 4s arm using implicit actuator models."""

RIZON4S_GRAV_CFG = RIZON4S_CFG.copy()
"""Configuration of Rizon 4s arm with Grav gripper."""
# Note: The USD file should include the Grav gripper variant
# Gripper joints: finger_joint, left_outer_finger_joint, right_outer_finger_joint
RIZON4S_GRAV_CFG.spawn.rigid_props.disable_gravity = True
RIZON4S_GRAV_CFG.init_state.joint_pos["finger_joint"] = 0.0
RIZON4S_GRAV_CFG.init_state.joint_pos["left_outer_finger_joint"] = 0.0
RIZON4S_GRAV_CFG.init_state.joint_pos["right_outer_finger_joint"] = 0.0

# Grav gripper actuator configuration
RIZON4S_GRAV_CFG.actuators["gripper"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],  # Only main controllable joint; mimics with -1 gearing follow automatically
    effort_limit_sim=200.0,
    velocity_limit_sim=0.6,
    stiffness=2e3,
    damping=1e1,
    friction=0.0,
    armature=0.0,
)
