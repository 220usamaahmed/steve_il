"""Canonical link / joint names of the minimal Steve robot.

Single source of truth for tests, the import check and (later) controllers and
environment configs. Pure Python on purpose: importable without Isaac Sim.

Always address joints by *name*: Isaac orders articulation DOFs by USD tree traversal,
not by URDF order.
"""

BASE_LINK = "base_link"
EE_LINK = "ur5e_tool0"  # tool frame of the UR5e (gripper mounts here)
GRIPPER_BASE_LINK = "robotiq_arg2f_base_link"

# --- actuated / mimic joints ------------------------------------------------------------
ARM_JOINTS = (
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
)

GRIPPER_DRIVEN_JOINT = "finger_joint"  # 0 (open) .. 0.8 rad (closed)
# joint -> multiplier w.r.t. finger_joint (URDF <mimic>, offset 0)
GRIPPER_MIMIC_JOINTS = {
    "left_inner_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_inner_finger_joint": -1.0,
}
GRIPPER_JOINTS = (GRIPPER_DRIVEN_JOINT, *GRIPPER_MIMIC_JOINTS)

# order: front_left, front_right, back_left, back_right (matches the /mmo700/*_cmds topics
# used by the older ROS 2 controller)
CASTER_JOINTS = (
    "mmo_700_caster_front_left_joint",
    "mmo_700_caster_front_right_joint",
    "mmo_700_caster_back_left_joint",
    "mmo_700_caster_back_right_joint",
)
WHEEL_JOINTS = (
    "mmo_700_wheel_front_left_joint",
    "mmo_700_wheel_front_right_joint",
    "mmo_700_wheel_back_left_joint",
    "mmo_700_wheel_back_right_joint",
)
BASE_JOINTS = (*CASTER_JOINTS, *WHEEL_JOINTS)

DYNAMIC_JOINTS = (*BASE_JOINTS, *ARM_JOINTS, *GRIPPER_JOINTS)

# --- full URDF structure: joint -> (type, parent, child) -----------------------------------
JOINTS = {
    # base
    **{
        j: ("continuous", BASE_LINK, j.replace("_joint", "_link")) for j in CASTER_JOINTS
    },
    **{
        w: ("continuous", c.replace("_joint", "_link"), w.replace("_joint", "_link"))
        for c, w in zip(CASTER_JOINTS, WHEEL_JOINTS)
    },
    "cabinet_joint": ("fixed", BASE_LINK, "cabinet_link"),
    # arm
    "ur5e_base_joint": ("fixed", "cabinet_link", "ur5e_base_link"),
    "ur5e_base_link_base_link_inertia": ("fixed", "ur5e_base_link", "ur5e_base_link_inertia"),
    "ur5e_shoulder_pan_joint": ("revolute", "ur5e_base_link_inertia", "ur5e_shoulder_link"),
    "ur5e_shoulder_lift_joint": ("revolute", "ur5e_shoulder_link", "ur5e_upper_arm_link"),
    "ur5e_elbow_joint": ("revolute", "ur5e_upper_arm_link", "ur5e_forearm_link"),
    "ur5e_wrist_1_joint": ("revolute", "ur5e_forearm_link", "ur5e_wrist_1_link"),
    "ur5e_wrist_2_joint": ("revolute", "ur5e_wrist_1_link", "ur5e_wrist_2_link"),
    "ur5e_wrist_3_joint": ("revolute", "ur5e_wrist_2_link", "ur5e_wrist_3_link"),
    "ur5e_wrist_3_link_ft_frame": ("fixed", "ur5e_wrist_3_link", "ur5e_ft_frame"),
    "ur5e_base_link_base_fixed_joint": ("fixed", "ur5e_base_link", "ur5e_base"),
    "ur5e_wrist_3_flange": ("fixed", "ur5e_wrist_3_link", "ur5e_flange"),
    "ur5e_flange_tool0": ("fixed", "ur5e_flange", "ur5e_tool0"),
    # gripper
    "ur5e_tool0_robotiq_coupling_joint": ("fixed", "ur5e_tool0", "robotiq_coupling"),
    "robotiq_coupling_base_joint": ("fixed", "robotiq_coupling", GRIPPER_BASE_LINK),
    "finger_joint": ("revolute", GRIPPER_BASE_LINK, "left_outer_knuckle"),
    "left_outer_finger_joint": ("fixed", "left_outer_knuckle", "left_outer_finger"),
    "left_inner_knuckle_joint": ("revolute", GRIPPER_BASE_LINK, "left_inner_knuckle"),
    "left_inner_finger_joint": ("revolute", "left_outer_finger", "left_inner_finger"),
    "left_inner_finger_pad_joint": ("fixed", "left_inner_finger", "left_inner_finger_pad"),
    "right_outer_knuckle_joint": ("revolute", GRIPPER_BASE_LINK, "right_outer_knuckle"),
    "right_outer_finger_joint": ("fixed", "right_outer_knuckle", "right_outer_finger"),
    "right_inner_knuckle_joint": ("revolute", GRIPPER_BASE_LINK, "right_inner_knuckle"),
    "right_inner_finger_joint": ("revolute", "right_outer_finger", "right_inner_finger"),
    "right_inner_finger_pad_joint": ("fixed", "right_inner_finger", "right_inner_finger_pad"),
}

LINKS = frozenset({BASE_LINK} | {child for _, _, child in JOINTS.values()})

# Joints that PhysX treats as DOFs (everything not fixed)
MOVABLE_JOINTS = frozenset(j for j, (t, _, _) in JOINTS.items() if t != "fixed")

# --- reference arm poses (from steve_simulation_moveit/config/mmo_700.srdf) ------------------
ARM_HOME = (1.5844, -1.4478, 2.2537, -0.8195, 1.5, 0.0)  # SRDF "home", order = ARM_JOINTS


# --- holonomic variant (steve_holonomic.urdf): virtual planar base ----------------------------
# world_anchor (welded to the world / env origin) -> x -> y -> yaw -> base_link
ANCHOR_LINK = "world_anchor"
BASE_X_JOINT = "base_x_joint"  # prismatic, anchor-frame x [m]
BASE_Y_JOINT = "base_y_joint"  # prismatic, anchor-frame y [m]
BASE_YAW_JOINT = "base_yaw_joint"  # continuous, about base_link z [rad]
BASE_PLANAR_JOINTS = (BASE_X_JOINT, BASE_Y_JOINT, BASE_YAW_JOINT)

HOLONOMIC_JOINTS = {
    BASE_X_JOINT: ("prismatic", ANCHOR_LINK, "base_x_link"),
    BASE_Y_JOINT: ("prismatic", "base_x_link", "base_y_link"),
    BASE_YAW_JOINT: ("continuous", "base_y_link", BASE_LINK),
    **JOINTS,
}
HOLONOMIC_LINKS = frozenset(LINKS | {ANCHOR_LINK, "base_x_link", "base_y_link"})
HOLONOMIC_MOVABLE_JOINTS = frozenset(MOVABLE_JOINTS | set(BASE_PLANAR_JOINTS))  # 23 DOFs

# Working "ready" configuration for the IL environments: TCP at READY_TCP_POS in the base_link frame,
# tool z pointing straight down (READY_TCP_QUAT = 180 deg about x). Found with the IK controller;
# wrist_3 is the 2*pi-equivalent solution closest to 0 (IK from ARM_HOME winds it ~-5 rad).
ARM_READY = (-0.344, -1.112, 1.876, 0.807, 1.571, 1.226)
READY_TCP_POS = (0.60, 0.0, 0.80)
READY_TCP_QUAT = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z)

# --- tool centre point ------------------------------------------------------------------------
# Midpoint between the two finger pads, expressed in the ur5e_tool0 frame (z = approach axis).
# Hand-derived from the URDF chain (0.0085 + 0.054904 - 0.0041 + 0.0471 + 0.03242); checked
# against the simulated pad positions by scripts/drive_test.py.
TCP_OFFSET_Z = 0.138824
FINGER_PAD_LINKS = ("left_inner_finger_pad", "right_inner_finger_pad")
