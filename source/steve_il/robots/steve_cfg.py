"""Isaac Lab articulation configs for Steve (require a running Isaac Sim app to import).

* ``STEVE_CFG``            free-floating robot standing on its wheels (milestone 1 verification).
* ``STEVE_HOLONOMIC_CFG``  virtual x/y/yaw base welded to the world; this is what the IL
                           environments use. Base is velocity-driven, arm/gripper position-driven.

Gains are first-pass values: they hold pose and track IK targets (checked by
``scripts/verify_import.py`` and ``scripts/drive_test.py``) but are not tuned further.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from steve_il import HOLONOMIC_USD_PATH, USD_PATH
from steve_il.robots import steve_names as n

def _spawn(usd_path, disable_gravity: bool = False) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=disable_gravity,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    )


def _joint_pos(arm=n.ARM_READY) -> dict[str, float]:
    return {
        **dict(zip(n.ARM_JOINTS, arm)),
        n.GRIPPER_DRIVEN_JOINT: 0.0,
        **{j: 0.0 for j in n.GRIPPER_MIMIC_JOINTS},
        **{j: 0.0 for j in n.CASTER_JOINTS},
        **{j: 0.0 for j in n.WHEEL_JOINTS},
    }


def _actuators() -> dict[str, ImplicitActuatorCfg]:
    return {
        "arm_big": ImplicitActuatorCfg(
            joint_names_expr=list(n.ARM_JOINTS[:3]), stiffness=1500.0, damping=150.0
        ),
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=list(n.ARM_JOINTS[3:]), stiffness=300.0, damping=30.0
        ),
        # one driven joint; the five mimics are slaved to it (PhysX mimic constraint) and stay passive
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[n.GRIPPER_DRIVEN_JOINT], stiffness=200.0, damping=10.0
        ),
        "gripper_mimic": ImplicitActuatorCfg(
            joint_names_expr=list(n.GRIPPER_MIMIC_JOINTS), stiffness=0.0, damping=0.0
        ),
        "casters": ImplicitActuatorCfg(
            joint_names_expr=list(n.CASTER_JOINTS), stiffness=400.0, damping=40.0
        ),
        # velocity-controlled: stiffness 0, target = rad/s (zero target brakes the wheels)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=list(n.WHEEL_JOINTS), stiffness=0.0, damping=50.0
        ),
    }


STEVE_CFG = ArticulationCfg(
    spawn=_spawn(USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.02),  # base_link origin sits ~at wheel-contact height
        joint_pos=_joint_pos(n.ARM_HOME),
    ),
    actuators=_actuators(),
)

STEVE_HOLONOMIC_CFG = ArticulationCfg(
    # Gravity is off for the robot's own links: a real UR5e holds position exactly (its controller
    # compensates gravity), whereas a plain PD joint would sag ~1 cm and, because differential IK
    # re-targets from the *current* joints every step, never converge. Payload (grasped objects)
    # still has its own gravity. The base is kinematic (virtual joints), so it is unaffected.
    spawn=_spawn(HOLONOMIC_USD_PATH, disable_gravity=True),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={**_joint_pos(), **{j: 0.0 for j in n.BASE_PLANAR_JOINTS}},
    ),
    actuators={
        **_actuators(),
        # velocity drive (stiffness 0): target velocity -> force via damping, saturating at the effort limit.
        # Same structure as Isaac Lab's Ridgeback-Franka base.
        "base_planar": ImplicitActuatorCfg(
            joint_names_expr=list(n.BASE_PLANAR_JOINTS),
            effort_limit_sim=10000.0,
            velocity_limit_sim=5.0,
            stiffness=0.0,
            damping=1.0e5,
        ),
    },
)
