"""Pick-and-place environment: Steve in front of a table with objects, obstacles and a target marker.

Inherits the scene, observations and timing of ``SteveEnvCfg`` (physics 120 Hz, control 30 Hz) but
the arm is driven by **joint position** targets (what a motion planner outputs) instead of a TCP pose.

Action (dim 10), see ``steve_il.mdp.actions``:
    [vx vy wz (normalised) | 6 absolute arm joint positions [rad], ``ARM_JOINTS`` order | finger_joint target [rad]]
    The gripper command is continuous (0 = open ... 0.8 = closed) and drives all six Robotiq joints, so a
    controller can close it slowly and with a force set by ``GRIPPER_STIFFNESS`` * (target - contact angle).

Scene objects (see ``pick_scene``): ``table`` (static), ``cube_red`` / ``cube_green`` / ``cube_blue`` (dynamic, graspable),
``wall`` / ``tower`` (kinematic obstacles), ``marker`` (kinematic, no collision: shows the target).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

import steve_il.mdp as mdp
from steve_il.envs import pick_scene as ps
from steve_il.envs.steve_env_cfg import SteveEnvCfg, SteveSceneCfg
from steve_il.robots import steve_names as n
from steve_il.robots.steve_cfg import STEVE_HOLONOMIC_CFG

# gripper (see MimicGripperPositionAction): the two outer knuckles are driven and set the squeeze force of
# each finger; the inner-finger joints hold the pads parallel to the palm; the inner knuckles just follow
GRIPPER_STIFFNESS = 4.0  # [Nm/rad] outer knuckles: with a ~0.3 rad error the pads press ~10-20 N
GRIPPER_DAMPING = 0.3
GRIPPER_INNER_FINGER_STIFFNESS = 300.0
GRIPPER_INNER_FINGER_DAMPING = 15.0
GRIPPER_KNUCKLE_STIFFNESS = 20.0
GRIPPER_KNUCKLE_DAMPING = 1.0
GRIPPER_DRIVEN = (n.GRIPPER_DRIVEN_JOINT, "right_outer_knuckle_joint")
# follower -> (outer knuckle of the same side, multiplier); multipliers are the URDF mimic values
GRIPPER_FOLLOWERS = {
    "left_inner_finger_joint": (n.GRIPPER_DRIVEN_JOINT, n.GRIPPER_MIMIC_JOINTS["left_inner_finger_joint"]),
    "left_inner_knuckle_joint": (n.GRIPPER_DRIVEN_JOINT, n.GRIPPER_MIMIC_JOINTS["left_inner_knuckle_joint"]),
    "right_inner_finger_joint": ("right_outer_knuckle_joint", n.GRIPPER_MIMIC_JOINTS["right_inner_finger_joint"]),
    "right_inner_knuckle_joint": ("right_outer_knuckle_joint", n.GRIPPER_MIMIC_JOINTS["right_inner_knuckle_joint"]),
}
ACTION_DIM = 10
BASE = slice(0, 3)
ARM = slice(3, 9)
GRIPPER = 9


def _spawn_cfg(spec: ps.ObjectSpec, kinematic: bool, collision: bool = True, friction: float = 1.5):
    """Spawner for one box / cylinder. High friction (``max`` combine mode) so the pads can hold it."""
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic, max_depenetration_velocity=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color),
    )
    if not kinematic:
        common["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass)
        common["physics_material"] = sim_utils.RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction, restitution=0.0, friction_combine_mode="max"
        )
    if spec.kind == "box":
        return sim_utils.CuboidCfg(size=spec.size, **common)
    return sim_utils.CylinderCfg(radius=spec.size[0], height=spec.size[1], axis="Z", **common)


def _pick_robot_cfg() -> ArticulationCfg:
    """The holonomic Steve with the gripper joints driven as ``MimicGripperPositionAction`` expects."""
    actuators = dict(STEVE_HOLONOMIC_CFG.actuators)
    actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=list(GRIPPER_DRIVEN), stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING
    )
    actuators["gripper_mimic"] = ImplicitActuatorCfg(  # the inner-finger joints
        joint_names_expr=[j for j in GRIPPER_FOLLOWERS if "finger" in j],
        stiffness=GRIPPER_INNER_FINGER_STIFFNESS,
        damping=GRIPPER_INNER_FINGER_DAMPING,
    )
    actuators["gripper_knuckles"] = ImplicitActuatorCfg(
        joint_names_expr=[j for j in GRIPPER_FOLLOWERS if "knuckle" in j],
        stiffness=GRIPPER_KNUCKLE_STIFFNESS,
        damping=GRIPPER_KNUCKLE_DAMPING,
    )
    return STEVE_HOLONOMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", actuators=actuators)


@configclass
class StevePickSceneCfg(SteveSceneCfg):
    robot: ArticulationCfg = _pick_robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=ps.TABLE.size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(ps.TABLE.pose[:3])),
    )

    def __post_init__(self):
        # default layout (replaced per episode by ``apply_scenario``)
        defaults = {
            "cube_red": (0.62, 0.15), "cube_green": (0.70, -0.10), "cube_blue": (0.62, -0.20),
            "wall": (0.70, 0.20), "tower": (0.76, 0.0),
        }
        for spec in (*ps.OBJECTS, *ps.OBSTACLES):
            x, y = defaults[spec.name]
            kinematic = spec in ps.OBSTACLES
            setattr(
                self,
                spec.name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                    spawn=_spawn_cfg(spec, kinematic=kinematic),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(spec.prim(x, y).pose[:3])),
                ),
            )
        marker = ps.ObjectSpec(ps.MARKER_NAME, "cylinder", (ps.MARKER_RADIUS, 0.002), (0.1, 0.8, 0.2))
        setattr(
            self,
            ps.MARKER_NAME,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{ps.MARKER_NAME}",
                spawn=_spawn_cfg(marker, kinematic=True, collision=False),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.3, ps.TABLE_TOP_Z + 0.002)),
            ),
        )


@configclass
class PickActionsCfg:
    # order defines the layout of the action vector: base (3) | arm joints (6) | gripper (1)
    base = mdp.HolonomicBaseVelocityActionCfg(
        asset_name="robot",
        joint_names=list(n.BASE_PLANAR_JOINTS),
        scale=(0.8, 0.8, 1.0),
    )
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(n.ARM_JOINTS),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    gripper = mdp.MimicGripperPositionActionCfg(
        asset_name="robot",
        driven_joints=list(GRIPPER_DRIVEN),
        followers=GRIPPER_FOLLOWERS,
    )


@configclass
class StevePickEnvCfg(SteveEnvCfg):
    scene: StevePickSceneCfg = StevePickSceneCfg(num_envs=1, env_spacing=8.0)
    actions: PickActionsCfg = PickActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 600.0  # scripted pick-and-place must not be cut by the time-out
        self.viewer.eye = (1.9, -1.6, 1.5)
        self.viewer.lookat = (0.7, 0.0, 0.5)
