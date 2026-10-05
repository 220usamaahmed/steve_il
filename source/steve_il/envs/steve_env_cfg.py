"""Base environment: Steve alone on a ground plane, with all controllers.

Later task environments (objects, tables, cameras, success terminations) inherit from
``SteveEnvCfg`` and extend the scene / observations / terminations.

Timing: physics 120 Hz, control 30 Hz (decimation 4).

Action (dim 11), see ``steve_il.mdp.actions``:
    [vx vy wz (normalised) | tcp x y z qw qx qy qz in base_link frame | gripper (>=0 open, <0 close)]
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import steve_il.mdp as mdp
from steve_il.robots import steve_names as n
from steve_il.robots.steve_cfg import STEVE_HOLONOMIC_CFG

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.75  # rad; the Robotiq 2F-85 blocks at ~0.7565 on its own pads


@configclass
class SteveSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot: ArticulationCfg = STEVE_HOLONOMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    # order defines the layout of the action vector: base (3) | arm (7) | gripper (1)
    base = mdp.HolonomicBaseVelocityActionCfg(
        asset_name="robot",
        joint_names=list(n.BASE_PLANAR_JOINTS),
        scale=(0.8, 0.8, 1.0),
    )
    arm = mdp.BaseFrameDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=list(n.ARM_JOINTS),
        body_name=n.EE_LINK,
        base_body_name=n.BASE_LINK,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, n.TCP_OFFSET_Z)),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[n.GRIPPER_DRIVEN_JOINT],
        open_command_expr={n.GRIPPER_DRIVEN_JOINT: GRIPPER_OPEN},
        close_command_expr={n.GRIPPER_DRIVEN_JOINT: GRIPPER_CLOSED},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(n.ARM_JOINTS), preserve_order=True)},
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(n.ARM_JOINTS), preserve_order=True)},
        )
        gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[n.GRIPPER_DRIVEN_JOINT])},
        )
        tcp_pose = ObsTerm(func=mdp.tcp_pose_in_base)  # base_link frame, [xyz, wxyz]
        base_pose = ObsTerm(func=mdp.base_pose_xy_yaw)  # env frame [x, y, yaw]
        base_vel = ObsTerm(func=mdp.base_velocity_body)  # body frame [vx, vy, wz]
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # keep keys separate (robomimic-style dict observations)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """No reward: imitation learning only. (RewardManager requires the section to exist.)"""


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SteveEnvCfg(ManagerBasedRLEnvCfg):
    scene: SteveSceneCfg = SteveSceneCfg(num_envs=1, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (2.8, -2.8, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.6)
