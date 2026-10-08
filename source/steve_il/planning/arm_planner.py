"""cuRobo planner for Steve's 6-DOF arm, working in the ``base_link`` frame.

Runs in its own process: Isaac Sim loads its bundled warp 1.8 into the app, cuRobo needs the newer pip
warp, and two versions cannot share a process. Inside Isaac Sim use ``planning.client.ArmPlannerClient``
(same API, talks to ``planning.server``); use ``ArmPlanner`` directly in plain python scripts.

The planner does not know where the base is: every call takes poses in the base frame, and
``set_world`` converts the world-frame obstacles into the base frame using the measured base pose.
Trajectories are joint positions sampled at ``dt`` (the env control period) in ``ARM_JOINTS`` order.

    planner = ArmPlanner()                                   # compiles CUDA kernels on first use (~30 s)
    planner.set_world(prims, base_xy_yaw)                    # obstacles, world frame
    plan = planner.plan_grasp(q, grasp_poses_base)           # approach -> grasp -> lift
    planner.attach("cube", q_at_grasp, obj_pose_base)        # object now travels with the gripper
    plan = planner.plan_grasp(q, place_poses_base)           # same chain, used to place
    planner.detach()
"""

from __future__ import annotations

import numpy as np
import torch
import yaml

from steve_il import ASSETS_DIR, CUROBO_ROBOT_PATH
from steve_il.planning.types import CONTROL_DT, GraspPlan, PlanFailure
from steve_il.planning.world import Prim, prims_to_scene_dict
from steve_il.robots import steve_names as n

DEVICE = "cuda"
ATTACHED_LINK = "attached_object"


def load_robot_config() -> dict:
    """The committed cuRobo robot YAML with its asset paths made absolute."""
    cfg = yaml.safe_load(CUROBO_ROBOT_PATH.read_text())
    kin = cfg["robot_cfg"]["kinematics"]
    kin["urdf_path"] = str(ASSETS_DIR / kin["urdf_path"])
    kin["asset_root_path"] = str(ASSETS_DIR / kin["asset_root_path"])
    return cfg


def _trim(traj, last_tstep) -> np.ndarray:
    """Valid part of a padded interpolated trajectory as [T, dof] numpy (same convention as cuRobo's trim)."""
    pos = traj.position
    pos = pos.reshape(-1, pos.shape[-1])
    last = int(last_tstep.reshape(-1)[0]) if last_tstep is not None else 0
    if last > 0:
        pos = pos[:last]
    return pos.detach().cpu().numpy().astype(np.float32)


class ArmPlanner:
    def __init__(
        self,
        max_goalset: int = 8,
        max_cuboids: int = 24,
        collision_margin: float = 0.005,
        warmup: bool = True,
    ):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self.max_goalset = max_goalset
        self.max_cuboids = max_cuboids
        self.collision_margin = collision_margin
        cfg = MotionPlannerCfg.create(
            robot=load_robot_config(),
            collision_cache={"cuboid": max_cuboids},
            interpolation_dt=CONTROL_DT,
            max_goalset=max_goalset,
        )
        self.planner = MotionPlanner(cfg)
        if warmup:
            self.planner.warmup(enable_graph=True, num_warmup_iterations=3)
        assert list(self.planner.joint_names) == list(n.ARM_JOINTS), self.planner.joint_names
        self._attached: str | None = None
        self._prims: dict[str, Prim] = {}
        self._reference_backup: torch.Tensor | None = None

    # -- helpers ---------------------------------------------------------------------------
    @property
    def tool_frame(self) -> str:
        return self.planner.tool_frames[0]

    def _js(self, q):
        from curobo.types import JointState

        t = torch.as_tensor(np.asarray(q, dtype=np.float32), device=DEVICE).reshape(1, -1)
        return JointState.from_position(t, joint_names=self.planner.joint_names)

    def _goal(self, poses: np.ndarray):
        from curobo.types import GoalToolPose

        poses = np.asarray(poses, dtype=np.float32).reshape(-1, 7)
        if len(poses) > self.max_goalset:
            raise ValueError(f"{len(poses)} candidates > max_goalset={self.max_goalset}")
        t = torch.as_tensor(poses, device=DEVICE)
        return GoalToolPose(
            tool_frames=self.planner.tool_frames,
            position=t[:, :3].reshape(1, 1, 1, -1, 3),
            quaternion=t[:, 3:].reshape(1, 1, 1, -1, 4),
        )

    # -- kinematics ------------------------------------------------------------------------
    def tcp_pose(self, q) -> np.ndarray:
        """Forward kinematics: TCP pose [7] in the base frame."""
        state = self.planner.compute_kinematics(self._js(q))
        p = state.tool_poses.get_link_pose(self.tool_frame)
        return torch.cat([p.position, p.quaternion], dim=-1).reshape(-1).cpu().numpy()

    # -- world -----------------------------------------------------------------------------
    def set_world(self, prims: list[Prim], base_xy_yaw) -> None:
        """Replace the collision world by ``prims`` (world frame), expressed relative to the base."""
        from curobo.scene import Scene

        if len(prims) > self.max_cuboids:
            raise ValueError(f"{len(prims)} obstacles > max_cuboids={self.max_cuboids}")
        scene = Scene.create(prims_to_scene_dict(prims, base_xy_yaw, margin=self.collision_margin))
        self.planner.update_world(scene)
        self._prims = {p.name: p for p in prims}

    # -- held object -----------------------------------------------------------------------
    def attach(self, name: str, q_grasp, obj_pose_base: np.ndarray, grid: int = 3) -> None:
        """Make world obstacle ``name`` part of the robot (rigidly on the TCP) from now on.

        ``obj_pose_base`` is the object's current pose in the base frame, ``q_grasp`` the arm
        configuration it is held in; their relative pose is frozen. The object is represented by a
        ``grid`` x ``grid`` x ``grid`` lattice of spheres covering its bounding box (cuRobo's own sphere
        fit returns almost nothing for a small cube: 2 of 32 slots).

        Two details of cuRobo that this works around: ``plan_grasp`` disables the contact links
        (including the held object) around some sub-plans and re-enables them from the *reference*
        spheres, which for the attached slots are "inactive", so the object would vanish after the first
        planning call; here the reference spheres are overwritten with the attached ones until ``detach``.
        """
        from curobo.types import Pose

        prim = self._prims[name]
        a, b, c = (d / grid for d in prim.box_dims)
        radius = 0.5 * float(np.sqrt(a * a + b * b + c * c)) + self.collision_margin
        offsets = [(np.arange(grid) - (grid - 1) / 2) * d for d in (a, b, c)]
        centres = np.array([[x, y, z] for x in offsets[0] for y in offsets[1] for z in offsets[2]])
        spheres = torch.as_tensor(
            np.concatenate([centres, np.full((len(centres), 1), radius)], axis=1), dtype=torch.float32, device=DEVICE
        )
        pose = torch.as_tensor(np.asarray(obj_pose_base, dtype=np.float32), device=DEVICE).reshape(1, 7)

        manager = self.planner.attachment_manager
        kparams = manager.kinematics_params
        idx = kparams.get_sphere_index_from_link_name(ATTACHED_LINK)
        if self._reference_backup is None:
            self._reference_backup = kparams.reference_link_spheres[:, idx, :].clone()
        manager.update(
            spheres,
            self._js(q_grasp),
            link_name=ATTACHED_LINK,
            world_objects_pose_offset=Pose(position=pose[:, :3], quaternion=pose[:, 3:]),
        )
        kparams.reference_link_spheres[:, idx, :] = kparams.link_spheres[:, idx, :].clone()
        self.planner.scene_collision_checker.enable_obstacle(name, enable=False, env_idx=0)  # it is on the robot now
        self._attached = name

    def detach(self) -> None:
        manager = self.planner.attachment_manager
        kparams = manager.kinematics_params
        if self._reference_backup is not None:
            idx = kparams.get_sphere_index_from_link_name(ATTACHED_LINK)
            kparams.reference_link_spheres[:, idx, :] = self._reference_backup
            manager.detach(link_name=ATTACHED_LINK)  # resets the slots from the (restored) reference
            self._reference_backup = None
        if self._attached is not None:
            self.planner.scene_collision_checker.enable_obstacle(self._attached, enable=True, env_idx=0)
        self._attached = None

    @property
    def attached(self) -> str | None:
        return self._attached

    # -- planning --------------------------------------------------------------------------
    def plan_grasp(
        self,
        q_start,
        tcp_poses_base: np.ndarray,
        approach_offset: float = 0.12,
        lift_offset: float = 0.15,
    ) -> GraspPlan | PlanFailure:
        """Approach -> straight-line grasp -> straight-line lift to one of the candidate TCP poses.

        ``approach_offset`` / ``lift_offset`` are distances [m] along the tool -z axis (away from the
        object). The same chain places an object (candidates = place poses, the lift is the retreat);
        with an object attached its collision is ignored during the final approach and the lift.
        """
        res = self.planner.plan_grasp(
            self._goal(tcp_poses_base),
            self._js(q_start),
            grasp_approach_offset=-approach_offset,
            grasp_lift_offset=-lift_offset,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
        )
        if res.success is None or not bool(res.success.any()):
            return PlanFailure(res.status or "unknown", res.planning_time)
        return GraspPlan(
            approach=_trim(res.approach_interpolated_trajectory, res.approach_interpolated_last_tstep),
            grasp=_trim(res.grasp_interpolated_trajectory, res.grasp_interpolated_last_tstep),
            lift=_trim(res.lift_interpolated_trajectory, res.lift_interpolated_last_tstep),
            goal_index=int(res.goalset_index.reshape(-1)[0]),
            planning_time=res.planning_time,
        )

    def plan_joints(self, q_start, q_goal) -> np.ndarray | PlanFailure:
        """Collision-free joint-space path to ``q_goal`` ([T, 6])."""
        res = self.planner.plan_cspace(self._js(q_goal), self._js(q_start))
        if res is None or not bool(res.success.any()):
            return PlanFailure("joint-space planning failed")
        return _trim(res.interpolated_trajectory, res.interpolated_last_tstep)
