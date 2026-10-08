"""Scripted pick-and-place with a stationary base: cuRobo plans the arm, the env executes it.

    with ArmPlannerClient() as planner:    # cuRobo runs in its own process (see steve_il.planning.client)
        runner = PickPlaceRunner(env, planner)
        runner.reset(scenario)             # place objects / obstacles / marker, let them settle
        result = runner.pick_and_place()   # pick ``scenario.pick`` and put it on ``scenario.target_xy``

All goals are given in the **world frame**; the base pose is read every time a plan is made and the
planner works in the ``base_link`` frame (``ArmPlanner.set_world`` converts the obstacles).
Env action layout: ``steve_il.envs.steve_pick_env_cfg`` (base 3 | arm joints 6 | finger angle 1).

Gripper: the finger target ramps (fast, then slowly) to ``GRIP_TARGET``; with the low actuator gain of the
pick env this holds the cube with a moderate, roughly constant force (see ``MimicGripperPositionAction``).
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

import numpy as np
import torch

from steve_il import PROJECT_ROOT
from steve_il.envs import pick_scene as ps
from steve_il.envs.steve_env_cfg import GRIPPER_OPEN
from steve_il.envs.steve_pick_env_cfg import ACTION_DIM, ARM, GRIPPER
from steve_il.planning import geometry as geo
from steve_il.planning import grasps
from steve_il.planning.types import PlanFailure
from steve_il.planning.world import Prim
from steve_il.robots import gripper
from steve_il.robots import steve_names as n

GRIP_OPEN = GRIPPER_OPEN  # finger_joint targets [rad]
GRIP_TARGET = 0.75  # [rad] finger target while holding. The squeeze force of each finger is gain * (target - angle), so a
# fixed, large target with a low actuator gain gives a roughly constant force whatever angle the finger stops at
GRIP_PRE_CLOSE = 0.15  # [rad] close quickly to this far before the expected contact angle ...
GRIP_CLOSE_RATE = 0.30  # [rad/s] ... then ramp the target this slowly up to GRIP_TARGET
OPEN_SECONDS = 0.8
SETTLE_SECONDS = 0.3
APPROACH_OFFSETS = (0.12, 0.16, 0.09)  # [m] pre-grasp / pre-place heights above the grasp, tried in this order
LIFT_OFFSET = 0.22  # [m] lift after the grasp: clears the wall obstacle
RETREAT_OFFSET = 0.15

PLACE_TOLERANCE_XY = 0.03  # [m]
PLACE_TOLERANCE_TILT = np.radians(10.0)  # cubes: tilt away from lying flat on any face
HOLDING_RADIUS = 0.03  # [m] sideways distance of the object centre from the TCP after closing
HOLDING_HEIGHT = 0.05  # [m] along the tool axis


@dataclass
class Result:
    success: bool
    stage: str = "done"  # last stage reached (or the one that failed)
    reason: str = ""
    xy_error: float = float("nan")
    tilt: float = float("nan")
    sim_seconds: float = 0.0
    planning_seconds: float = 0.0
    max_tracking_error: float = 0.0


class PickPlaceRunner:
    def __init__(self, env, planner, verbose: bool = True, trace: bool = False):
        """``planner``: an ``ArmPlannerClient`` (or an ``ArmPlanner`` in a process that can run cuRobo)."""
        self.env = env
        self.planner = planner
        self.verbose = verbose
        self.trace = trace
        self.dt = env.step_dt
        self.device = env.device
        self.obs = None
        self.scenario: ps.Scenario | None = None
        self._q_cmd = np.array(n.ARM_READY, dtype=np.float32)
        self._grip = GRIP_OPEN
        self._grip_closed = GRIP_TARGET
        self._expected_contact = 0.4
        self.sim_steps = 0
        self.max_tracking_error = 0.0

    # -- simulation access -----------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _plan_with_retries(self, q_start, candidates_base, lift_offset):
        """``plan_grasp`` with a few approach heights: a plan that fails at one height often works at another."""
        failure = None
        for offset in APPROACH_OFFSETS:
            plan = self.planner.plan_grasp(q_start, candidates_base, offset, lift_offset)
            if not isinstance(plan, PlanFailure):
                return plan
            failure = plan
        return failure

    def _dump_failure(self, stage: str, q_start: np.ndarray, candidates_base: np.ndarray, base, attach=None) -> None:
        """Save the inputs of a failed plan to ``logs/plan_failures/`` so it can be replayed offline."""
        out = PROJECT_ROOT / "logs" / "plan_failures"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{stage}_{int(time.time())}.pkl"
        data = dict(stage=stage, q_start=q_start, candidates_base=candidates_base, base=tuple(base),
                    prims=self.world_prims(), scenario=self.scenario, attach=attach)  # attach: (name, q, pose in base)
        path.write_bytes(pickle.dumps(data))
        self._say(f"   (inputs saved to {path})")

    def _snap(self, label: str) -> None:
        """Trace line: where the picked object is relative to the world and the TCP (``trace=True``)."""
        if not (self.trace and self.scenario):
            return
        o = self.object_pose(self.scenario.pick)
        tcp = geo.base_to_world(self.planner.tcp_pose(self.arm_q), self.base_pose)
        rel = geo.pose_mul(geo.pose_inv(tcp), o)
        self._say(
            f"   [{label:>14s}] obj z {o[2]:.3f} tilt {np.degrees(geo.quat_tilt(o[3:])):5.1f} deg  "
            f"in TCP {np.round(rel[:3], 3)}  speed {self.object_speed(self.scenario.pick):.2f} m/s  "
            f"finger {self.finger:.3f}/{self._grip:.3f}"
        )

    @property
    def pol(self) -> dict:
        return self.obs["policy"]

    @property
    def arm_q(self) -> np.ndarray:
        return self.pol["arm_joint_pos"][0].cpu().numpy()

    @property
    def base_pose(self) -> np.ndarray:
        return self.pol["base_pose"][0].cpu().numpy()  # x, y, yaw in the env frame

    @property
    def finger(self) -> float:
        return float(self.pol["gripper_pos"][0, 0])

    def object_pose(self, name: str) -> np.ndarray:
        """Pose [7] of a scene object in the env frame."""
        o = self.env.scene[name]
        pos = (o.data.root_pos_w[0] - self.env.scene.env_origins[0]).cpu().numpy()
        return np.concatenate([pos, o.data.root_quat_w[0].cpu().numpy()])

    def object_speed(self, name: str) -> float:
        return float(self.env.scene[name].data.root_lin_vel_w[0].norm())

    def world_prims(self) -> list[Prim]:
        """Everything the arm must not hit: table, obstacles and objects, at their current poses."""
        prims = [ps.TABLE]
        for spec in (*ps.OBJECTS, *ps.OBSTACLES):
            p = Prim(spec.name, spec.kind, spec.size)
            p.pose = self.object_pose(spec.name)
            prims.append(p)
        return prims

    # -- stepping --------------------------------------------------------------------------------
    def step(self, q_target: np.ndarray | None = None, grip: float | None = None) -> None:
        if q_target is not None:
            self._q_cmd = np.asarray(q_target, dtype=np.float32)
        if grip is not None:
            self._grip = grip
        act = torch.zeros(1, ACTION_DIM, device=self.device)
        act[0, ARM] = torch.as_tensor(self._q_cmd, device=self.device)
        act[0, GRIPPER] = self._grip
        self.obs, *_ = self.env.step(act)
        self.sim_steps += 1
        self.max_tracking_error = max(self.max_tracking_error, float(np.abs(self._q_cmd - self.arm_q).max()))

    def hold(self, seconds: float, grip: float | None = None) -> None:
        for _ in range(round(seconds / self.dt)):
            self.step(grip=grip)

    def close_gripper(self, expected_angle: float | None = None, hold: float = 0.5) -> None:
        """Close on whatever is between the pads: fast to just before the expected contact angle, then the
        target ramps slowly up to ``GRIP_TARGET`` and holds."""
        if expected_angle is None:
            expected_angle = self._expected_contact
        pre = max(self._grip, expected_angle - GRIP_PRE_CLOSE)
        steps = max(1, round(0.5 / self.dt))
        start = self._grip
        for i in range(1, steps + 1):
            self.step(grip=start + (pre - start) * i / steps)
        target = pre
        while target < GRIP_TARGET:
            target = min(GRIP_TARGET, target + GRIP_CLOSE_RATE * self.dt)
            self.step(grip=target)
        self._grip_closed = GRIP_TARGET
        self.hold(hold, self._grip_closed)

    def execute(self, traj: np.ndarray, grip: float | None = None) -> None:
        """Follow a joint trajectory [T, 6] sampled at the control period, then settle briefly."""
        for q in traj:
            self.step(q, grip)
        self.hold(SETTLE_SECONDS)

    # -- scenario --------------------------------------------------------------------------------
    def reset(self, scenario: ps.Scenario, settle: float = 1.0) -> None:
        """Reset the env, put every object / obstacle / the marker where ``scenario`` says, let them settle.

        Order matters: a kinematic obstacle teleported next to a resting object makes PhysX imply a huge
        velocity and the object is flung away. So the cubes are parked out of the way, the obstacles are
        moved and given a few steps to arrive, and only then are the cubes put down. Raises if anything
        still ends up away from where it was put.
        """
        self.scenario = scenario
        self.obs, _ = self.env.reset()
        self._q_cmd = np.array(n.ARM_READY, dtype=np.float32)
        self._grip = GRIP_OPEN
        origin = self.env.scene.env_origins[0]
        for i, spec in enumerate(ps.OBJECTS):  # parked high above the floor, far from everything
            self._write_pose(spec.name, np.array([20.0 + i, 20.0, 20.0, 1, 0, 0, 0]), origin)
        for spec in ps.OBSTACLES:
            self._write_pose(spec.name, spec.prim(*scenario.poses[spec.name]).pose, origin)
        marker = np.array([*scenario.target_xy, ps.TABLE_TOP_Z + 0.002, 1, 0, 0, 0], dtype=float)
        self._write_pose(ps.MARKER_NAME, marker, origin)
        self.hold(0.1, GRIP_OPEN)
        for spec in ps.OBJECTS:
            self._write_pose(spec.name, spec.prim(*scenario.poses[spec.name]).pose, origin)
        self.max_tracking_error = 0.0
        self.sim_steps = 0
        self.hold(settle, GRIP_OPEN)
        moved = {
            k: float(np.linalg.norm(self.object_pose(k)[:2] - np.array(v[:2])))
            for k, v in scenario.poses.items()
            if np.linalg.norm(self.object_pose(k)[:2] - np.array(v[:2])) > 0.01
        }
        if moved:
            raise RuntimeError(f"scene did not settle: {moved} (m moved from where they were put)")

    def _write_pose(self, name: str, pose: np.ndarray, origin: torch.Tensor) -> None:
        obj = self.env.scene[name]
        t = torch.as_tensor(pose, dtype=torch.float32, device=self.device).reshape(1, 7).clone()
        t[:, :3] += origin
        obj.write_root_pose_to_sim(t)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))

    # -- the task --------------------------------------------------------------------------------
    def pick_and_place(self) -> Result:
        assert self.scenario is not None, "call reset(scenario) first"
        sc = self.scenario
        res = Result(False)
        t_plan = 0.0

        def fail(stage: str, reason: str) -> Result:
            res.stage, res.reason = stage, reason
            res.sim_seconds = self.sim_steps * self.dt
            res.planning_seconds = t_plan
            res.max_tracking_error = self.max_tracking_error
            self._say(f"[FAIL] {stage}: {reason}")
            return res

        spec = ps.SPECS[sc.pick]
        planner = self.planner

        # ---- plan the pick
        t0 = time.time()
        base = self.base_pose
        planner.set_world(self.world_prims(), base)
        obj_w = self.object_pose(sc.pick)
        prim = Prim(spec.name, spec.kind, spec.size)
        prim.pose = obj_w
        cands_w = grasps.topdown_grasps(prim, ps.TABLE_TOP_Z)
        if len(cands_w) == 0:
            return fail("pick-plan", "object does not fit the gripper")
        cands_b = np.array([geo.world_to_base(c, base) for c in cands_w])
        plan = self._plan_with_retries(self.arm_q, cands_b, LIFT_OFFSET)
        t_plan += time.time() - t0
        if isinstance(plan, PlanFailure):
            self._dump_failure("pick-plan", self.arm_q, cands_b, base)
            return fail("pick-plan", plan.status)
        width = grasps.grasp_width(prim, cands_w[plan.goal_index][3:])
        self._expected_contact = gripper.finger_angle_for_width(width)
        self._say(f"pick planned in {time.time() - t0:.2f}s ({sum(len(s) for s in plan.segments) * self.dt:.1f}s of motion)")

        # ---- execute approach + grasp, close the gripper
        self.execute(plan.approach, GRIP_OPEN)
        self._snap("approach")
        self.execute(plan.grasp, GRIP_OPEN)
        self._snap("grasp")
        self.close_gripper()
        self._snap("closed")
        base = self.base_pose
        tcp_b = planner.tcp_pose(self.arm_q)
        obj_b = geo.world_to_base(self.object_pose(sc.pick), base)
        obj_in_tcp = geo.pose_mul(geo.pose_inv(tcp_b), obj_b)
        lateral = float(np.linalg.norm(obj_in_tcp[:2]))
        if lateral > HOLDING_RADIUS or abs(obj_in_tcp[2]) > HOLDING_HEIGHT:
            return fail("grasp", f"object is off the TCP by {lateral * 100:.1f} cm sideways, {obj_in_tcp[2] * 100:.1f} cm along the tool after closing")
        self._say(f"grasped {sc.pick}: object in TCP frame {np.round(obj_in_tcp[:3], 3)}")

        # ---- lift, then plan the place from where the arm is now with the object as part of the robot
        self.execute(plan.lift, self._grip_closed)
        self._snap("lifted")
        if self.object_pose(sc.pick)[2] < ps.TABLE_TOP_Z + 0.5 * spec.size[-1] + 0.05:
            return fail("lift", "object did not come up with the gripper")
        t0 = time.time()
        base = self.base_pose
        planner.set_world(self.world_prims(), base)
        obj_b = geo.world_to_base(self.object_pose(sc.pick), base)
        planner.attach(sc.pick, self.arm_q, obj_b)
        # where the cube sits in the hand, with its in-grip tilt removed: it lands flat when released, and
        # compensating the tilt would tilt the tool and make the straight-line descent harder to reach
        flat_obj_b = geo.world_to_base(grasps.flatten(self.object_pose(sc.pick)), base)
        obj_in_tcp = geo.pose_mul(geo.pose_inv(planner.tcp_pose(self.arm_q)), flat_obj_b)
        rest_z = ps.TABLE_TOP_Z + 0.5 * prim.height + 0.002
        place_w = grasps.place_candidates(obj_in_tcp, sc.target_xy, rest_z)
        place_b = np.array([geo.world_to_base(c, base) for c in place_w])
        place = self._plan_with_retries(self.arm_q, place_b, RETREAT_OFFSET)
        t_plan += time.time() - t0
        if isinstance(place, PlanFailure):
            self._dump_failure("place-plan", self.arm_q, place_b, base, attach=(sc.pick, self.arm_q.copy(), obj_b))
            planner.detach()
            return fail("place-plan", place.status)
        self._say(f"place planned in {time.time() - t0:.2f}s")

        self.execute(place.approach, self._grip_closed)
        self._snap("place approach")
        self.execute(place.grasp, self._grip_closed)
        self._snap("place descent")
        self.hold(OPEN_SECONDS, GRIP_OPEN)
        self._snap("released")
        planner.detach()
        self.execute(place.lift, GRIP_OPEN)
        self._snap("retreated")
        self.hold(0.5, GRIP_OPEN)

        # ---- check
        final = self.object_pose(sc.pick)
        res.xy_error = float(np.linalg.norm(final[:2] - np.array(sc.target_xy)))
        res.tilt = geo.cube_tilt(final[3:]) if min(spec.size) == max(spec.size) else geo.quat_tilt(final[3:])
        res.stage = "done"
        res.success = res.xy_error <= PLACE_TOLERANCE_XY and res.tilt <= PLACE_TOLERANCE_TILT
        if not res.success:
            res.reason = f"object ended {res.xy_error * 100:.1f} cm from the target, tilt {np.degrees(res.tilt):.0f} deg"
        res.sim_seconds = self.sim_steps * self.dt
        res.planning_seconds = t_plan
        res.max_tracking_error = self.max_tracking_error
        self._say(
            f"[{'PASS' if res.success else 'FAIL'}] {sc.pick}: xy error {res.xy_error * 100:.1f} cm, "
            f"tilt {np.degrees(res.tilt):.1f} deg, {res.sim_seconds:.1f}s sim, planning {t_plan:.1f}s"
        )
        return res
