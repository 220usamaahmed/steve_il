"""Checks of the cuRobo arm planner on its own (needs a GPU and cuRobo, not Isaac Sim).

    python scripts/planner_check.py          # exit code != 0 on failure

Covers what went wrong while building it: the model's kinematics against the simulator's,
planning in a rotated / translated base frame, segment end points and limits, and the held
object being visible to the planner (cuRobo's own attach left 2 of 32 spheres and the planner's
disable / enable cycle wiped them, so the cube was dragged through the wall).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import numpy as np  # noqa: E402

from steve_il.envs import pick_scene as ps  # noqa: E402
from steve_il.planning import geometry as geo  # noqa: E402
from steve_il.planning import grasps  # noqa: E402
from steve_il.planning.arm_planner import ATTACHED_LINK, ArmPlanner  # noqa: E402
from steve_il.planning.types import CONTROL_DT, GraspPlan  # noqa: E402
from steve_il.planning.world import Prim  # noqa: E402
from steve_il.robots import steve_names as n  # noqa: E402

FAILURES: list[str] = []


def report(ok: bool, what: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {what}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(what)


def active_spheres(planner: ArmPlanner) -> int:
    kp = planner.planner.attachment_manager.kinematics_params
    idx = kp.get_sphere_index_from_link_name(ATTACHED_LINK)
    return int((kp.link_spheres[0, idx, 3] > 0).sum())


def cube_points(pose: np.ndarray, size: float, k: int = 5) -> np.ndarray:
    """A k^3 lattice of points filling a cube of edge ``size`` with the given pose."""
    g = (np.arange(k) - (k - 1) / 2) * size / (k - 1)
    local = np.array([[x, y, z] for x in g for y in g for z in g])
    return np.array([pose[:3] + geo.quat_rotate(pose[3:], p) for p in local])


def inside_box(points: np.ndarray, box: Prim) -> np.ndarray:
    """Which points lie inside the oriented box ``box`` (world frame)."""
    half = np.array(box.box_dims) / 2
    local = np.array([geo.quat_rotate(geo.quat_conj(box.pose[3:]), p - box.pose[:3]) for p in points])
    return np.all(np.abs(local) <= half, axis=1)


def main() -> int:
    planner = ArmPlanner()
    q0 = np.array(n.ARM_READY, dtype=np.float32)

    # ---- A. the model's kinematics agree with the simulator's ready pose
    tcp = planner.tcp_pose(q0)
    err = np.linalg.norm(tcp[:3] - np.array(n.READY_TCP_POS))
    report(err < 1e-3 and abs(np.dot(tcp[3:], n.READY_TCP_QUAT)) > 0.9999, "FK at ARM_READY = simulated TCP pose", f"{err * 1000:.2f} mm")

    # ---- B/C/D. pick plan from a base standing at an arbitrary pose
    base = (1.3, -0.7, 2.1)
    scenario = ps.sample_scenario(np.random.default_rng(3))
    prims_w = [Prim(p.name, p.kind, p.size, geo.base_to_world(p.pose, base)) for p in [ps.TABLE, *scenario.prims()]]
    pick_w = next(p for p in prims_w if p.name == scenario.pick)
    planner.set_world(prims_w, base)
    # the base only turns about z, so heights are the same in the base and world frames: the grasp
    # candidates can be generated directly from the object's pose in the base frame
    pick_b = geo.world_to_base(pick_w.pose, base)
    candidates_b = grasps.topdown_grasps(Prim("pick", "box", pick_w.size, pick_b), support_z=ps.TABLE_TOP_Z)
    plan = planner.plan_grasp(q0, candidates_b, 0.12, 0.22)
    report(isinstance(plan, GraspPlan), "pick plan found with the base at (1.3, -0.7, yaw 2.1)")
    if not isinstance(plan, GraspPlan):
        return 1
    goal = candidates_b[plan.goal_index]
    end = planner.tcp_pose(plan.grasp[-1])
    report(np.linalg.norm(end[:3] - goal[:3]) < 1e-3 and abs(np.dot(end[3:], goal[3:])) > 0.9999,
           "grasp segment ends at the chosen grasp pose", f"{np.linalg.norm(end[:3] - goal[:3]) * 1000:.3f} mm")
    approach_end = planner.tcp_pose(plan.approach[-1])
    report(abs(approach_end[2] - goal[2] - 0.12) < 2e-3, "approach ends 12 cm above the grasp", f"{(approach_end[2] - goal[2]) * 100:.1f} cm")
    lift_end = planner.tcp_pose(plan.lift[-1])
    report(abs(lift_end[2] - goal[2] - 0.22) < 2e-3, "lift ends 22 cm above the grasp", f"{(lift_end[2] - goal[2]) * 100:.1f} cm")
    steps = np.concatenate([np.abs(np.diff(s, axis=0)) for s in plan.segments]) / CONTROL_DT
    report(steps.max() < 0.5 * np.pi + 0.05, "joint speed within 0.5 x the UR5e limit", f"max {steps.max():.2f} rad/s")
    report(np.allclose(plan.approach[0], q0, atol=1e-3), "approach starts at the current configuration")

    # ---- E. the held cube is visible to the planner and is carried clear of the wall
    tcp_at_grasp = planner.tcp_pose(plan.grasp[-1])
    held_in_tcp = geo.pose_mul(geo.pose_inv(tcp_at_grasp), pick_b)
    held_flat = geo.pose_mul(geo.pose_inv(tcp_at_grasp), grasps.flatten(pick_b))
    planner.attach(scenario.pick, plan.grasp[-1], pick_b)
    report(active_spheres(planner) >= 20, "attached cube has collision spheres", f"{active_spheres(planner)}/32")
    wall_w = next(p for p in prims_w if p.name == "wall")
    wall_b = Prim("wall", "box", wall_w.size, geo.world_to_base(wall_w.pose, base))
    target_b = np.array([*scenario.target_xy, ps.TABLE_TOP_Z, 1, 0, 0, 0])  # the scenario's coordinates are base-frame ones
    place = grasps.place_candidates(held_flat, tuple(target_b[:2]), ps.TABLE_TOP_Z + 0.027)
    # everything here is already in the base frame, which is what the planner wants
    place_plan = planner.plan_grasp(plan.lift[-1], place, 0.12, 0.15)
    report(isinstance(place_plan, GraspPlan), "place plan found with the cube attached")
    report(active_spheres(planner) >= 20, "attached spheres survive planning", f"{active_spheres(planner)}/32")
    if isinstance(place_plan, GraspPlan):
        worst = 0
        for q in place_plan.approach:
            cube = geo.pose_mul(planner.tcp_pose(q), held_in_tcp)
            worst = max(worst, int(inside_box(cube_points(cube, pick_w.size[0]), wall_b).sum()))
        report(worst == 0, "carried cube never intersects the wall", f"{worst} lattice points inside")
    planner.detach()
    report(active_spheres(planner) == 0, "detach removes the attached spheres")

    print(f"\n{'all checks passed' if not FAILURES else str(len(FAILURES)) + ' check(s) FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
