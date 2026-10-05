"""Verify the converted Steve USD: structure (names/types/mimics) and sim behaviour.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_import.py --headless

Checks
  1. USD structure   rigid bodies, joints, joint types == steve_names; mimic API present.
  2. Articulation    Isaac Lab sees exactly the expected DOF / body names, one root.
  3. Rest            robot settles on the ground plane: finite state, base height stable,
                     arm holds ARM_HOME (reports sag), wheels do not creep.
  4. Gripper         commanding finger_joint drags every mimic joint by its multiplier.
Exit code is non-zero if any check fails.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--rest-steps", type=int, default=360, help="sim steps for the rest test")
parser.add_argument("--gripper-steps", type=int, default=240, help="sim steps for the gripper test")
parser.add_argument("--dump", action="store_true", help="print the full body/joint tables")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

from steve_il import URDF_PATH, USD_PATH  # noqa: E402
from steve_il.robots import steve_names as n  # noqa: E402
from steve_il.robots.steve_cfg import STEVE_CFG  # noqa: E402

FAILURES: list[str] = []


def report(ok: bool, what: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {what}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(what)


def check_usd_structure() -> None:
    stage = Usd.Stage.Open(str(USD_PATH))
    bodies = {p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    roots = [p.GetPath().pathString for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]

    report(bodies == set(n.LINKS), "USD rigid bodies == expected links",
           f"missing={sorted(set(n.LINKS) - bodies)} extra={sorted(bodies - set(n.LINKS))}")
    report(set(joints) == set(n.JOINTS), "USD joints == expected joints",
           f"missing={sorted(set(n.JOINTS) - set(joints))} extra={sorted(set(joints) - set(n.JOINTS))}")
    report(len(roots) == 1, "single articulation root", str(roots))

    type_of = {"continuous": UsdPhysics.RevoluteJoint, "revolute": UsdPhysics.RevoluteJoint,
               "fixed": UsdPhysics.FixedJoint}
    bad = [j for j, (t, _, _) in n.JOINTS.items() if j in joints and not joints[j].IsA(type_of[t])]
    report(not bad, "USD joint types match", str(bad))

    mimic_api = getattr(PhysxSchema, "PhysxMimicJointAPI", None)
    if mimic_api is None:
        report(False, "PhysxMimicJointAPI available in this Isaac Sim build")
    else:
        have = {j for j, p in joints.items() if any(a.startswith("PhysxMimicJointAPI") for a in p.GetAppliedSchemas())}
        report(have == set(n.GRIPPER_MIMIC_JOINTS), "mimic joints imported",
               f"found={sorted(have)}")


def build_scene() -> tuple[sim_utils.SimulationContext, Articulation]:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    ground, light = sim_utils.GroundPlaneCfg(), sim_utils.DomeLightCfg(intensity=2000.0)
    ground.func("/World/ground", ground)
    light.func("/World/light", light)
    robot = Articulation(STEVE_CFG.replace(prim_path="/World/Steve"))
    sim.reset()
    return sim, robot


def check_articulation(robot: Articulation) -> None:
    report(set(robot.joint_names) == set(n.MOVABLE_JOINTS), "articulation DOFs == expected movable joints",
           f"missing={sorted(set(n.MOVABLE_JOINTS) - set(robot.joint_names))} "
           f"extra={sorted(set(robot.joint_names) - set(n.MOVABLE_JOINTS))}")
    report(set(robot.body_names) == set(n.LINKS), "articulation bodies == expected links",
           f"missing={sorted(set(n.LINKS) - set(robot.body_names))} "
           f"extra={sorted(set(robot.body_names) - set(n.LINKS))}")
    report(robot.num_joints == 20, "20 DOFs", str(robot.num_joints))
    urdf_mass = sum(
        float(m.get("value")) for m in ET.parse(URDF_PATH).getroot().iter("mass")
    )
    mass = robot.root_physx_view.get_masses().sum().item()
    report(abs(mass - urdf_mass) < 0.01, "total mass matches URDF (no phantom link masses)",
           f"sim {mass:.3f} kg vs urdf {urdf_mass:.3f} kg")
    if args.dump:
        for i, name in enumerate(robot.joint_names):
            print(f"  dof {i:2d} {name}")
        for i, name in enumerate(robot.body_names):
            print(f"  body {i:2d} {name}")


def check_rest(sim: sim_utils.SimulationContext, robot: Articulation) -> None:
    """Drop onto the ground, let it settle, then require it to hold still for a window."""
    dt = sim.get_physics_dt()
    arm_ids = [robot.joint_names.index(j) for j in n.ARM_JOINTS]
    wheel_ids = [robot.joint_names.index(j) for j in n.WHEEL_JOINTS]
    target = robot.data.default_joint_pos.clone()

    def run(steps: int) -> None:
        for _ in range(steps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

    run(args.rest_steps)  # settle
    p0 = robot.data.root_pos_w[0].clone()
    run(args.rest_steps)  # observation window
    p1 = robot.data.root_pos_w[0]
    window = args.rest_steps * dt

    finite = bool(torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.root_pos_w).all())
    report(finite, "state finite after rest test")
    drift = float(torch.linalg.norm(p1[:2] - p0[:2]))
    report(drift < 0.005, "base holds still once settled", f"xy drift {drift * 1000:.2f} mm in {window:.1f} s")
    report(-0.02 < p1[2].item() < 0.05, "base rests at wheel height", f"z = {p1[2].item():.4f} m")
    sag = (robot.data.joint_pos[0, arm_ids] - robot.data.default_joint_pos[0, arm_ids]).abs().max().item()
    report(sag < 0.05, "arm holds ARM_HOME under gravity", f"max deviation {sag:.4f} rad")
    wheel_speed = robot.data.joint_vel[0, wheel_ids].abs().max().item()
    report(wheel_speed < 0.05, "wheels at rest", f"max |w| {wheel_speed:.4f} rad/s")


def check_gripper(sim: sim_utils.SimulationContext, robot: Articulation) -> None:
    """Command finger_joint; every mimic joint must follow multiplier * finger_joint."""
    dt = sim.get_physics_dt()
    names = robot.joint_names
    drv = names.index(n.GRIPPER_DRIVEN_JOINT)
    target = robot.data.default_joint_pos.clone()
    target[:, drv] = 0.4

    def mimic_err() -> dict[str, float]:
        q = robot.data.joint_pos[0]
        return {j: q[names.index(j)].item() - m * q[drv].item() for j, m in n.GRIPPER_MIMIC_JOINTS.items()}

    for step in range(1, args.gripper_steps + 1):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if args.dump and step % 60 == 0:
            e = mimic_err()
            print(f"  t={step * dt:4.2f}s finger={robot.data.joint_pos[0, drv].item():.3f} "
                  + " ".join(f"{k.replace('_joint', '')}={v:+.3f}" for k, v in e.items()), flush=True)
    q = robot.data.joint_pos[0]
    report(abs(q[drv].item() - 0.4) < 0.05, "finger_joint reaches target", f"{q[drv].item():.3f} rad")
    for j, err in mimic_err().items():
        report(abs(err) < 0.05, f"mimic {j} follows x{n.GRIPPER_MIMIC_JOINTS[j]:+.0f}", f"err {err:+.4f} rad")


def main() -> int:
    check_usd_structure()
    sim, robot = build_scene()
    check_articulation(robot)
    check_rest(sim, robot)
    check_gripper(sim, robot)
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILED: ' + '; '.join(FAILURES)}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Kit's close() swallows the exit status
