"""Build assets/steve/curobo/steve_arm.yml, the cuRobo robot model of Steve's arm (base_link frame).

Pipeline: ``steve_arm_planning.urdf`` -> cuRobo RobotBuilder (collision spheres fitted to the
collision meshes + self-collision ignore matrix) -> patched with the Steve-specific parts:

* ``tcp``              extra link at ``ur5e_tool0`` + TCP_OFFSET_Z; the planner's only tool frame
* ``attached_object``  extra link on ``tcp`` with sphere slots for a grasped object
* grasp-contact links  finger links whose collision is disabled while approaching / lifting an object
* joint limits         conservative velocity / acceleration / jerk for the 6 arm joints
* default joint pos    ``ARM_READY``

The fitted file is committed, so this only needs re-running when the URDF or the settings below
change. ``RobotBuilder`` imports ``viser``, which needs a newer ``websockets`` than Isaac Sim pins
(12.0), so run it with a newer copy on ``PYTHONPATH`` instead of touching the Isaac Sim env:

    pip install --target /tmp/ws16 "websockets>=13.1,<17"
    python tools/make_arm_planning_urdf.py       # only if steve_minimal.urdf changed
    PYTHONPATH=/tmp/ws16 python tools/build_curobo_robot.py

Needs a CUDA GPU. Paths in the written file are relative to ``assets/steve``; ``steve_il.planning``
resolves them at load time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from steve_il.robots import steve_names as n  # noqa: E402

TCP_LINK = "tcp"
ATTACHED_LINK = "attached_object"
ATTACHED_SPHERES = 32  # sphere slots reserved for a grasped object

BODY_LINKS = ("base_link", "cabinet_link")  # large boxes: coarse spheres are enough
BODY_DENSITY = 0.3
GRIPPER_LINKS = (
    "robotiq_coupling",
    "robotiq_arg2f_base_link",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_finger_pad",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_finger_pad",
    "right_inner_knuckle",
)
# fingers and knuckles may touch the object (the knuckles sit right on top of short objects): their
# collision is disabled for the final approach and the lift. The palm (robotiq_arg2f_base_link) is not.
GRASP_CONTACT_LINKS = (
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_finger_pad",
    "left_inner_knuckle",
    "left_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_finger_pad",
    "right_inner_knuckle",
    "right_outer_knuckle",
)

VELOCITY_SCALE = 0.5  # fraction of the URDF joint velocity limit (pi rad/s)
MAX_ACCELERATION = 4.0  # rad/s^2
MAX_JERK = 60.0  # rad/s^3


def build(urdf: Path, out: Path, seed: int, num_collision_samples: int) -> None:
    from curobo.robot_builder import RobotBuilder

    np.random.seed(seed)
    torch.manual_seed(seed)

    builder = RobotBuilder(urdf_path=str(urdf), asset_path=str(urdf.parent), tool_frames=[n.EE_LINK])
    builder.fit_collision_spheres(sphere_density=1.0, use_collision_mesh=True, compute_metrics=True)
    for link in BODY_LINKS:
        builder.refit_link_spheres(link, sphere_density=BODY_DENSITY, use_collision_mesh=True)
    print(f"{builder.num_spheres} spheres on {len(builder.collision_link_names)} links")
    builder.compute_collision_matrix(num_samples=num_collision_samples, seed=seed)

    tmp = out.with_suffix(".raw.yml")
    builder.save(builder.build(), str(tmp))
    kin = yaml.safe_load(tmp.read_text())["kinematics"]
    tmp.unlink()

    # --- repo-relative paths (resolved by steve_il.planning at load time)
    kin["asset_root_path"] = "urdf"
    kin["urdf_path"] = f"urdf/{urdf.name}"

    # --- tool frame and held-object link
    identity = [1, 0, 0, 0]
    kin["extra_links"] = {
        TCP_LINK: {
            "fixed_transform": [0.0, 0.0, n.TCP_OFFSET_Z, *identity],
            "joint_name": "tcp_joint",
            "joint_type": "FIXED",
            "link_name": TCP_LINK,
            "parent_link_name": n.EE_LINK,
        },
        ATTACHED_LINK: {
            "fixed_transform": [0.0, 0.0, 0.0, *identity],
            "joint_name": "attach_joint",
            "joint_type": "FIXED",
            "link_name": ATTACHED_LINK,
            "parent_link_name": TCP_LINK,
        },
    }
    kin["tool_frames"] = [TCP_LINK]
    kin["extra_collision_spheres"] = {ATTACHED_LINK: ATTACHED_SPHERES}
    kin["collision_link_names"] = [*kin["collision_link_names"], ATTACHED_LINK]
    kin["self_collision_buffer"][ATTACHED_LINK] = 0.0
    kin["grasp_contact_link_names"] = [*GRASP_CONTACT_LINKS, ATTACHED_LINK]

    # --- the held object touches the gripper and wrist by construction
    ignore = kin["self_collision_ignore"]
    attached_ignore = [*GRIPPER_LINKS, "ur5e_wrist_3_link", "ur5e_wrist_2_link"]
    ignore[ATTACHED_LINK] = attached_ignore
    for link in attached_ignore:
        ignore.setdefault(link, [])
        if ATTACHED_LINK not in ignore[link]:
            ignore[link].append(ATTACHED_LINK)

    # --- joint space
    cspace = kin["cspace"]
    cspace["default_joint_position"] = list(n.ARM_READY)
    cspace["velocity_scale"] = [VELOCITY_SCALE] * len(n.ARM_JOINTS)
    cspace["max_acceleration"] = [MAX_ACCELERATION] * len(n.ARM_JOINTS)
    cspace["max_jerk"] = [MAX_JERK] * len(n.ARM_JOINTS)
    assert cspace["joint_names"] == list(n.ARM_JOINTS), cspace["joint_names"]

    kin["lock_joints"] = None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": kin}}, sort_keys=False, default_flow_style=None))
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", type=Path, default=ROOT / "assets/steve/urdf/steve_arm_planning.urdf")
    p.add_argument("--out", type=Path, default=ROOT / "assets/steve/curobo/steve_arm.yml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-collision-samples", type=int, default=1000)
    args = p.parse_args()
    build(args.urdf.resolve(), args.out.resolve(), args.seed, args.num_collision_samples)


if __name__ == "__main__":
    main()
