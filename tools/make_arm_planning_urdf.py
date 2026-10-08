"""Derive assets/steve/urdf/steve_arm_planning.urdf (the cuRobo planning model) from steve_minimal.urdf.

The arm planner works in the ``base_link`` frame with a stationary base, so the model only needs the
chassis body, cabinet, UR5e and gripper:

* the 4 casters and 4 wheels (and their joints) are removed: the arm cannot reach them
* the 6 gripper joints become ``fixed`` at the open pose (all zero): the gripper is not planned,
  it is opened / closed by the controller. This also avoids ``<mimic>`` handling in cuRobo.

Everything else (link names, joint names, limits, meshes) is byte-identical to the source, which
``tests/test_urdf.py`` enforces.

    python tools/make_arm_planning_urdf.py
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from steve_il.robots import steve_names as n  # noqa: E402

REMOVED_JOINTS = frozenset(n.BASE_JOINTS)
REMOVED_LINKS = frozenset(j.replace("_joint", "_link") for j in n.BASE_JOINTS)
FIXED_JOINTS = frozenset(n.GRIPPER_JOINTS)


def derive(src: Path) -> ET.ElementTree:
    tree = ET.parse(src)
    robot = tree.getroot()
    robot.set("name", "steve_arm_planning")
    for el in list(robot):
        name = el.get("name")
        if (el.tag == "joint" and name in REMOVED_JOINTS) or (el.tag == "link" and name in REMOVED_LINKS):
            robot.remove(el)
        elif el.tag == "joint" and name in FIXED_JOINTS:
            el.set("type", "fixed")
            for tag in ("axis", "limit", "mimic", "joint_properties", "dynamics"):
                for child in el.findall(tag):
                    el.remove(child)
    return tree


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=ROOT / "assets/steve/urdf/steve_minimal.urdf")
    p.add_argument("--out", type=Path, default=ROOT / "assets/steve/urdf/steve_arm_planning.urdf")
    args = p.parse_args()

    tree = derive(args.src)
    ET.indent(tree, space="  ")
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    robot = tree.getroot()
    print(f"wrote {args.out} (links={len(robot.findall('link'))} joints={len(robot.findall('joint'))})")


if __name__ == "__main__":
    main()
