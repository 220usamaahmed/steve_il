"""Derive assets/steve/urdf/steve_holonomic.urdf from steve_minimal.urdf.

Adds a world-fixed anchor and three virtual joints under ``base_link`` so the base can be
velocity-driven in the plane (same pattern as Isaac Lab's Ridgeback-Franka):

    world_anchor --(base_x_joint, prismatic x)--> base_x_link
                 --(base_y_joint, prismatic y)--> base_y_link
                 --(base_yaw_joint, continuous z)--> base_link --> (unchanged robot)

x/y are expressed in the anchor (= environment) frame, yaw about base_link's z axis. The
wheels and casters stay in the model (collision + visuals) but do not drive the robot.
The importer must be run with fix_base=True so ``world_anchor`` is welded to the world.

    python tools/make_holonomic_urdf.py
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Height of base_link above the ground plane. With 0 the wheels sit ~3 mm *above* the ground
# (measured: a free robot rests at base_link z = -3.2 mm), so they never touch it and cannot
# fight the virtual drive with friction.
BASE_Z = 0.0
VIRTUAL_MASS = "1.0"
VIRTUAL_INERTIA = "1e-3"
LIN_LIMIT = "1000"  # [m]; the anchor is the env origin, tasks never get near this
EFFORT = "10000"
LIN_VEL = "5.0"
ANG_VEL = "6.283"


def inertial(mass: str, inertia: str) -> ET.Element:
    el = ET.Element("inertial")
    ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(el, "mass", value=mass)
    ET.SubElement(el, "inertia", ixx=inertia, ixy="0", ixz="0", iyy=inertia, iyz="0", izz=inertia)
    return el


def link(name: str) -> ET.Element:
    el = ET.Element("link", name=name)
    el.append(inertial(VIRTUAL_MASS, VIRTUAL_INERTIA))
    return el


def joint(name: str, jtype: str, parent: str, child: str, axis: str, z: str = "0") -> ET.Element:
    el = ET.Element("joint", name=name, type=jtype)
    ET.SubElement(el, "origin", xyz=f"0 0 {z}", rpy="0 0 0")
    ET.SubElement(el, "parent", link=parent)
    ET.SubElement(el, "child", link=child)
    ET.SubElement(el, "axis", xyz=axis)
    if jtype == "prismatic":
        ET.SubElement(el, "limit", lower=f"-{LIN_LIMIT}", upper=LIN_LIMIT, effort=EFFORT, velocity=LIN_VEL)
    else:
        ET.SubElement(el, "limit", effort=EFFORT, velocity=ANG_VEL)
    return el


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=ROOT / "assets/steve/urdf/steve_minimal.urdf")
    p.add_argument("--out", type=Path, default=ROOT / "assets/steve/urdf/steve_holonomic.urdf")
    args = p.parse_args()

    tree = ET.parse(args.src)
    robot = tree.getroot()
    robot.set("name", "steve_holonomic")

    # new root structure goes first so world_anchor is the first link in the file
    new = [
        link("world_anchor"),
        link("base_x_link"),
        link("base_y_link"),
        joint("base_x_joint", "prismatic", "world_anchor", "base_x_link", "1 0 0"),
        joint("base_y_joint", "prismatic", "base_x_link", "base_y_link", "0 1 0"),
        joint("base_yaw_joint", "continuous", "base_y_link", "base_link", "0 0 1", z=str(BASE_Z)),
    ]
    first_link_idx = next(i for i, el in enumerate(robot) if el.tag == "link")
    for offset, el in enumerate(new):
        robot.insert(first_link_idx + offset, el)

    ET.indent(tree, space="  ")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {args.out} (links={len(robot.findall('link'))} joints={len(robot.findall('joint'))})")


if __name__ == "__main__":
    main()
