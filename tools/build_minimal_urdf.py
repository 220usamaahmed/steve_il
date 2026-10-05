"""Derive the minimal Steve URDF (base + cabinet + UR5e + Robotiq 2F-85) from the
full xacro-expanded URDF in the ``steve_isaacsim`` repo and vendor its meshes.

This is a provenance tool: the generated ``assets/steve`` folder is committed, so the
project does not need the source repos at run time. Re-run it only to regenerate.

    python tools/build_minimal_urdf.py \
        --src ~/Work/steve_isaacsim/steve_new_gripper.urdf \
        --meshes ~/Work/steve_isaacsim/HRL-Steve-IsaacSim/steve_description/meshes

What is removed: IMU, both lidars, pan-tilt tower + Dynamixels + L515, wrist camera
mount + D405, all ``<gazebo>`` / ``<transmission>`` blocks.
What is rewritten: mesh paths become relative (``../meshes/...``) so the importer needs
neither a ROS package path nor the original ``/workspace`` container layout.
What is patched: see ``PATCHES`` below (every deviation from the source is listed there).
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# links (and every joint touching them) that are not part of the minimal robot
DROP_LINK = re.compile(
    r"^(imu|lidar_\d|pan_tilt_tower|dynamixel_|head_lidar_camera|ur5_wrist_cam_mount|wrist_camera)"
)

# Gripper links in the source carry mass=0.002 kg but inertia=1.0 kg m^2 (placeholder from
# the Robotiq xacro). ~500x too large for PhysX; replaced by a small isotropic inertia.
PLACEHOLDER_INERTIA = ("1.0", "1.0", "1.0")
PATCHED_INERTIA = "1e-06"

# USD prim names must be [A-Za-z0-9_]; the Isaac URDF importer builds prim paths from link,
# joint and *mesh file* names (and silently renames joints, which would make URDF and USD
# names differ), so everything is sanitised up front.
INVALID = re.compile(r"[^A-Za-z0-9_]")

# Frame-only links (ur5e_base_link, ur5e_flange, ur5e_tool0, ...) have no <inertial>; the Isaac
# importer then assigns 1 kg each (5 kg of phantom mass, 3 kg of it at the wrist).
FRAME_MASS = "0.001"
FRAME_INERTIA = "1e-07"

PATCHES = [
    f"links without <inertial> (frames) get mass {FRAME_MASS} kg / inertia {FRAME_INERTIA} kg m^2; "
    "otherwise the importer assigns 1 kg each",
    f"links with placeholder inertia {PLACEHOLDER_INERTIA} -> isotropic {PATCHED_INERTIA} kg m^2 "
    "(mass kept as in source)",
    "visual materials with an empty name (name=\"\") get a unique '<link>_material' name; the "
    "Isaac importer hangs on empty material names",
    "COLLADA (.dae) files without a <unit> element (all Robotiq meshes) get "
    "<unit name=\"meter\" meter=\"1\"/>; the Isaac importer hangs forever on them. Same convention "
    "as the UR5e / MPO-700 meshes: the files are *labelled* metres but the URDF scale 0.001 applies",
    "<material> inside <collision> removed (not valid URDF; the pads had a red debug colour there)",
    "joint/link names and mesh file stems: characters outside [A-Za-z0-9_] -> '_' "
    "(e.g. ur5e_flange-tool0 -> ur5e_flange_tool0, MPO-700-BODY.dae -> MPO_700_BODY.dae)",
]


def sanitize(name: str) -> str:
    return INVALID.sub("_", name)


def add_collada_unit(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "<unit" in text:
        return
    if "<up_axis>" not in text:
        raise ValueError(f"cannot add <unit> to {path}: no <up_axis> anchor")
    path.write_text(
        text.replace("<up_axis>", '<unit name="meter" meter="1"/>\n    <up_axis>', 1), encoding="utf-8"
    )


def sanitize_path(rel: str) -> str:
    """Sanitise the stem of a relative mesh path, keep directories and extension."""
    p = Path(rel)
    return str(p.with_name(sanitize(p.stem) + p.suffix))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="full expanded steve URDF")
    p.add_argument("--meshes", type=Path, required=True, help="steve_description/meshes dir")
    p.add_argument("--out", type=Path, default=ROOT / "assets" / "steve")
    args = p.parse_args()

    tree = ET.parse(args.src)
    robot = tree.getroot()
    robot.set("name", "steve")

    keep_links = {l.get("name") for l in robot.findall("link") if not DROP_LINK.match(l.get("name"))}

    for el in list(robot):
        if el.tag in ("transmission", "gazebo"):
            robot.remove(el)
        elif el.tag == "link" and el.get("name") not in keep_links:
            robot.remove(el)
        elif el.tag == "joint":
            parent, child = el.find("parent"), el.find("child")
            if parent is None or child is None:  # stray joint refs (transmissions)
                robot.remove(el)
            elif parent.get("link") not in keep_links or child.get("link") not in keep_links:
                robot.remove(el)

    meshes_out = args.out / "meshes"
    if meshes_out.exists():
        shutil.rmtree(meshes_out)
    copied: set[str] = set()
    for mesh in robot.iter("mesh"):
        rel = mesh.get("filename").split("meshes/", 1)[1]
        src = args.meshes / rel
        if not src.is_file():
            raise FileNotFoundError(f"mesh missing in source tree: {src}")
        rel_out = sanitize_path(rel)
        if rel_out not in copied:
            dst = meshes_out / rel_out
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if dst.suffix.lower() == ".dae":
                add_collada_unit(dst)
            copied.add(rel_out)
        mesh.set("filename", f"../meshes/{rel_out}")

    for el in robot.iter("link"):
        el.set("name", sanitize(el.get("name")))
    for el in robot.iter("joint"):
        el.set("name", sanitize(el.get("name")))
        for ref in ("parent", "child"):
            e = el.find(ref)
            e.set("link", sanitize(e.get("link")))
        mimic = el.find("mimic")
        if mimic is not None:
            mimic.set("joint", sanitize(mimic.get("joint")))

    n_patched = 0
    for link in robot.findall("link"):
        inertia = link.find("inertial/inertia")
        if inertia is None:
            continue
        if (inertia.get("ixx"), inertia.get("iyy"), inertia.get("izz")) == PLACEHOLDER_INERTIA:
            for k in ("ixx", "iyy", "izz"):
                inertia.set(k, PATCHED_INERTIA)
            for k in ("ixy", "ixz", "iyz"):
                inertia.set(k, "0")
            n_patched += 1

    for link in robot.findall("link"):
        for collision in link.findall("collision"):
            for mat in collision.findall("material"):
                collision.remove(mat)
        for visual in link.findall("visual"):
            mat = visual.find("material")
            if mat is not None and not mat.get("name"):
                mat.set("name", f"{link.get('name')}_material")

    n_frames = 0
    for link in robot.findall("link"):
        if link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=FRAME_MASS)
            ET.SubElement(
                inertial, "inertia", ixx=FRAME_INERTIA, ixy="0", ixz="0",
                iyy=FRAME_INERTIA, iyz="0", izz=FRAME_INERTIA,
            )
            n_frames += 1

    # drop materials no remaining visual refers to
    for mat in robot.findall("material"):
        name = mat.get("name")
        if not any(m.get("name") == name for l in robot.findall("link") for m in l.iter("material")):
            robot.remove(mat)

    ET.indent(tree, space="  ")
    out_urdf = args.out / "urdf" / "steve_minimal.urdf"
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)

    n_joints = len(robot.findall("joint"))
    print(f"links={len(robot.findall('link'))} joints={n_joints} meshes={len(copied)} "
          f"inertia_patched={n_patched} frame_inertials_added={n_frames}")
    print(f"wrote {out_urdf}")


if __name__ == "__main__":
    main()
