"""Static checks on the vendored minimal URDF. No Isaac Sim needed.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from steve_il import HOLONOMIC_URDF_PATH, URDF_PATH  # noqa: E402
from steve_il.robots import steve_names as names  # noqa: E402


class MinimalUrdfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = ET.parse(URDF_PATH).getroot()
        cls.links = {l.get("name"): l for l in cls.robot.findall("link")}
        cls.joints = {j.get("name"): j for j in cls.robot.findall("joint")}

    def test_links_match_names(self):
        self.assertEqual(set(self.links), set(names.LINKS))

    def test_joints_match_names(self):
        self.assertEqual(set(self.joints), set(names.JOINTS))
        for name, (jtype, parent, child) in names.JOINTS.items():
            j = self.joints[name]
            self.assertEqual(j.get("type"), jtype, name)
            self.assertEqual(j.find("parent").get("link"), parent, name)
            self.assertEqual(j.find("child").get("link"), child, name)

    def test_single_root_and_tree(self):
        children = {j.find("child").get("link") for j in self.joints.values()}
        roots = set(self.links) - children
        self.assertEqual(roots, {names.BASE_LINK})
        self.assertEqual(len(self.joints), len(self.links) - 1)

    def test_no_stripped_content(self):
        self.assertEqual(self.robot.findall("gazebo"), [])
        self.assertEqual(self.robot.findall("transmission"), [])
        for banned in ("imu", "lidar", "pan_tilt", "dynamixel", "camera"):
            self.assertFalse([n for n in self.links if banned in n], banned)

    def test_meshes_relative_and_present(self):
        meshes = list(self.robot.iter("mesh"))
        self.assertTrue(meshes)
        for m in meshes:
            f = m.get("filename")
            self.assertFalse(f.startswith(("package://", "/", "file:")), f)
            self.assertTrue((URDF_PATH.parent / f).resolve().is_file(), f)

    def test_material_names_nonempty_and_unique_colors(self):
        """Empty material names make the Isaac URDF importer hang."""
        defs = {}
        for mat in self.robot.iter("material"):
            name = mat.get("name")
            self.assertTrue(name, "material with empty name")
            color = mat.find("color")
            if color is not None:
                self.assertNotIn(name, defs, f"duplicate colour definition {name}")
                defs[name] = color.get("rgba")

    def test_collada_files_declare_unit(self):
        """DAEs without <unit> hang the Isaac URDF importer."""
        daes = list((URDF_PATH.parent.parent / "meshes").rglob("*.dae"))
        self.assertTrue(daes)
        for dae in daes:
            self.assertIn("<unit", dae.read_text(encoding="utf-8", errors="ignore"), dae.name)

    def test_names_are_usd_safe(self):
        import re

        for name in (*self.links, *self.joints):
            self.assertRegex(name, r"^[A-Za-z0-9_]+$")
        for m in self.robot.iter("mesh"):
            self.assertRegex(Path(m.get("filename")).stem, r"^[A-Za-z0-9_]+$")

    def test_limits_on_movable_joints(self):
        for name in names.MOVABLE_JOINTS:
            j = self.joints[name]
            lim = j.find("limit")
            self.assertIsNotNone(lim, name)
            self.assertIsNotNone(lim.get("effort"), name)
            self.assertIsNotNone(lim.get("velocity"), name)
            if j.get("type") == "revolute":
                self.assertLess(float(lim.get("lower")), float(lim.get("upper")), name)

    def test_arm_joint_limits(self):
        for name in names.ARM_JOINTS:
            lim = self.joints[name].find("limit")
            # UR5e shoulder/wrist joints +-2pi, elbow +-pi (as in the source description)
            bound = 3.1416 if "elbow" in name else 6.2832
            self.assertAlmostEqual(float(lim.get("upper")), bound, places=3, msg=name)
            self.assertAlmostEqual(float(lim.get("lower")), -bound, places=3, msg=name)

    def test_gripper_mimics(self):
        for name, mult in names.GRIPPER_MIMIC_JOINTS.items():
            mimic = self.joints[name].find("mimic")
            self.assertIsNotNone(mimic, name)
            self.assertEqual(mimic.get("joint"), names.GRIPPER_DRIVEN_JOINT)
            self.assertEqual(float(mimic.get("multiplier")), mult)
            self.assertEqual(float(mimic.get("offset", 0)), 0.0)
        self.assertIsNone(self.joints[names.GRIPPER_DRIVEN_JOINT].find("mimic"))

    def test_inertials_sane(self):
        """Principal inertia must not exceed m * (1 m)^2 (catches 0.002 kg / 1.0 kg m^2)."""
        for name, link in self.links.items():
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            self.assertGreater(mass, 0.0, name)
            i = inertial.find("inertia")
            diag = [float(i.get(k)) for k in ("ixx", "iyy", "izz")]
            self.assertTrue(all(d > 0 for d in diag), name)
            self.assertLessEqual(max(diag), mass * 1.0, f"{name}: inertia too large for mass")
            # triangle inequality for a physically realisable inertia
            a, b, c = sorted(diag)
            self.assertLessEqual(c, a + b + 1e-9, name)

    def test_total_mass(self):
        """Change detector for the mass budget (kg)."""
        expected = {
            "base": 200.0,
            "casters": 4 * 9.0,
            "wheels": 4 * 5.9,
            "cabinet": 2.0,
            "ur5e": 4.0 + 3.761 + 8.058 + 2.846 + 1.37 + 1.3 + 0.365,
            "gripper": 0.002 + 0.22652 + 10 * 0.002,
            "frames": 5 * 0.001,  # ur5e_base_link, _base, _ft_frame, _flange, _tool0
        }
        total = sum(
            float(l.find("inertial/mass").get("value"))
            for l in self.links.values()
            if l.find("inertial") is not None
        )
        self.assertAlmostEqual(total, sum(expected.values()), places=3)


class HolonomicUrdfTest(unittest.TestCase):
    """steve_holonomic.urdf == steve_minimal.urdf + world_anchor / x / y / yaw virtual base."""

    @classmethod
    def setUpClass(cls):
        cls.robot = ET.parse(HOLONOMIC_URDF_PATH).getroot()
        cls.minimal = ET.parse(URDF_PATH).getroot()
        cls.links = {l.get("name"): l for l in cls.robot.findall("link")}
        cls.joints = {j.get("name"): j for j in cls.robot.findall("joint")}

    def test_structure_matches_names(self):
        self.assertEqual(set(self.links), set(names.HOLONOMIC_LINKS))
        self.assertEqual(set(self.joints), set(names.HOLONOMIC_JOINTS))
        for name, (jtype, parent, child) in names.HOLONOMIC_JOINTS.items():
            j = self.joints[name]
            self.assertEqual(j.get("type"), jtype, name)
            self.assertEqual(j.find("parent").get("link"), parent, name)
            self.assertEqual(j.find("child").get("link"), child, name)

    def test_robot_part_is_unchanged(self):
        """Everything except the three virtual links/joints is identical to the minimal URDF."""
        virtual_links = {names.ANCHOR_LINK, "base_x_link", "base_y_link"}
        for link in self.minimal.findall("link"):
            a = ET.tostring(link)
            b = ET.tostring(self.links[link.get("name")])
            self.assertEqual(a.strip(), b.strip(), link.get("name"))
        for joint in self.minimal.findall("joint"):
            self.assertEqual(
                ET.tostring(joint).strip(), ET.tostring(self.joints[joint.get("name")]).strip(), joint.get("name")
            )
        self.assertEqual(set(self.links) - virtual_links, {l.get("name") for l in self.minimal.findall("link")})

    def test_single_root_is_anchor(self):
        children = {j.find("child").get("link") for j in self.joints.values()}
        self.assertEqual(set(self.links) - children, {names.ANCHOR_LINK})
        self.assertEqual(len(self.joints), len(self.links) - 1)

    def test_planar_axes(self):
        axes = {j: tuple(float(v) for v in self.joints[j].find("axis").get("xyz").split()) for j in names.BASE_PLANAR_JOINTS}
        self.assertEqual(axes[names.BASE_X_JOINT], (1.0, 0.0, 0.0))
        self.assertEqual(axes[names.BASE_Y_JOINT], (0.0, 1.0, 0.0))
        self.assertEqual(axes[names.BASE_YAW_JOINT], (0.0, 0.0, 1.0))

    def test_dof_count(self):
        self.assertEqual(len(names.HOLONOMIC_MOVABLE_JOINTS), 23)
        movable = {n for n, j in self.joints.items() if j.get("type") != "fixed"}
        self.assertEqual(movable, set(names.HOLONOMIC_MOVABLE_JOINTS))

    def test_virtual_inertials_sane(self):
        for name in (names.ANCHOR_LINK, "base_x_link", "base_y_link"):
            mass = float(self.links[name].find("inertial/mass").get("value"))
            self.assertGreater(mass, 0.0)
            self.assertLess(mass, 5.0)  # must stay negligible next to the 283 kg robot


class ReferencePoseTest(unittest.TestCase):
    def test_ready_pose_inside_limits(self):
        self.assertEqual(len(names.ARM_READY), len(names.ARM_JOINTS))
        robot = ET.parse(URDF_PATH).getroot()
        for q, joint in zip(names.ARM_READY, names.ARM_JOINTS):
            lim = robot.find(f"joint[@name='{joint}']/limit")
            self.assertLess(float(lim.get("lower")), q, joint)
            self.assertLess(q, float(lim.get("upper")), joint)
        # wrist_3 must not be wound up: demos start from a pose far from its +-2pi limits
        self.assertLess(abs(names.ARM_READY[5]), 3.2)

    def test_ready_quat_is_unit(self):
        self.assertAlmostEqual(sum(c * c for c in names.READY_TCP_QUAT), 1.0)


if __name__ == "__main__":
    unittest.main()
