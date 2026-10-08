"""Checks of the planning helpers that need neither Isaac Sim nor cuRobo nor a GPU.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "tools"))

from steve_il import ARM_PLANNING_URDF_PATH, URDF_PATH  # noqa: E402
from steve_il.envs import pick_scene as ps  # noqa: E402
from steve_il.planning import geometry as geo  # noqa: E402
from steve_il.planning import grasps  # noqa: E402
from steve_il.planning.world import Prim, prims_to_scene_dict  # noqa: E402
from steve_il.robots import gripper  # noqa: E402
from steve_il.robots import steve_names as names  # noqa: E402


def random_pose(rng) -> np.ndarray:
    q = rng.normal(size=4)
    return np.concatenate([rng.uniform(-1, 1, 3), q / np.linalg.norm(q)])


class GeometryTest(unittest.TestCase):
    def test_pose_inverse_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = random_pose(rng)
            ident = geo.pose_mul(p, geo.pose_inv(p))
            np.testing.assert_allclose(ident[:3], 0, atol=1e-9)
            self.assertAlmostEqual(abs(ident[3]), 1.0, places=9)

    def test_world_base_roundtrip(self):
        rng = np.random.default_rng(1)
        base = (1.3, -0.7, 2.1)
        for _ in range(20):
            p = random_pose(rng)
            back = geo.base_to_world(geo.world_to_base(p, base), base)
            np.testing.assert_allclose(back[:3], p[:3], atol=1e-9)
            self.assertAlmostEqual(abs(np.dot(back[3:], p[3:])), 1.0, places=9)

    def test_world_to_base_known_case(self):
        # base at (1, 2) turned 90 deg: a point 1 m ahead of the base (world +y) has base x = 1
        p_world = np.array([1.0, 3.0, 0.5, 1, 0, 0, 0])
        p_base = geo.world_to_base(p_world, (1.0, 2.0, np.pi / 2))
        np.testing.assert_allclose(p_base[:3], [1.0, 0.0, 0.5], atol=1e-9)
        self.assertAlmostEqual(geo.quat_to_yaw(p_base[3:]), -np.pi / 2, places=9)

    def test_yaw_and_tilt(self):
        for yaw in (-3.0, -0.5, 0.0, 1.2, 3.0):
            q = geo.quat_from_yaw(yaw)
            self.assertAlmostEqual(geo.quat_to_yaw(q), yaw, places=9)
            self.assertAlmostEqual(geo.quat_tilt(q), 0.0, places=9)
        flipped = geo.quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), geo.quat_from_yaw(0.4))
        self.assertAlmostEqual(geo.quat_tilt(flipped), np.pi, places=9)


class CubeTiltTest(unittest.TestCase):
    def test_any_face_down_is_flat(self):
        for q in ([1, 0, 0, 0], [np.sqrt(0.5), np.sqrt(0.5), 0, 0], [np.sqrt(0.5), 0, np.sqrt(0.5), 0], [0, 1, 0, 0]):
            self.assertAlmostEqual(geo.cube_tilt(np.array(q, dtype=float)), 0.0, places=9)

    def test_edge_down_is_tilted_45(self):
        q = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])  # 45 deg about x
        self.assertAlmostEqual(geo.cube_tilt(q), np.pi / 4, places=9)


class GripperTest(unittest.TestCase):
    def test_angle_width_roundtrip(self):
        for w in (0.02, 0.05, 0.07):
            theta = gripper.finger_angle_for_width(w)
            self.assertAlmostEqual(gripper.pad_distance(theta) - gripper.PAD_THICKNESS, w, places=6)

    def test_narrower_objects_need_more_closing(self):
        self.assertGreater(gripper.finger_angle_for_width(0.03), gripper.finger_angle_for_width(0.06))

    def test_out_of_range_width_raises(self):
        with self.assertRaises(ValueError):
            gripper.finger_angle_for_width(0.12)

    def test_cube_closes_far_from_fully_closed(self):
        # the reason for this module: a 5 cm cube blocks the pads around 0.4 rad, not at the 0.75 rad "closed" command
        self.assertAlmostEqual(gripper.finger_angle_for_width(0.05), 0.39, delta=0.05)


class GraspTest(unittest.TestCase):
    def test_grasp_width_across_the_right_side(self):
        brick = Prim("brick", "box", (0.04, 0.065, 0.08), [0.6, 0, 0.5, *geo.quat_from_yaw(0.3)])
        along_x = grasps.topdown_quat(0.3)  # closing axis along the brick's x side
        along_y = grasps.topdown_quat(0.3 + np.pi / 2)
        self.assertAlmostEqual(grasps.grasp_width(brick, along_x), 0.04)
        self.assertAlmostEqual(grasps.grasp_width(brick, along_y), 0.065)
        can = Prim("can", "cylinder", (0.0275, 0.1), [0.6, 0, 0.5, 1, 0, 0, 0])
        self.assertAlmostEqual(grasps.grasp_width(can, along_x), 0.055)


    def test_topdown_orientation_is_ready_pose_at_zero_yaw(self):
        np.testing.assert_allclose(grasps.topdown_quat(0.0), names.READY_TCP_QUAT, atol=1e-12)

    def test_tool_axes(self):
        for yaw in (0.0, 0.7, -2.0):
            q = grasps.topdown_quat(yaw)
            approach = geo.quat_rotate(q, np.array([0.0, 0.0, 1.0]))
            closing = geo.quat_rotate(q, np.array([1.0, 0.0, 0.0]))
            np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)  # tool z points down
            np.testing.assert_allclose(closing, [np.cos(yaw), np.sin(yaw), 0], atol=1e-9)

    def test_cube_candidates(self):
        cube = ps.OBJECTS[0].prim(0.65, 0.1, 0.4)
        c = grasps.topdown_grasps(cube, ps.TABLE_TOP_Z)
        self.assertEqual(len(c), 4)  # 2 closing axes x both directions
        np.testing.assert_allclose(c[:, :2], [[0.65, 0.1]] * 4)
        # the closing axis lines up with a cube side
        for pose in c:
            axis = geo.quat_rotate(pose[3:], np.array([1.0, 0.0, 0.0]))
            rel = geo.wrap_angle(np.arctan2(axis[1], axis[0]) - 0.4)
            self.assertAlmostEqual(abs(np.sin(2 * rel)), 0.0, places=6)

    def test_grasp_height_limits(self):
        cube = ps.OBJECTS[0].prim(0.65, 0.0, 0.0)
        top = cube.pose[2] + cube.height / 2
        for pose in grasps.topdown_grasps(cube, ps.TABLE_TOP_Z):
            z = pose[2]
            self.assertGreaterEqual(z, ps.TABLE_TOP_Z + grasps.TIP_CLEARANCE - 1e-9)  # fingertips clear the table
            self.assertGreaterEqual(z, top - grasps.PALM_OFFSET + grasps.PALM_CLEARANCE - 1e-9)  # palm clears the top
            self.assertLessEqual(z, top - grasps.PAD_OVERLAP + 1e-9)  # pads still overlap the cube

    def test_too_wide_object_has_no_grasp(self):
        wide = Prim("wide", "box", (0.12, 0.12, 0.05), [0.6, 0, 0.475, 1, 0, 0, 0])
        self.assertEqual(len(grasps.topdown_grasps(wide, ps.TABLE_TOP_Z)), 0)

    def test_cylinder_candidates(self):
        can = Prim("can", "cylinder", (0.0275, 0.1), [0.6, 0, 0.5, 1, 0, 0, 0])
        self.assertEqual(len(grasps.topdown_grasps(can, ps.TABLE_TOP_Z, num_yaw=8)), 8)

    def test_place_puts_object_on_target(self):
        obj_in_tcp = np.array([-0.011, 0.002, 0.013, *geo.quat_from_yaw(0.1)])  # slightly off-centre in the hand
        target = (0.7, -0.2)
        poses = grasps.place_candidates(obj_in_tcp, target, rest_z=0.477)
        self.assertEqual(len(poses), 8)
        for tcp in poses:
            obj = geo.pose_mul(tcp, obj_in_tcp)
            np.testing.assert_allclose(obj[:3], [*target, 0.477], atol=1e-9)
            self.assertAlmostEqual(geo.quat_tilt(obj[3:]), 0.0, places=9)

    def test_place_after_flatten_keeps_the_tool_pointing_down(self):
        # TCP top-down at the grasp; the cube tilted ~11 deg in the grip
        tcp_w = np.array([0.6, 0.0, 0.5, *grasps.topdown_quat(0.3)])
        tilt = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])
        cube_w = np.array([0.6, 0.0, 0.48, *geo.quat_mul(geo.quat_from_yaw(0.3), tilt)])
        in_hand = geo.pose_mul(geo.pose_inv(tcp_w), grasps.flatten(cube_w))
        for tcp in grasps.place_candidates(in_hand, (0.7, -0.2), rest_z=0.477):
            np.testing.assert_allclose(geo.quat_rotate(tcp[3:], np.array([0.0, 0.0, 1.0])), [0, 0, -1], atol=1e-9)
            np.testing.assert_allclose(geo.pose_mul(tcp, in_hand)[:3], [0.7, -0.2, 0.477], atol=1e-9)

    def test_flatten_keeps_position_and_heading(self):
        q = geo.quat_mul(geo.quat_from_yaw(0.8), np.array([np.cos(0.15), np.sin(0.15), 0.0, 0.0]))
        flat = grasps.flatten(np.array([0.1, 0.2, 0.3, *q]))
        np.testing.assert_allclose(flat[:3], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(geo.quat_tilt(flat[3:]), 0.0, places=9)
        self.assertAlmostEqual(geo.quat_to_yaw(flat[3:]), geo.quat_to_yaw(q), places=9)


class WorldTest(unittest.TestCase):
    def test_scene_dict_is_in_base_frame(self):
        base = (1.0, 2.0, np.pi / 2)
        cube = ps.OBJECTS[0].prim(1.0, 3.0, np.pi / 2)  # 1 m ahead of the base, same heading
        scene = prims_to_scene_dict([cube], base, margin=0.01)["cuboid"][cube.name]
        np.testing.assert_allclose(scene["pose"][:3], [1.0, 0.0, cube.pose[2]], atol=1e-9)
        np.testing.assert_allclose(scene["dims"], [0.07, 0.07, 0.07])
        self.assertAlmostEqual(abs(scene["pose"][3]), 1.0, places=9)  # same heading as the base: identity rotation

    def test_cylinder_becomes_bounding_cuboid(self):
        can = Prim("can", "cylinder", (0.03, 0.1), [0.6, 0, 0.5, 1, 0, 0, 0])
        dims = prims_to_scene_dict([can], (0, 0, 0))["cuboid"]["can"]["dims"]
        np.testing.assert_allclose(dims, [0.06, 0.06, 0.1])

    def test_prim_validation(self):
        with self.assertRaises(ValueError):
            Prim("x", "sphere", (0.1,))
        with self.assertRaises(ValueError):
            Prim("x", "box", (0.1, 0.1))


class ScenarioTest(unittest.TestCase):
    def test_nothing_overlaps_and_target_is_far(self):
        for seed in range(100):
            sc = ps.sample_scenario(np.random.default_rng(seed))
            items = [(np.array(v[:2]), ps.footprint_radius(ps.SPECS[k])) for k, v in sc.poses.items()]
            for i, (a, ra) in enumerate(items):
                for b, rb in items[i + 1 :]:
                    self.assertGreaterEqual(np.linalg.norm(a - b), ra + rb, f"seed {seed}")
            self.assertGreaterEqual(
                np.linalg.norm(np.array(sc.poses[sc.pick][:2]) - np.array(sc.target_xy)), ps.MIN_TARGET_GAP - 1e-9
            )

    def test_wall_between_object_and_target(self):
        sc = ps.sample_scenario(np.random.default_rng(5), wall_on_path=True)
        a, b, w = np.array(sc.poses[sc.pick][:2]), np.array(sc.target_xy), np.array(sc.poses["wall"][:2])
        np.testing.assert_allclose(w, (a + b) / 2, atol=1e-9)

    def test_everything_inside_the_zone_and_table(self):
        for seed in range(50):
            sc = ps.sample_scenario(np.random.default_rng(seed))
            for name in (o.name for o in ps.OBJECTS):
                x, y, _ = sc.poses[name]
                self.assertTrue(ps.ZONE_X[0] <= x <= ps.ZONE_X[1] and ps.ZONE_Y[0] <= y <= ps.ZONE_Y[1])
            x, y = sc.target_xy
            self.assertTrue(ps.ZONE_X[0] <= x <= ps.ZONE_X[1] and ps.ZONE_Y[0] <= y <= ps.ZONE_Y[1])

    def test_objects_fit_the_gripper(self):
        for spec in ps.OBJECTS:
            self.assertLessEqual(min(spec.size[:2]), grasps.MAX_GRASP_WIDTH)


class PlanningUrdfTest(unittest.TestCase):
    """steve_arm_planning.urdf must be exactly what tools/make_arm_planning_urdf.py derives from steve_minimal."""

    @classmethod
    def setUpClass(cls):
        import make_arm_planning_urdf as tool

        cls.tool = tool
        cls.robot = ET.parse(ARM_PLANNING_URDF_PATH).getroot()

    def test_matches_generator(self):
        derived = self.tool.derive(URDF_PATH)
        ET.indent(derived, space="  ")
        regenerated = ET.tostring(derived.getroot(), encoding="unicode")
        on_disk = ET.tostring(self.robot, encoding="unicode")
        self.assertEqual(regenerated, on_disk, "regenerate with tools/make_arm_planning_urdf.py")

    def test_only_the_arm_moves(self):
        movable = {j.get("name") for j in self.robot.findall("joint") if j.get("type") != "fixed"}
        self.assertEqual(movable, set(names.ARM_JOINTS))

    def test_no_wheels_or_casters(self):
        links = {l.get("name") for l in self.robot.findall("link")}
        self.assertFalse(links & self.tool.REMOVED_LINKS)
        self.assertEqual(links, set(names.LINKS) - self.tool.REMOVED_LINKS)

    def test_root_is_base_link(self):
        children = {j.find("child").get("link") for j in self.robot.findall("joint")}
        roots = {l.get("name") for l in self.robot.findall("link")} - children
        self.assertEqual(roots, {names.BASE_LINK})


if __name__ == "__main__":
    unittest.main()
