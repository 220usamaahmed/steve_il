"""Candidate TCP poses for picking and placing objects with the Robotiq 2F-85 (analytic, no learning).

Gripper facts (from ``steve_arm_planning.urdf``): the fingers close along the tool **x** axis, the
approach direction is the tool **z** axis, and the open pads are 0.0924 m apart (0.086 m usable). A
top-down grasp therefore has tool z pointing down (``READY_TCP_QUAT`` is such an orientation) and
tool x horizontal, along the line the fingers close on.

All poses are TCP poses in the *world* frame; the caller converts them to the base frame.
"""

from __future__ import annotations

import numpy as np

from steve_il.planning import geometry as geo
from steve_il.planning.world import Prim

MAX_GRASP_WIDTH = 0.070  # [m] widest object we try to grasp (gripper opens to ~0.086 m)
TIP_CLEARANCE = 0.030  # [m] minimum TCP height above a support surface (fingertips reach ~0.02 m below the TCP)
PALM_OFFSET = 0.038  # [m] the palm's collision spheres reach this far above the TCP (opposite to the approach direction)
PALM_CLEARANCE = 0.015  # [m] wanted gap between the palm and the object's top
PAD_OVERLAP = 0.012  # [m] the pads must reach at least this far down from the object's top
Q_FLIP_X = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x: tool z -> -z (pointing down)


def topdown_quat(closing_yaw: float) -> np.ndarray:
    """TCP orientation with tool z pointing down and the closing axis (tool x) at ``closing_yaw`` in the world."""
    return geo.quat_mul(geo.quat_from_yaw(closing_yaw), Q_FLIP_X)


def closing_yaw_of(tcp_quat: np.ndarray) -> float:
    """Heading of the tool x axis (the finger closing line) in the frame ``tcp_quat`` is given in."""
    return geo.quat_to_yaw(tcp_quat)


def topdown_grasps(obj: Prim, support_z: float, num_yaw: int = 8) -> np.ndarray:
    """Top-down grasp candidates [N, 7] (world frame TCP poses) for an upright box or cylinder.

    * box: close across either horizontal side that fits the gripper, from both directions
    * cylinder: ``num_yaw`` evenly spaced closing directions

    The TCP sits at the object's mid-height, moved up if the palm would otherwise touch the top of a short
    object (``PALM_*``) or the fingertips the support (``TIP_CLEARANCE``), and never higher than the pads
    still overlap the object (``PAD_OVERLAP``). Returns an empty array if the object does not fit the gripper.
    """
    centre = obj.pose[:3]
    top = centre[2] + obj.height / 2
    z_lo = max(support_z + TIP_CLEARANCE, top - PALM_OFFSET + PALM_CLEARANCE)
    z_hi = top - PAD_OVERLAP
    z = float(min(max(centre[2], z_lo), max(z_lo, z_hi)))
    yaw0 = obj.yaw
    if obj.kind == "box":
        sx, sy, _ = obj.size
        yaws = []
        if sx <= MAX_GRASP_WIDTH:
            yaws += [yaw0, yaw0 + np.pi]
        if sy <= MAX_GRASP_WIDTH:
            yaws += [yaw0 + np.pi / 2, yaw0 + 3 * np.pi / 2]
    else:
        if 2 * obj.size[0] > MAX_GRASP_WIDTH:
            return np.zeros((0, 7))
        yaws = [yaw0 + 2 * np.pi * k / num_yaw for k in range(num_yaw)]
    return np.array([[centre[0], centre[1], z, *topdown_quat(geo.wrap_angle(y))] for y in yaws]).reshape(-1, 7)


def place_candidates(
    obj_in_tcp: np.ndarray, target_xy: tuple[float, float], rest_z: float, num_yaw: int = 8
) -> np.ndarray:
    """TCP poses [N, 7] (world frame) that put the held object upright at ``target_xy`` on a surface.

    ``obj_in_tcp`` is the measured pose of the object in the TCP frame (so the offset in the hand is
    accounted for); ``rest_z`` is the object-centre height when it rests on the surface. The heading
    of the object at the target takes ``num_yaw`` values; its tilt is zero.

    The tool orientation follows the object's orientation in the hand. If the object tilted in the grip
    and the tool should stay upright anyway (a cube lands flat when released), pass ``obj_in_tcp``
    computed from the object's *flattened* orientation (see ``flatten``).
    """
    out = []
    for k in range(num_yaw):
        obj_w = np.concatenate([[target_xy[0], target_xy[1], rest_z], geo.quat_from_yaw(2 * np.pi * k / num_yaw)])
        out.append(geo.pose_mul(obj_w, geo.pose_inv(obj_in_tcp)))
    return np.array(out).reshape(-1, 7)


def flatten(pose: np.ndarray) -> np.ndarray:
    """The same position with the orientation reduced to its heading (no tilt)."""
    return np.concatenate([pose[:3], geo.quat_from_yaw(geo.quat_to_yaw(pose[3:]))])


def grasp_width(obj: Prim, tcp_quat_world: np.ndarray) -> float:
    """Distance between the fingers' contact points [m] when ``obj`` is grasped with ``tcp_quat_world``."""
    if obj.kind == "cylinder":
        return 2 * obj.size[0]
    axis = geo.quat_rotate(tcp_quat_world, np.array([1.0, 0.0, 0.0]))  # finger closing direction
    local = geo.wrap_angle(np.arctan2(axis[1], axis[0]) - obj.yaw)
    return obj.size[0] if abs(np.cos(local)) >= abs(np.sin(local)) else obj.size[1]
