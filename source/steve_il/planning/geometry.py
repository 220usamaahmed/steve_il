"""Small pose algebra on numpy arrays (no torch, no Isaac, no cuRobo).

A pose is a length-7 array ``[x, y, z, qw, qx, qy, qz]`` (unit quaternion, wxyz like Isaac Lab and
cuRobo). ``base`` means the ``base_link`` frame, ``world`` the environment frame; the base pose is
``(x, y, yaw)`` in the world, with ``base_link`` at z = 0 (see ``tools/make_holonomic_urdf.py``).
"""

from __future__ import annotations

import numpy as np

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b of (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q``."""
    qv = np.array([0.0, *v])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_to_yaw(q: np.ndarray) -> float:
    """Heading of a quaternion (rotation about z of its x axis)."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quat_tilt(q: np.ndarray) -> float:
    """Angle [rad] between the body z axis and the world z axis."""
    v = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    return float(np.arctan2(np.hypot(v[0], v[1]), v[2]))


def cube_tilt(q: np.ndarray) -> float:
    """Tilt [rad] of a cube: how far it is from lying flat on *any* of its faces (0 ... pi/4)."""
    return min(
        float(np.arctan2(np.hypot(*v[:2]), abs(v[2])))
        for v in (quat_rotate(q, e) for e in np.eye(3))
    )


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def pose_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose poses: the pose of frame B (given in A) expressed in A's parent frame."""
    pos = a[:3] + quat_rotate(a[3:], b[:3])
    quat = quat_mul(a[3:], b[3:])
    return np.concatenate([pos, quat / np.linalg.norm(quat)])


def pose_inv(p: np.ndarray) -> np.ndarray:
    qi = quat_conj(p[3:])
    return np.concatenate([-quat_rotate(qi, p[:3]), qi])


def base_pose_to_transform(base_xy_yaw) -> np.ndarray:
    """``base_link`` pose in the world as a 7-vector, from (x, y, yaw)."""
    x, y, yaw = (float(v) for v in base_xy_yaw)
    return np.concatenate([[x, y, 0.0], quat_from_yaw(yaw)])


def world_to_base(pose_w: np.ndarray, base_xy_yaw) -> np.ndarray:
    """Express a world-frame pose in the base_link frame."""
    return pose_mul(pose_inv(base_pose_to_transform(base_xy_yaw)), pose_w)


def base_to_world(pose_b: np.ndarray, base_xy_yaw) -> np.ndarray:
    """Express a base_link-frame pose in the world frame."""
    return pose_mul(base_pose_to_transform(base_xy_yaw), pose_b)
