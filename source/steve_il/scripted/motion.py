"""Helpers to build env actions and smooth TCP motions (shared by tests and scripted demos).

Action layout (dim 11): [vx vy wz | x y z qw qx qy qz | gripper], see ``steve_il.mdp.actions``.
Every pose is expressed in the base_link frame.
"""

from __future__ import annotations

import torch

BASE = slice(0, 3)
TCP_POS = slice(3, 6)
TCP_QUAT = slice(6, 10)
TCP_POSE = slice(3, 10)
GRIPPER = 10
ACTION_DIM = 11

GRIP_OPEN = 1.0  # Isaac Lab binary action: >= 0 open
GRIP_CLOSE = -1.0  # < 0 close


def make_action(
    tcp_pose: torch.Tensor,
    base: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gripper: float = GRIP_OPEN,
) -> torch.Tensor:
    """Action for one env from a [7] TCP pose tensor (device taken from it)."""
    act = torch.zeros(1, ACTION_DIM, device=tcp_pose.device)
    act[0, BASE] = torch.tensor(base, device=tcp_pose.device)
    act[0, TCP_POSE] = tcp_pose.reshape(-1)[:7]
    act[0, GRIPPER] = gripper
    return act


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation of two (w, x, y, z) quaternions (shortest path)."""
    dot = (q0 * q1).sum()
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel: lerp
        q = q0 + t * (q1 - q0)
        return q / q.norm()
    theta = torch.acos(dot.clamp(-1.0, 1.0))
    return (torch.sin((1 - t) * theta) * q0 + torch.sin(t * theta) * q1) / torch.sin(theta)


def quat_angle(q0: torch.Tensor, q1: torch.Tensor) -> float:
    """Rotation angle [rad] between two (w, x, y, z) quaternions."""
    return float(2.0 * torch.acos((q0 * q1).sum().abs().clamp(max=1.0)))


def smoothstep(t: float) -> float:
    """C1 ease-in/out on [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def tcp_waypoints(start: torch.Tensor, goal: torch.Tensor, steps: int):
    """Yield ``steps`` TCP poses ([7] tensors) easing from ``start`` to ``goal`` (pos lerp, quat slerp)."""
    for i in range(1, steps + 1):
        s = smoothstep(i / steps)
        pos = start[:3] + s * (goal[:3] - start[:3])
        quat = quat_slerp(start[3:7], goal[3:7], s)
        yield torch.cat([pos, quat])
