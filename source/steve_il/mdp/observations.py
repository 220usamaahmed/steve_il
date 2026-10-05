"""Observation functions for the holonomic Steve robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from steve_il.robots import steve_names as n

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def tcp_pose_in_base(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tool_body: str = n.EE_LINK,
    tcp_offset_z: float = n.TCP_OFFSET_Z,
    base_body: str = n.BASE_LINK,
) -> torch.Tensor:
    """Tool-centre-point pose in the base_link frame: [x, y, z, qw, qx, qy, qz]."""
    robot = env.scene[asset_cfg.name]
    tool_id = robot.find_bodies(tool_body)[0][0]
    base_id = robot.find_bodies(base_body)[0][0]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, base_id],
        robot.data.body_quat_w[:, base_id],
        robot.data.body_pos_w[:, tool_id],
        robot.data.body_quat_w[:, tool_id],
    )
    offset = torch.zeros_like(pos_b)
    offset[:, 2] = tcp_offset_z
    identity = torch.zeros_like(quat_b)
    identity[:, 0] = 1.0
    pos_b, quat_b = math_utils.combine_frame_transforms(pos_b, quat_b, offset, identity)
    return torch.cat([pos_b, quat_b], dim=-1)


def base_pose_xy_yaw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base pose in the environment frame [x, y, yaw] (read from the virtual joints)."""
    robot = env.scene[asset_cfg.name]
    ids = robot.find_joints(n.BASE_PLANAR_JOINTS, preserve_order=True)[0]
    return robot.data.joint_pos[:, ids]


def base_velocity_body(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base twist in the body frame [vx, vy, wz]."""
    robot = env.scene[asset_cfg.name]
    ids = robot.find_joints(n.BASE_PLANAR_JOINTS, preserve_order=True)[0]
    yaw = robot.data.joint_pos[:, ids[2]]
    vx_w, vy_w, wz = robot.data.joint_vel[:, ids].unbind(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * vx_w + s * vy_w, -s * vx_w + c * vy_w, wz], dim=-1)
