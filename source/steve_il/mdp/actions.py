"""Action terms for the holonomic Steve robot.

Action vector layout used by the environments (all terms in this order, dim 11):

    [ base: vx vy wz | arm: x y z qw qx qy qz | gripper ]

* base     body-frame velocity, normalised to [-1, 1] and scaled to physical units by ``scale``;
           the applied velocity ramps towards the command with a limited acceleration.
* arm      *absolute* tool-centre-point pose in the **base_link frame**; tracked with
           differential IK (the pose does not change when the base moves or turns).
* gripper  Isaac Lab binary action: value >= 0 -> open, value < 0 -> close.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


# ----------------------------------------------------------------------------------------------
# Holonomic base
# ----------------------------------------------------------------------------------------------
class HolonomicBaseVelocityAction(ActionTerm):
    """Body-frame planar velocity command -> velocity targets of the virtual x / y / yaw joints.

    The prismatic joints move along the anchor (environment) axes, so the commanded body-frame
    ``(vx, vy)`` is rotated by the current yaw every physics step. ``yaw`` is read from the yaw
    joint, which is exactly the base heading.

    The velocity that is actually applied ramps towards the command with ``max_lin_acc`` /
    ``max_ang_acc`` (per axis, in the body frame, updated every physics step). Without this the
    virtual drive reaches full speed within one control step (~25 m/s^2), which no real base does
    and which jolts the arm.
    """

    cfg: HolonomicBaseVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: HolonomicBaseVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(ids) != 3:
            raise ValueError(f"expected 3 joints (x, y, yaw), got {names}")
        self._joint_ids = ids
        self._yaw_id = ids[2]
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.tensor(cfg.scale, device=self.device).repeat(self.num_envs, 1)
        self._vel_target = torch.zeros_like(self._raw_actions)
        self._applied = torch.zeros_like(self._raw_actions)  # body-frame twist currently commanded to the drive
        self._max_step = torch.tensor(
            [cfg.max_lin_acc, cfg.max_lin_acc, cfg.max_ang_acc], device=self.device
        ) * env.physics_dt

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Body-frame twist [vx (m/s), vy (m/s), wz (rad/s)] after clipping and scaling."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0) * self._scale

    def apply_actions(self):
        # ramp the applied body-frame twist towards the command (acceleration limit)
        delta = self._processed_actions - self._applied
        self._applied += torch.clamp(delta, -self._max_step, self._max_step)

        yaw = self._asset.data.joint_pos[:, self._yaw_id]
        c, s = torch.cos(yaw), torch.sin(yaw)
        vx, vy, wz = self._applied.unbind(-1)
        self._vel_target[:, 0] = c * vx - s * vy
        self._vel_target[:, 1] = s * vx + c * vy
        self._vel_target[:, 2] = wz
        self._asset.set_joint_velocity_target(self._vel_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._applied[env_ids] = 0.0


@configclass
class HolonomicBaseVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HolonomicBaseVelocityAction

    joint_names: list[str] = MISSING
    """The three virtual joints in the order [x, y, yaw]."""
    scale: tuple[float, float, float] = (0.8, 0.8, 1.0)
    """Physical value of a normalised action of 1.0: [m/s, m/s, rad/s]."""
    max_lin_acc: float = 2.0
    """Linear acceleration limit per body axis [m/s^2]."""
    max_ang_acc: float = 3.0
    """Angular acceleration limit [rad/s^2]."""


# ----------------------------------------------------------------------------------------------
# Arm IK in the base frame
# ----------------------------------------------------------------------------------------------
class BaseFrameDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Differential IK whose command / feedback frame is ``base_link`` instead of the articulation root.

    With the virtual-joint base the articulation root is the world-fixed anchor, so the stock
    term would express poses in the world and ignore where the robot stands. Here the end-effector
    pose is measured relative to the (moving) ``base_link`` body and the Jacobian is rotated into
    that frame, so a command means "TCP pose relative to the robot" regardless of base motion.
    """

    cfg: BaseFrameDifferentialIKActionCfg

    def __init__(self, cfg: BaseFrameDifferentialIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        ids, names = self._asset.find_bodies(cfg.base_body_name)
        if len(ids) != 1:
            raise ValueError(f"expected one body matching {cfg.base_body_name!r}, got {names}")
        self._base_body_idx = ids[0]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.body_quat_w[:, self._base_body_idx]
        rot_inv = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(rot_inv, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(rot_inv, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        base_pos_w = self._asset.data.body_pos_w[:, self._base_body_idx]
        base_quat_w = self._asset.data.body_quat_w[:, self._base_body_idx]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
        if self.cfg.body_offset is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
            )
        return ee_pos_b, ee_quat_b


@configclass
class BaseFrameDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = BaseFrameDifferentialIKAction

    base_body_name: str = "base_link"
    """Body whose frame the commands and the Jacobian are expressed in."""
