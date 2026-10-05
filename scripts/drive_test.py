"""Exercise every controller of the Steve environment and check the measured behaviour.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/drive_test.py --headless   # pass/fail, exit code
    OMNI_KIT_ACCEPT_EULA=YES python scripts/drive_test.py              # same, watch it in the viewer

Sections: A reset state / TCP offset, B base velocity (body frame), C arm IK in the base frame,
D frame invariance (arm pose holds while the base drives), E gripper.
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from steve_il.envs.steve_env_cfg import GRIPPER_CLOSED, SteveEnvCfg  # noqa: E402
from steve_il.robots import steve_names as n  # noqa: E402
from steve_il.scripted import motion as mo  # noqa: E402

FAILURES: list[str] = []


def report(ok: bool, what: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {what}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(what)


class Runner:
    """Holds the env and the latest observation; every step re-sends the full action vector."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.dt = env.step_dt
        self.obs, _ = env.reset()
        self.tcp_cmd = self.tcp.clone()  # hold the current pose unless told otherwise
        self.grip = mo.GRIP_OPEN

    @property
    def pol(self):
        return self.obs["policy"]

    @property
    def tcp(self) -> torch.Tensor:
        return self.pol["tcp_pose"][0].clone()

    @property
    def base(self) -> torch.Tensor:
        return self.pol["base_pose"][0].clone()  # x, y, yaw

    def step(self, base=(0.0, 0.0, 0.0), tcp=None, grip=None) -> None:
        if tcp is not None:
            self.tcp_cmd = tcp
        if grip is not None:
            self.grip = grip
        self.obs, *_ = self.env.step(mo.make_action(self.tcp_cmd, base, self.grip))

    def run(self, seconds: float, base=(0.0, 0.0, 0.0), **kw) -> None:
        for _ in range(round(seconds / self.dt)):
            self.step(base, **kw)

    def move_tcp(self, goal: torch.Tensor, seconds: float, settle: float = 0.7, base=(0.0, 0.0, 0.0)) -> None:
        for wp in mo.tcp_waypoints(self.tcp_cmd, goal, round(seconds / self.dt)):
            self.step(base, tcp=wp)
        self.run(settle, base)


def pose(pos, quat=n.READY_TCP_QUAT, device="cuda:0") -> torch.Tensor:
    return torch.tensor([*pos, *quat], device=device, dtype=torch.float32)


def section_a(r: Runner) -> None:
    print("\n== A. reset state", flush=True)
    robot = r.env.scene["robot"]
    ready = pose(n.READY_TCP_POS, device=r.env.device)
    perr = (r.tcp[:3] - ready[:3]).norm().item()
    oerr = mo.quat_angle(r.tcp[3:], ready[3:])
    report(perr < 0.003 and oerr < 0.01, "reset TCP == READY pose", f"pos {perr * 1000:.2f} mm, rot {oerr:.4f} rad")
    report(r.env.action_manager.action_term_dim == [3, 7, 1], "action layout [3, 7, 1]",
           str(r.env.action_manager.action_term_dim))
    report(robot.is_fixed_base and robot.num_joints == 23, "fixed-base articulation, 23 DOFs",
           f"fixed={robot.is_fixed_base} dofs={robot.num_joints}")
    names = robot.body_names
    tool = names.index(n.EE_LINK)
    pads = [names.index(p) for p in n.FINGER_PAD_LINKS]
    mid = (robot.data.body_pos_w[0, pads[0]] + robot.data.body_pos_w[0, pads[1]]) / 2
    rel = math_utils.quat_apply_inverse(robot.data.body_quat_w[0, tool], mid - robot.data.body_pos_w[0, tool])
    err = (rel - torch.tensor([0, 0, n.TCP_OFFSET_Z], device=rel.device)).norm().item()
    report(err < 0.002, "TCP offset == finger-pad midpoint (tool0 frame)", f"error {err * 1000:.2f} mm")


def displacement(r: Runner, base_cmd, seconds, stop: float = 1.0):
    """Net base displacement of a command held for ``seconds`` plus the ramp-down until standstill.

    With symmetric acceleration limits the ramp-up lag (v^2/2a) is recovered during the ramp-down,
    so the total equals v * seconds.
    """
    b0 = r.base
    r.run(seconds, base_cmd)
    r.run(stop)
    return r.base - b0


def section_b(r: Runner) -> None:
    print("\n== B. base velocity (body frame)", flush=True)
    vmax, wmax = r.env.cfg.actions.base.scale[0], r.env.cfg.actions.base.scale[2]
    amax = r.env.cfg.actions.base.max_lin_acc
    T = 1.5

    # acceleration limit: speed 0.2 s after a full-speed command from rest is a*0.2
    r.run(0.2, (1.0, 0.0, 0.0))
    v02 = r.pol["base_vel"][0, 0].item()
    report(abs(v02 - amax * 0.2) < 0.1 * amax * 0.2, "acceleration limit honoured (ramp)",
           f"v(0.2 s) = {v02:.3f} m/s (exp {amax * 0.2:.3f} = a*t)")
    r.run(1.0)  # stop and settle (also counts towards later displacement baselines)

    d = displacement(r, (1.0, 0.0, 0.0), T)
    exp = vmax * T
    report(abs(d[0].item() - exp) < 0.03 * exp and abs(d[1].item()) < 0.01 and abs(d[2].item()) < 0.01,
           "forward: dx ~ v*T, no sideways/yaw drift",
           f"dx {d[0].item():.3f} (exp {exp:.3f}), dy {d[1].item():+.4f}, dyaw {d[2].item():+.4f}")

    d = displacement(r, (0.0, 1.0, 0.0), T)
    report(abs(d[1].item() - exp) < 0.03 * exp and abs(d[0].item()) < 0.01 and abs(d[2].item()) < 0.01,
           "strafe left: dy ~ v*T", f"dy {d[1].item():.3f} (exp {exp:.3f}), dx {d[0].item():+.4f}, dyaw {d[2].item():+.4f}")

    T_rot = (math.pi / 2) / wmax
    d = displacement(r, (0.0, 0.0, 1.0), T_rot)
    report(abs(d[2].item() - math.pi / 2) < 0.03 * math.pi / 2 and d[:2].norm().item() < 0.01,
           "rotate +90 deg in place", f"dyaw {d[2].item():.4f} (exp {math.pi / 2:.4f}), xy {d[:2].norm().item():.4f} m")

    # body-frame semantics: now facing ~ +y, a forward command must move along world +y
    b0 = r.base
    r.run(1.0, (0.5, 0.0, 0.0))
    r.run(1.0)
    dw = r.base - b0
    yaw = b0[2].item() + 0.0
    exp_dir = torch.tensor([math.cos(yaw), math.sin(yaw)])
    got_dir = dw[:2].cpu() / dw[:2].norm().cpu()
    report(float((exp_dir * got_dir).sum()) > 0.999,
           "forward after rotation moves along the new heading",
           f"heading {math.degrees(yaw):.1f} deg, motion dir {math.degrees(math.atan2(dw[1].item(), dw[0].item())):.1f} deg")

    # simultaneous translation + rotation: speed in the body frame must match the command
    r.run(1.5, (1.0, 0.0, 1.0))  # accelerate to full speed
    bv = r.pol["base_vel"][0]
    report(abs(bv[0].item() - vmax) < 0.03 and abs(bv[1].item()) < 0.03 and abs(bv[2].item() - wmax) < 0.03,
           "arc motion: body-frame twist == command",
           f"v=({bv[0].item():.3f}, {bv[1].item():+.3f}) w={bv[2].item():.3f}; cmd ({vmax}, 0, {wmax})")

    # saturation (normalised action clipped to [-1, 1]) and stop
    r.run(1.5, (5.0, -5.0, 5.0))
    bv = r.pol["base_vel"][0]
    report(bv[:2].norm().item() < vmax * math.sqrt(2) * 1.03 and abs(bv[2].item()) < wmax * 1.03,
           "oversized command is clipped", f"|v|={bv[:2].norm().item():.3f} (max {vmax * math.sqrt(2):.3f}), w={bv[2].item():.3f}")
    r.run(1.0)
    bv = r.pol["base_vel"][0].abs().max().item()
    report(bv < 0.005, "base stops when command is zero", f"max |twist| {bv:.5f}")


def section_c(r: Runner) -> None:
    print("\n== C. arm IK in the base frame", flush=True)
    dev = r.env.device
    targets = {
        "front-low": pose((0.60, 0.00, 0.55), device=dev),
        "left": pose((0.55, 0.25, 0.75), device=dev),
        "right": pose((0.55, -0.25, 0.75), device=dev),
        "far": pose((0.80, 0.00, 0.65), device=dev),
        "near": pose((0.45, 0.00, 0.65), device=dev),
        "ready": pose(n.READY_TCP_POS, device=dev),
    }
    for name, goal in targets.items():
        r.move_tcp(goal, seconds=2.0)
        perr = (r.tcp[:3] - goal[:3]).norm().item()
        oerr = mo.quat_angle(r.tcp[3:], goal[3:])
        q = r.pol["arm_joint_pos"][0]
        report(perr < 0.002 and oerr < 0.01, f"TCP reaches '{name}'",
               f"pos {perr * 1000:.2f} mm, rot {oerr:.4f} rad, max|q| {q.abs().max().item():.2f}")


def section_d(r: Runner) -> None:
    """The commanded TCP pose is relative to base_link, so base motion must not change it.

    Measured behaviour (see README): a few-mm transient when the base accelerates (finite joint
    stiffness), ~1 mm steady offset while turning and translating
    (centrifugal load), exactly 0 at rest. A wrong reference frame would instead give errors of
    the order of the base travel (metres), so the thresholds below separate the two cases.
    """
    print("\n== D. frame invariance: arm pose holds while the base drives", flush=True)
    dev = r.env.device
    r.move_tcp(pose((0.60, 0.15, 0.70), device=dev), seconds=1.5)
    hold = r.tcp
    robot = r.env.scene["robot"]
    ee = robot.body_names.index(n.EE_LINK)
    w0 = robot.data.body_pos_w[0, ee].clone()
    yaw0 = r.base[2].item()

    def deviation():
        return (r.tcp[:3] - hold[:3]).norm().item(), mo.quat_angle(r.tcp[3:], hold[3:])

    steps = round(3.0 / r.dt)
    peak_p = peak_o = steady_p = steady_o = 0.0
    for i in range(steps):
        r.step((1.0, 0.5, 1.0))  # translate and turn at once
        p, o = deviation()
        peak_p, peak_o = max(peak_p, p), max(peak_o, o)
        if i >= steps - round(1.0 / r.dt):  # last second: well past the start-up transient
            steady_p, steady_o = max(steady_p, p), max(steady_o, o)
    moved = (robot.data.body_pos_w[0, ee] - w0).norm().item()
    turned = abs(r.base[2].item() - yaw0)
    r.run(1.0)
    rest_p, rest_o = deviation()

    report(moved > 1.0 and turned > 2.5, "base really carried the arm through the world",
           f"TCP moved {moved:.2f} m, base turned {turned:.2f} rad")
    report(peak_p < 0.010 and peak_o < 0.03, "start-up transient stays small",
           f"peak {peak_p * 1000:.2f} mm / {peak_o:.4f} rad")
    report(steady_p < 0.002 and steady_o < 0.01, "steady-state TCP pose in base frame holds",
           f"{steady_p * 1000:.2f} mm / {steady_o:.4f} rad while turning + translating")
    report(rest_p < 0.0005 and rest_o < 0.002, "TCP pose exact again after the base stops",
           f"{rest_p * 1000:.3f} mm / {rest_o:.4f} rad")
    r.move_tcp(pose(n.READY_TCP_POS, device=dev), seconds=1.5)


def section_e(r: Runner) -> None:
    print("\n== E. gripper", flush=True)
    robot = r.env.scene["robot"]
    ids = [robot.joint_names.index(j) for j in n.GRIPPER_JOINTS]

    def settle(grip):
        r.run(1.5, grip=grip)
        return robot.data.joint_pos[0, ids]

    q = settle(mo.GRIP_CLOSE)
    drv = q[0].item()
    report(abs(drv - GRIPPER_CLOSED) < 0.03, "close: finger_joint reaches closed target", f"{drv:.3f} rad (cmd {GRIPPER_CLOSED})")
    errs = [abs(q[1 + i].item() - m * drv) for i, m in enumerate(n.GRIPPER_MIMIC_JOINTS.values())]
    report(max(errs) < 0.02, "close: mimic joints follow", f"max err {max(errs):.4f} rad")
    q = settle(mo.GRIP_OPEN)
    report(abs(q[0].item()) < 0.03, "open: finger_joint returns to 0", f"{q[0].item():.3f} rad")
    tcp_drift = (r.tcp[:3] - r.tcp_cmd[:3]).norm().item()
    report(tcp_drift < 0.002, "gripper motion does not disturb the TCP", f"{tcp_drift * 1000:.2f} mm")


def main() -> int:
    cfg = SteveEnvCfg()
    cfg.episode_length_s = 300.0  # the test must not hit the time-out reset
    env = ManagerBasedRLEnv(cfg)
    r = Runner(env)
    section_a(r)
    section_b(r)
    section_c(r)
    section_d(r)
    section_e(r)
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILED: ' + '; '.join(FAILURES)}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Kit's close() swallows the exit status
