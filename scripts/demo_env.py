"""Play a looping demo of the Steve environment (base + arm + gripper) in the viewer.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/demo_env.py                      # GUI, loops until closed
    OMNI_KIT_ACCEPT_EULA=YES python scripts/demo_env.py --screenshot data/demo.png   # capture and exit

Orbit with Alt+LMB, pan with Alt+MMB, zoom with the wheel.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--screenshot", type=Path, default=None, help="save the viewport to this PNG and exit")
parser.add_argument("--screenshot-at", type=float, default=9.0, help="sim seconds into the demo")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from steve_il.envs.steve_env_cfg import SteveEnvCfg  # noqa: E402
from steve_il.robots import steve_names as n  # noqa: E402
from steve_il.scripted import motion as mo  # noqa: E402

READY = n.READY_TCP_POS
# (seconds, base command [vx, vy, wz] normalised, TCP position in base frame, gripper)
SEQUENCE = [
    (2.0, (0.0, 0.0, 0.0), (0.60, 0.00, 0.55), mo.GRIP_OPEN),   # reach down in front
    (1.0, (0.0, 0.0, 0.0), (0.60, 0.00, 0.55), mo.GRIP_CLOSE),  # close
    (2.0, (0.0, 0.0, 0.0), (0.55, 0.25, 0.75), mo.GRIP_CLOSE),  # lift to the left
    (3.0, (0.6, 0.0, 0.5), (0.55, 0.25, 0.75), mo.GRIP_CLOSE),  # drive an arc, arm holds its pose
    (2.0, (0.0, 0.0, 0.0), (0.55, -0.25, 0.75), mo.GRIP_OPEN),  # swing right, release
    (3.0, (-0.6, 0.0, -0.5), READY, mo.GRIP_OPEN),               # drive back, return to ready
]


def main() -> None:
    cfg = SteveEnvCfg()
    cfg.episode_length_s = 600.0
    env = ManagerBasedRLEnv(cfg)
    obs, _ = env.reset()
    dt = env.step_dt
    quat = torch.tensor(n.READY_TCP_QUAT, device=env.device)
    tcp_cmd = obs["policy"]["tcp_pose"][0].clone()
    t, shot_done = 0.0, False
    while simulation_app.is_running():
        for seconds, base, pos, grip in SEQUENCE:
            goal = torch.cat([torch.tensor(pos, device=env.device, dtype=torch.float32), quat])
            ramp = min(seconds, 1.5)  # arm eases in over the first part of the segment
            steps = round(seconds / dt)
            wps = list(mo.tcp_waypoints(tcp_cmd, goal, round(ramp / dt)))
            for i in range(steps):
                tcp_cmd = wps[i] if i < len(wps) else goal
                obs, *_ = env.step(mo.make_action(tcp_cmd, base, grip))
                t += dt
                if args.screenshot and not shot_done and t >= args.screenshot_at:
                    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    capture_viewport_to_file(get_active_viewport(), str(args.screenshot.resolve()))
                    shot_done = True
                if shot_done and t >= args.screenshot_at + 1.0:
                    return
                if not simulation_app.is_running():
                    return


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # Kit's close() can hang in GUI mode
