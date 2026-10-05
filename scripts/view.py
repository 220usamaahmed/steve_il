"""Open Steve in the Isaac Sim viewer (GUI).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/view.py            # static, close the window to quit
    OMNI_KIT_ACCEPT_EULA=YES python scripts/view.py --demo     # gripper + arm joints move slowly
    ... python scripts/view.py --screenshot data/view.png      # save the viewport, then exit

Orbit with Alt+LMB, pan with Alt+MMB, zoom with the wheel. The robot is at /World/Steve.
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--demo", action="store_true", help="slowly move the gripper and arm joints")
parser.add_argument("--screenshot", type=Path, default=None, help="save the viewport to this PNG and exit")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from steve_il.robots import steve_names as n  # noqa: E402
from steve_il.robots.steve_cfg import STEVE_CFG  # noqa: E402


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view(eye=(2.4, -2.4, 1.7), target=(0.0, 0.0, 0.6))

    ground, light = sim_utils.GroundPlaneCfg(), sim_utils.DomeLightCfg(intensity=2500.0)
    ground.func("/World/ground", ground)
    light.func("/World/light", light)
    robot = Articulation(STEVE_CFG.replace(prim_path="/World/Steve"))
    sim.reset()

    dt = sim.get_physics_dt()
    names = robot.joint_names
    home = robot.data.default_joint_pos.clone()
    pan, finger = names.index(n.ARM_JOINTS[0]), names.index(n.GRIPPER_DRIVEN_JOINT)
    t, frames = 0.0, 0
    while simulation_app.is_running():
        target = home.clone()
        if args.demo:
            target[:, pan] += 0.6 * math.sin(0.5 * t)
            target[:, finger] = 0.4 * (1 - math.cos(0.8 * t))  # 0 .. 0.8 rad
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        t += dt
        frames += 1
        if args.screenshot and frames == 240:  # let it settle and the renderer warm up
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            capture_viewport_to_file(get_active_viewport(), str(args.screenshot.resolve()))
        if args.screenshot and frames == 300:
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # SimulationApp.close() can hang on shutdown; nothing to clean up here
