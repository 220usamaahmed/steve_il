"""Scripted pick-and-place with cuRobo arm planning (stationary base).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/pick_place_demo.py                  # viewer, loops until closed
    OMNI_KIT_ACCEPT_EULA=YES python scripts/pick_place_demo.py --headless --episodes 10   # pass/fail, exit code
    ... --pick can --seed 3 --episodes 1                                         # one specific scenario

Each episode: random table layout (3 cubes, a wall between the picked object and the target, a tower),
cuRobo plans approach -> grasp -> lift, then (with the object attached) approach -> place -> retreat.
The first run compiles CUDA kernels (~1 min); later runs use the cache.
Orbit with Alt+LMB, pan with Alt+MMB, zoom with the wheel.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

from steve_il.envs import pick_scene as ps  # noqa: E402  (plain python, no Isaac needed)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--episodes", type=int, default=0, help="number of episodes (0: loop until the window is closed)")
parser.add_argument("--seed", type=int, default=0, help="seed of the first episode (episode i uses seed + i)")
parser.add_argument("--pick", default=None, choices=[o.name for o in ps.OBJECTS], help="cube to move (default: random)")
parser.add_argument("--trace", action="store_true", help="print the object / TCP state after every motion segment")
parser.add_argument("--no-wall", action="store_true", help="no obstacle between the object and the target")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from steve_il.envs.steve_pick_env_cfg import StevePickEnvCfg  # noqa: E402
from steve_il.planning.client import ArmPlannerClient  # noqa: E402
from steve_il.scripted.pick_place import PickPlaceRunner  # noqa: E402


def main() -> int:
    env = ManagerBasedRLEnv(StevePickEnvCfg())
    print("starting the cuRobo planner process (first run compiles CUDA kernels, ~1 min) ...", flush=True)
    planner = ArmPlannerClient()
    try:
        failures, i = run_episodes(env, planner)
    finally:
        planner.close()
    print(f"\n{i - len(failures)}/{i} episodes succeeded")
    for f in failures:
        print("  FAIL", f)
    return 1 if failures else 0


def run_episodes(env, planner) -> tuple[list[str], int]:
    runner = PickPlaceRunner(env, planner, trace=args.trace)
    failures, i = [], 0
    while simulation_app.is_running() and (args.episodes == 0 or i < args.episodes):
        seed = args.seed + i
        scenario = ps.sample_scenario(np.random.default_rng(seed), pick=args.pick, wall_on_path=not args.no_wall)
        print(f"--- episode {i} (seed {seed}): move {scenario.pick} to {np.round(scenario.target_xy, 2)}", flush=True)
        runner.reset(scenario)
        result = runner.pick_and_place()
        if not result.success:
            failures.append(f"seed {seed} ({scenario.pick}): {result.stage}: {result.reason}")
        i += 1
    return failures, i


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # Kit's close() can hang; nothing to clean up here
