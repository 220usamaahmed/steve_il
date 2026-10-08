# steve_il

Steve (Neobotix MPO-700 base + UR5e + Robotiq 2F-85) in Isaac Lab, for scripted trajectory
collection and imitation learning.

**Status: milestones 1 and 2 done, milestone 3 started.** The robot imports and is verified (1); a manager-based
Isaac Lab environment with base, arm and gripper controllers exists and its behaviour is measured (2). Scripted
demo collection (3) has its first step: **arm-only pick and place with cuRobo from a stationary base** (see
"Pick and place with cuRobo" below). Not yet: base motion in the plan, recording, policy training (4).

## Requirements
Conda env `env_isaaclab` with Isaac Sim 5.1 + Isaac Lab 2.3.2, an NVIDIA GPU.
```bash
export OMNI_KIT_ACCEPT_EULA=YES
PY=~/miniconda3/envs/env_isaaclab/bin/python
```
Pick and place also needs [cuRobo](https://github.com/NVlabs/curobo) (v2 rewrite, 0.8.0), installed into the same env:
```bash
git clone https://github.com/NVlabs/curobo ~/curobo && cd ~/curobo
$PY -m pip install ".[cu12]"          # kernels are compiled at run time (warp / cuda-core), no nvcc needed
$PY -m pip install websockets==12.0   # cuRobo's viser pulls websockets 16, which breaks Isaac Sim's pin (12.0)
```
cuRobo cannot run inside the Isaac Sim process (Kit loads its own warp 1.8, cuRobo needs the pip warp 1.18), so
the scripts start it as a separate planner process (`steve_il.planning.client` / `server`) and talk to it over a
local socket. The first planner start compiles kernels (~1 min); later starts take ~10 s.

## Workflow
```bash
python3 -m unittest discover -s tests -v        # static URDF / name checks, no Isaac needed (21 tests)
python3 tools/make_holonomic_urdf.py            # only if steve_minimal.urdf changed
$PY scripts/convert_urdf.py --headless                       # free variant      -> assets/steve/usd/free/
$PY scripts/convert_urdf.py --headless --variant holonomic   # holonomic variant -> assets/steve/usd/holonomic/
$PY scripts/verify_import.py --headless         # milestone 1: 20 checks on the free robot
$PY scripts/drive_test.py --headless            # milestone 2: 26 checks on all controllers (~40 s)
$PY scripts/drive_test.py                       # same, but watch it in the Isaac Sim viewer
$PY scripts/demo_env.py                         # looping demo of the environment in the viewer (base + arm + gripper)
$PY scripts/view.py --demo                      # just look at the free (milestone 1) robot

# pick and place with cuRobo (arm only, stationary base)
$PY scripts/planner_check.py                    # 12 checks of the cuRobo planner alone (GPU, no Isaac Sim)
$PY scripts/pick_place_demo.py                  # viewer: random scenarios, loops until the window is closed
$PY scripts/pick_place_demo.py --headless --episodes 30 --seed 200   # pass/fail per episode, exit code
$PY scripts/pick_place_demo.py --pick cube_red --seed 3 --episodes 1 --trace   # one scenario, state after each segment
python3 tools/make_arm_planning_urdf.py         # only if steve_minimal.urdf changed -> steve_arm_planning.urdf
PYTHONPATH=<dir with websockets>=13> $PY tools/build_curobo_robot.py   # only if that URDF changed (fits collision spheres)
```
Every check script exits non-zero on failure.
`verify_import.py` checks: USD bodies/joints/types == `steve_names`, mimic joints imported,
articulation DOFs/bodies, total mass == URDF, base holds still on the ground, arm holds
`ARM_HOME` under gravity, wheels at rest, `finger_joint` drags all 5 mimic joints.
The "Unresolved reference prim path .../visuals/..." warnings are importer noise for frame-only
links (`ur5e_tool0`, `ur5e_flange`, ...) and are harmless.

## Layout
```
assets/steve/        urdf/ + meshes/ (committed, see its README), usd/ (generated, gitignored)
tools/               build_minimal_urdf.py   regenerate steve_minimal.urdf + meshes from steve_isaacsim
                     make_holonomic_urdf.py  steve_minimal -> steve_holonomic (virtual x/y/yaw base)
                     make_arm_planning_urdf.py  steve_minimal -> steve_arm_planning (no wheels, gripper fixed open)
                     build_curobo_robot.py   steve_arm_planning -> assets/steve/curobo/steve_arm.yml (cuRobo model)
source/steve_il/
  robots/steve_names.py   canonical names, groups, ARM_READY, TCP offset (pure python)
  robots/steve_cfg.py     STEVE_CFG (free, milestone 1) and STEVE_HOLONOMIC_CFG (used by the envs)
  robots/gripper.py       measured pad distance vs finger angle (pure python)
  mdp/actions.py          HolonomicBaseVelocityAction, BaseFrameDifferentialIKAction, MimicGripperPositionAction
  mdp/observations.py     tcp_pose_in_base, base_pose_xy_yaw, base_velocity_body
  envs/steve_env_cfg.py   SteveEnvCfg: scene, actions, observations (task envs will inherit from it)
  envs/pick_scene.py      table / cubes / obstacles / marker definitions + random scenario sampler (pure python)
  envs/steve_pick_env_cfg.py  StevePickEnvCfg: joint-position arm, finger-angle gripper, the pick scene
  planning/               geometry, world, grasps (pure numpy); arm_planner (cuRobo); server + client (process boundary)
  scripted/motion.py      action builder + smooth TCP interpolation (reused by demo scripts)
  scripted/pick_place.py  PickPlaceRunner: reset a scenario, pick, place, check
scripts/             convert_urdf.py, verify_import.py, drive_test.py, demo_env.py, view.py,
                     planner_check.py, pick_place_demo.py
tests/               test_urdf.py, test_planning.py (no Isaac / GPU needed)
configs/ data/       (empty for now; data/ is gitignored)
```
Joints are always addressed by name. Isaac orders DOFs by USD traversal, not URDF order
(e.g. `mmo_700_caster_back_left_joint` is DOF 0).

## Key facts
- 34 links / 33 joints / 20 DOFs: 4 casters, 4 wheels, 6 arm, 1 driven + 5 mimic gripper joints.
- 283.55 kg total (base 200, casters 36, wheels 23.6, cabinet 2, UR5e 21.7, gripper 0.25).
- Root `base_link`, articulation root `/steve/base_link`, end-effector frame `ur5e_tool0`.
- Gripper: command only `finger_joint` (0 open .. 0.8 rad closed); mimic joints follow through
  PhysX mimic constraints (tuned to 200 rad/s, critically damped, in `convert_urdf.py`).
- `merge_fixed_joints=False` is required to keep frames like `ur5e_tool0` as bodies.
- Isaac Lab's `convert_mimic_joints_to_normal_joints=True` is what *enables* mimic import
  (the name is inverted; it is forwarded to `set_parse_mimic`).

## Environment (milestone 2)
`SteveEnvCfg` (`steve_il/envs/steve_env_cfg.py`) is an Isaac Lab `ManagerBasedRLEnv` config: Steve on a
ground plane, physics 120 Hz, control 30 Hz, no reward (imitation learning), 30 s episodes.
```python
env = ManagerBasedRLEnv(SteveEnvCfg())       # after AppLauncher(...)
obs, _ = env.reset()
obs, _, terminated, truncated, _ = env.step(action)   # action shape (num_envs, 11)
```
**Action (dim 11)**

| slice | meaning |
|---|---|
| `0:3` | base body-frame twist `vx vy wz`, normalised to [-1, 1]; 1.0 = 0.8 m/s, 0.8 m/s, 1.0 rad/s; applied velocity ramps with 2.0 m/s^2 / 3.0 rad/s^2 |
| `3:10` | **absolute** TCP pose `x y z qw qx qy qz` in the `base_link` frame, tracked with differential IK (DLS) |
| `10` | gripper, Isaac Lab binary convention: `>= 0` open, `< 0` close (finger_joint 0 / 0.75 rad) |

**Observations** (`obs["policy"]`, a dict, not concatenated): `arm_joint_pos` (6), `arm_joint_vel` (6),
`gripper_pos` (1), `tcp_pose` (7, base frame), `base_pose` (x, y, yaw in the env frame), `base_vel` (body frame), `last_action` (11).

**Design decisions to know about**
- *Holonomic base = virtual joints.* `base_x/base_y` (prismatic) and `base_yaw` (continuous) under a world-welded
  anchor, velocity-driven like Isaac Lab's Ridgeback-Franka. The wheels/casters are in the model for looks and
  collision but **do not drive and are not animated**; `base_link` floats ~3 mm above the ground so they never
  touch it. The real swerve drive is not simulated.
- *Arm IK is in the base frame.* Isaac Lab's stock IK term uses the articulation root, which here is the fixed
  anchor, so `BaseFrameDifferentialIKAction` measures against the moving `base_link` instead. Moving or turning the
  base does not change a commanded TCP pose.
- *Gravity is off for the robot's own links.* With plain PD joints the arm sagged ~1 cm and differential IK, which
  re-targets from the current joints each step, never converged. A real UR5e holds position exactly. Grasped
  objects keep their own gravity. Consequence: no payload-induced sag is modelled beyond what the object weighs.
- *TCP* = midpoint between the two finger pads, 0.1388 m along `ur5e_tool0` z (checked against the simulated pads).
- *Ready pose* `ARM_READY`: TCP at (0.60, 0, 0.80) m in the base frame, tool z pointing down.

**Measured behaviour** (`drive_test.py`, all 26 checks pass): forward/strafe displacement matches v*T within 0.1 %,
a 90 deg turn within 0.3 %, TCP reaches six workspace poses (z 0.55-0.80, y +-0.25, x 0.45-0.80) to <0.02 mm, and a
commanded base-frame TCP pose holds to 2.8 mm peak (start-up transient), 1.3 mm
steady while turning and translating at once, and 0.007 mm after the base stops.

**Gotchas for scripted trajectories**
- Keep the tool orientation close to the ready pose. Switching between very different orientations (e.g. down ->
  forward -> down) makes IK wind `wrist_3` by ~-5 rad, up against its +-2 pi limit.
- Top-down reach was checked down to z = 0.45 m at x = 0.55 m.
- Targets are absolute: interpolate (see `scripted/motion.py`) instead of jumping, otherwise the arm lunges.
- The 30 s episode time-out auto-resets the env mid-motion; set `cfg.episode_length_s` higher for long scripts.

## Pick and place with cuRobo (milestone 3, step 1: arm only, stationary base)
`scripts/pick_place_demo.py` picks a cube from a table and puts it on a target, with a wall in between. cuRobo plans
every arm motion; the base does not move. Result: 60 of 60 episodes succeeded on seeds 300-359 (and 30 of 30 on seeds 200-229, which the
scenario sampler was tuned on); every cube ended within 0.7 cm of the target (median 0.6 cm) and flat; ~16 s of
simulated time and ~1.5 s of planning per episode.

**Per episode** (`scripted/pick_place.py`): sample a scenario (`envs/pick_scene.py`: 3 cubes, a 16 cm wall halfway
between the picked cube and the target, a 26 cm tower, the target marker) -> read the base pose and the object poses
from the simulator -> cuRobo `plan_grasp` over 4 top-down grasp candidates: approach 12 cm above, straight-line
descent, straight-line lift of 22 cm -> execute the joint trajectories at 30 Hz -> close the gripper -> attach the
cube to the robot in the planner -> `plan_grasp` again over 8 candidate place poses (approach, descent, retreat) ->
release. Success = cube within 3 cm of the target and lying flat. If a plan fails the runner retries with other
approach heights; inputs of plans that still fail are saved to `logs/plan_failures/*.pkl` for offline replay.

**World-frame goals.** The planner works in the `base_link` frame and knows nothing about where the base is: every
plan re-expresses the obstacles and goals in the *measured* base frame (`planning/geometry.py`,
`ArmPlanner.set_world`). That is what will let the base move (and be imprecise) in the next steps.

**Action space of `StevePickEnvCfg`** (dim 10): `[vx vy wz (normalised) | 6 absolute arm joint positions [rad], ARM_JOINTS
order | finger angle [rad], 0 open ... 0.8 closed]`. The arm is joint-position driven because that is what a planner
outputs; `SteveEnvCfg` (TCP pose + binary gripper) is unchanged.

**Things that were not obvious** (each cost real debugging time):
- *cuRobo runs in its own process.* Kit's bundled warp 1.8 and cuRobo's warp 1.18 cannot share a process.
- *The planner must see the held cube.* cuRobo's `attach` fitted 2 of 32 spheres to a 5 cm cube and its own
  disable / enable cycles inside `plan_grasp` then removed them, so the cube was carried through the wall. The cube is
  now a 3x3x3 sphere lattice and the reference spheres are overwritten while attached (`ArmPlanner.attach`).
  `scripts/planner_check.py` fails if that regresses.
- *Grasp contact links.* Fingers and knuckles are excluded from collision during the final approach and the lift (the
  knuckles sit right on top of a short cube); the palm is not, so the TCP is raised for short objects
  (`planning/grasps.py`, `PALM_*`). cuRobo has no cylinder collision, cylinders become bounding cuboids.
- *The imported gripper is soft.* Only `finger_joint` is driven; the four-bar linkage is replaced by weak mimic joints,
  so under load the pads pitch and a cube tilts out. Squeezing hard makes it worse (a stiff gain gave >100 N on a
  0.1 kg cube). `MimicGripperPositionAction` drives *both* outer knuckles with a low gain (4 Nm/rad; force is
  gain x remaining error, so a fixed target of 0.75 rad gives a roughly constant force; by my estimate 10-20 N, not measured) and makes each inner-finger
  joint stiffly follow the outer knuckle of its own side. The finger angle is not a "holding" signal; the controller
  checks where the cube is instead.
- *Place with the tool upright.* A cube that tilted a few degrees in the grip lands flat when released; compensating
  the tilt would tilt the tool and make the straight-line descent harder to reach (`grasps.flatten`).
- *Scenario setup order.* Teleporting a kinematic obstacle next to a resting cube makes PhysX imply a huge velocity and
  the cube is flung away (looked like random planning failures). `PickPlaceRunner.reset` parks the cubes, moves the
  obstacles, waits, then places the cubes, and raises if anything moved.
- *The tower needs room.* Behind the picked cube the UR5e's wrist links stand above the gripper in a top-down grasp,
  so a tower within ~0.17 m of a cube or the target makes the grasp infeasible for every candidate (`OBSTACLE_CLEARANCE`,
  chosen from planner feasibility over 60 sampled layouts: 4/60 infeasible at 0.09 m, 0/60 at 0.19 m).

**Limits for now.** Base stationary; cubes only, top-down grasps; ground-truth object poses (no perception);
no latency, noise or slip is modelled beyond what PhysX does (the worst joint tracking error of each episode is
recorded in `Result.max_tracking_error` but not reported here).
