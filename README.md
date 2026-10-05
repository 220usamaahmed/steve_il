# steve_il

Steve (Neobotix MPO-700 base + UR5e + Robotiq 2F-85) in Isaac Lab, for scripted trajectory
collection and imitation learning.

**Status: milestones 1 and 2 done.** The robot imports and is verified (1); a manager-based Isaac Lab
environment with base, arm and gripper controllers exists and its behaviour is measured (2).
Not yet: scripted demo collection (3), policy training / evaluation (4).

## Requirements
Conda env `env_isaaclab` with Isaac Sim 5.1 + Isaac Lab 2.3.2, an NVIDIA GPU.
```bash
export OMNI_KIT_ACCEPT_EULA=YES
PY=~/miniconda3/envs/env_isaaclab/bin/python
```

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
source/steve_il/
  robots/steve_names.py   canonical names, groups, ARM_READY, TCP offset (pure python)
  robots/steve_cfg.py     STEVE_CFG (free, milestone 1) and STEVE_HOLONOMIC_CFG (used by the envs)
  mdp/actions.py          HolonomicBaseVelocityAction, BaseFrameDifferentialIKAction
  mdp/observations.py     tcp_pose_in_base, base_pose_xy_yaw, base_velocity_body
  envs/steve_env_cfg.py   SteveEnvCfg: scene, actions, observations (task envs will inherit from it)
  scripted/motion.py      action builder + smooth TCP interpolation (reused by demo scripts)
scripts/             convert_urdf.py, verify_import.py, drive_test.py, demo_env.py, view.py
tests/               test_urdf.py
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
