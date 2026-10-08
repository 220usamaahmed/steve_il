"""Steve mobile manipulator (Neobotix MPO-700 + UR5e + Robotiq 2F-85) for Isaac Lab."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets" / "steve"
URDF_PATH = ASSETS_DIR / "urdf" / "steve_minimal.urdf"
USD_PATH = ASSETS_DIR / "usd" / "free" / "steve.usd"

# holonomic variant: virtual x/y/yaw joints under base_link, world-fixed anchor (see tools/make_holonomic_urdf.py)
HOLONOMIC_URDF_PATH = ASSETS_DIR / "urdf" / "steve_holonomic.urdf"
HOLONOMIC_USD_PATH = ASSETS_DIR / "usd" / "holonomic" / "steve_holonomic.usd"

# cuRobo arm planning model: URDF derived by tools/make_arm_planning_urdf.py, robot config by
# tools/build_curobo_robot.py (paths inside the YAML are relative to ASSETS_DIR)
ARM_PLANNING_URDF_PATH = ASSETS_DIR / "urdf" / "steve_arm_planning.urdf"
CUROBO_ROBOT_PATH = ASSETS_DIR / "curobo" / "steve_arm.yml"
