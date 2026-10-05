"""Convert assets/steve/urdf/steve_minimal.urdf to USD (Isaac Sim 5.1 / Isaac Lab 2.3).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/convert_urdf.py --headless

Settings that matter for this robot:
  * merge_fixed_joints=False  keeps frames such as ``ur5e_tool0`` / ``ur5e_flange`` (the
    default would collapse them and break name-based lookups).
  * fix_base                  False for the free variant; True for --variant holonomic.
  * mimic parsing on          the 5 Robotiq mimic joints become PhysX mimic joints.
  * joint drives are zeroed   actuator gains come from the ArticulationCfg at spawn time.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--variant", choices=("free", "holonomic"), default="free",
                    help="free: floating base with wheels (steve.usd); "
                         "holonomic: virtual x/y/yaw base welded to the world (steve_holonomic.usd)")
parser.add_argument("--urdf", type=Path, default=None, help="input URDF (default: assets/steve/urdf/...)")
parser.add_argument("--usd", type=Path, default=None, help="output USD (default: assets/steve/usd/steve.usd)")
parser.add_argument("--mimic-freq", type=float, default=200.0,
                    help="PhysX mimic-joint natural frequency [rad/s] (importer default 25)")
parser.add_argument("--mimic-damping-ratio", type=float, default=1.0,
                    help="PhysX mimic-joint damping ratio (importer default 0.005)")
parser.add_argument(
    "--merge-fixed", action="store_true", help="merge links joined by fixed joints (importer default)"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from steve_il import HOLONOMIC_URDF_PATH, HOLONOMIC_USD_PATH, URDF_PATH, USD_PATH  # noqa: E402


def tune_mimic_joints(usd_path: Path, natural_frequency: float, damping_ratio: float) -> int:
    """Stiffen the PhysX mimic constraints written by the importer.

    The importer leaves naturalFrequency=25 / dampingRatio=0.005. The constraint stiffness
    scales with link inertia; the gripper links are very light, so with the defaults the inner
    fingers sag 0.06-0.13 rad off finger_joint under gravity. 200 rad/s, critically damped,
    brings that to ~0 without jitter at 120 Hz.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    count = 0
    for prim in stage.Traverse():
        for schema in prim.GetAppliedSchemas():
            if schema.startswith("PhysxMimicJointAPI:"):
                axis = schema.split(":")[1]
                prim.GetAttribute(f"physxMimicJoint:{axis}:naturalFrequency").Set(natural_frequency)
                prim.GetAttribute(f"physxMimicJoint:{axis}:dampingRatio").Set(damping_ratio)
                count += 1
    stage.GetRootLayer().Save()
    return count


def main() -> None:
    holonomic = args.variant == "holonomic"
    urdf = (args.urdf or (HOLONOMIC_URDF_PATH if holonomic else URDF_PATH)).resolve()
    usd = (args.usd or (HOLONOMIC_USD_PATH if holonomic else USD_PATH)).resolve()
    usd.parent.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(usd.parent),
        usd_file_name=usd.name,
        fix_base=holonomic,  # welds world_anchor to the world; the free variant must stay free
        merge_fixed_joints=args.merge_fixed,
        # Isaac Lab forwards this flag to the importer's set_parse_mimic(): True = import <mimic>
        # as PhysX mimic joints (the flag name is misleading; False silently drops the coupling).
        convert_mimic_joints_to_normal_joints=True,
        self_collision=False,
        collider_type="convex_hull",
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    converter = UrdfConverter(cfg)
    n_mimic = tune_mimic_joints(Path(converter.usd_path), args.mimic_freq, args.mimic_damping_ratio)
    print(f"[convert_urdf] {urdf} -> {converter.usd_path} (mimic joints tuned: {n_mimic})", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
