"""Pick-and-place scene: a table in front of Steve with cubes, obstacles and a target marker.

Plain python / numpy (no Isaac imports at module level) so the sizes are shared by the Isaac Lab
scene config, the scenario sampler and the planner's collision world. Frame: the environment frame;
Steve starts at the origin facing +x, the table is in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from steve_il.planning import geometry as geo
from steve_il.planning.world import Prim

TABLE_TOP_Z = 0.45
TABLE = Prim("table", "box", (0.80, 1.50, TABLE_TOP_Z), np.array([0.90, 0.0, TABLE_TOP_Z / 2, 1, 0, 0, 0]))


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    kind: str  # "box" | "cylinder"
    size: tuple[float, ...]
    color: tuple[float, float, float]
    mass: float = 0.0  # kg; 0 for non-graspable (kinematic) obstacles

    def prim(self, x: float, y: float, yaw: float = 0.0) -> Prim:
        """Prim resting on the table top at (x, y) with heading ``yaw``."""
        p = Prim(self.name, self.kind, self.size)
        p.pose = np.array([x, y, TABLE_TOP_Z + p.height / 2, *geo.quat_from_yaw(yaw)])
        return p


# graspable objects: 5 cm cubes (all the same size, different colours)
OBJECTS = (
    ObjectSpec("cube_red", "box", (0.05, 0.05, 0.05), (0.85, 0.15, 0.15), mass=0.10),
    ObjectSpec("cube_green", "box", (0.05, 0.05, 0.05), (0.20, 0.70, 0.25), mass=0.10),
    ObjectSpec("cube_blue", "box", (0.05, 0.05, 0.05), (0.15, 0.35, 0.85), mass=0.10),
)
# kinematic obstacles on the table
OBSTACLES = (
    ObjectSpec("wall", "box", (0.14, 0.03, 0.16), (0.55, 0.55, 0.60)),
    ObjectSpec("tower", "box", (0.06, 0.06, 0.26), (0.30, 0.70, 0.35)),
)
MARKER_RADIUS = 0.035
MARKER_NAME = "marker"
SPECS = {s.name: s for s in (*OBJECTS, *OBSTACLES)}

# reachable part of the table top (base-frame x/y of the base at the origin)
ZONE_X = (0.56, 0.78)
ZONE_Y = (-0.30, 0.30)
MIN_TARGET_GAP = 0.32  # picked object -> target: leaves room for the wall in between
CLEARANCE = 0.035  # [m] free space between the footprints of two things on the table
OBSTACLE_CLEARANCE = 0.19  # [m] free space between the tower and objects / the target: the wrist links, which stand
# above and behind the gripper in a top-down grasp, must clear it (0.09: 4/60 picks infeasible, 0.19: 0/60)
WALL_CLEARANCE = 0.03  # [m] the wall is thin and sits across the path, so it can be closer


def footprint_radius(spec: ObjectSpec) -> float:
    """Radius of the circle around the object's footprint (the real footprint is never larger)."""
    if spec.kind == "cylinder":
        return spec.size[0]
    return 0.5 * float(np.hypot(spec.size[0], spec.size[1]))


@dataclass
class Scenario:
    """Where everything is, and what to do. Poses are (x, y, yaw) on the table top."""

    poses: dict[str, tuple[float, float, float]]
    pick: str
    target_xy: tuple[float, float]

    def prims(self) -> list[Prim]:
        return [SPECS[k].prim(*v) for k, v in self.poses.items()]


class _Placer:
    """Collects circular footprints on the table and tests new ones against them."""

    def __init__(self):
        self.items: list[tuple[np.ndarray, float]] = []

    def fits(self, xy: np.ndarray, radius: float, clearance: float = CLEARANCE) -> bool:
        return all(np.linalg.norm(xy - c) >= r + radius + clearance for c, r in self.items)

    def add(self, xy: np.ndarray, radius: float) -> None:
        self.items.append((xy, radius))


def sample_scenario(rng: np.random.Generator, pick: str | None = None, wall_on_path: bool = True) -> Scenario:
    """Random layout: 3 objects, a target for one of them, a wall between them and a tower elsewhere.

    Nothing overlaps: every footprint keeps ``CLEARANCE`` from every other one, including the target
    marker (so the placed object has room) and the wall.
    """
    pick = pick or str(rng.choice([o.name for o in OBJECTS]))
    lo, hi = (ZONE_X[0], ZONE_Y[0]), (ZONE_X[1], ZONE_Y[1])
    r_marker = MARKER_RADIUS
    for _ in range(2000):
        placer = _Placer()
        pts = {}
        ok = True
        for spec in OBJECTS:
            xy = rng.uniform(lo, hi)
            if not placer.fits(xy, footprint_radius(spec)):
                ok = False
                break
            placer.add(xy, footprint_radius(spec))
            pts[spec.name] = xy
        if not ok:
            continue
        target = rng.uniform(lo, hi)
        r_pick = footprint_radius(SPECS[pick])
        if np.linalg.norm(pts[pick] - target) < MIN_TARGET_GAP or not placer.fits(target, max(r_marker, r_pick)):
            continue
        placer.add(target, max(r_marker, r_pick))

        poses = {k: (float(v[0]), float(v[1]), float(rng.uniform(-np.pi, np.pi))) for k, v in pts.items()}
        d = target - pts[pick]
        wall = SPECS["wall"]
        if wall_on_path:
            mid = pts[pick] + 0.5 * d
            if not placer.fits(mid, footprint_radius(wall), WALL_CLEARANCE):
                continue
            placer.add(mid, footprint_radius(wall))
            poses["wall"] = (float(mid[0]), float(mid[1]), float(np.arctan2(d[1], d[0]) + np.pi / 2))
        else:  # parked at the edge of the table, out of the way
            poses["wall"] = (1.15, 0.62, 0.0)
        tower = SPECS["tower"]
        for _ in range(300):
            t = rng.uniform((ZONE_X[0] - 0.02, ZONE_Y[0]), (ZONE_X[1] + 0.12, ZONE_Y[1]))
            if placer.fits(t, footprint_radius(tower), OBSTACLE_CLEARANCE):
                poses["tower"] = (float(t[0]), float(t[1]), float(rng.uniform(-np.pi, np.pi)))
                break
        else:
            continue
        return Scenario(poses, pick, (float(target[0]), float(target[1])))
    raise RuntimeError("could not sample a scenario")
