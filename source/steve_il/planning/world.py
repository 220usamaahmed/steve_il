"""Collision world description shared by the simulator side and the cuRobo planner.

Everything here is plain numpy / python: the environment fills ``Prim`` poses from the
simulation, ``prims_to_scene_dict`` turns them into a cuRobo scene in the ``base_link`` frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from steve_il.planning import geometry as geo


@dataclass
class Prim:
    """A box or a vertical cylinder with a pose in the world frame.

    ``size`` is ``(sx, sy, sz)`` for a box and ``(radius, height)`` for a cylinder (axis = local z).
    """

    name: str
    kind: str  # "box" | "cylinder"
    size: tuple[float, ...]
    pose: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0, 1, 0, 0, 0]))

    def __post_init__(self):
        if self.kind not in ("box", "cylinder"):
            raise ValueError(f"unknown prim kind {self.kind!r}")
        expected = 3 if self.kind == "box" else 2
        if len(self.size) != expected:
            raise ValueError(f"{self.kind} needs {expected} size values, got {self.size}")
        self.pose = np.asarray(self.pose, dtype=float)

    @property
    def height(self) -> float:
        return self.size[2] if self.kind == "box" else self.size[1]

    @property
    def box_dims(self) -> tuple[float, float, float]:
        """Oriented bounding box (in the prim frame): the prim itself, or the cylinder's cuboid hull."""
        if self.kind == "box":
            return tuple(self.size)
        r, h = self.size
        return (2 * r, 2 * r, h)

    @property
    def yaw(self) -> float:
        return geo.quat_to_yaw(self.pose[3:])


def prims_to_scene_dict(prims: list[Prim], base_xy_yaw, margin: float = 0.0) -> dict:
    """cuRobo scene dict (all cuboids, base_link frame) for ``prims`` given in the world frame.

    Cylinders are replaced by their bounding cuboid (conservative). ``margin`` inflates every
    cuboid by that much on each side.
    """
    cuboids = {}
    for p in prims:
        pose_b = geo.world_to_base(p.pose, base_xy_yaw)
        dims = [d + 2 * margin for d in p.box_dims]
        cuboids[p.name] = {"dims": [float(d) for d in dims], "pose": [float(v) for v in pose_b]}
    return {"cuboid": cuboids}
