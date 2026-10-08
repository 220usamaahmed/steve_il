"""Result types shared by the planner server (cuRobo process) and its clients (Isaac Sim process)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONTROL_DT = 1.0 / 30.0  # env control period; planned trajectories are sampled at this rate


@dataclass
class GraspPlan:
    """Three joint trajectories [T_i, 6] (float32 numpy) sampled at ``dt``, plus the chosen candidate."""

    approach: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    goal_index: int
    dt: float = CONTROL_DT
    planning_time: float = 0.0

    @property
    def segments(self) -> list[np.ndarray]:
        return [self.approach, self.grasp, self.lift]


@dataclass
class PlanFailure:
    status: str
    planning_time: float = 0.0
    details: dict = field(default_factory=dict)
