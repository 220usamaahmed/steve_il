"""Robotiq 2F-85 closing geometry as simulated (pure python, no Isaac).

The pad centre distance as a function of the driven ``finger_joint`` angle was measured in the
simulator by ramping the finger target from 0 to 0.8 rad with nothing between the pads (the
four-bar linkage is imported as mimic joints, so there is no closed form to rely on). The pick
controller uses it to close quickly to just before the pads should touch an object of known width and
only then ramp slowly, so the object is not shoved.
"""

from __future__ import annotations

import numpy as np

PAD_THICKNESS = 0.00635  # [m] box collider of the inner finger pads along the closing axis

# finger_joint [rad] -> distance between the two pad *centres* [m]
_THETA = np.array([0.0, 0.1067, 0.2133, 0.32, 0.4267, 0.5333, 0.64, 0.7467, 0.8])
_PAD_DISTANCE = np.array([0.0924, 0.0829, 0.0725, 0.0616, 0.0501, 0.0383, 0.0262, 0.0140, 0.0079])


def pad_distance(theta: float) -> float:
    """Distance between the pad centres [m] when ``finger_joint`` is at ``theta`` [rad] (unobstructed)."""
    return float(np.interp(theta, _THETA, _PAD_DISTANCE))


def finger_angle_for_width(width: float) -> float:
    """``finger_joint`` angle [rad] at which the pad faces touch an object of the given ``width`` [m]."""
    target = width + PAD_THICKNESS
    if not _PAD_DISTANCE[-1] <= target <= _PAD_DISTANCE[0]:
        raise ValueError(f"width {width:.3f} m is outside the gripper's range")
    return float(np.interp(target, _PAD_DISTANCE[::-1], _THETA[::-1]))
