from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Phase


@dataclass
class Waypoint:
    phase: Phase
    duration_s: float
    full_joint_positions: np.ndarray


class JointTrajectory:
    """Piecewise-linear full-articulation position trajectory.

    The returned command always contains every articulation DOF.  The dataset
    action is obtained later by selecting the explicit 16 action joints.
    """

    def __init__(self, waypoints: list[Waypoint], gripper_open: dict[str, np.ndarray], gripper_close: dict[str, np.ndarray], gripper_indices: dict[str, np.ndarray], close_start_s: float, release_start_s: float, phase_offset_s: float = 0.0):
        if not waypoints:
            raise ValueError("trajectory needs at least one waypoint")
        self.waypoints = waypoints
        self.gripper_open = {side: np.asarray(values, dtype=np.float64).reshape(2) for side, values in gripper_open.items()}
        self.gripper_close = {side: np.asarray(values, dtype=np.float64).reshape(2) for side, values in gripper_close.items()}
        self.gripper_indices = gripper_indices
        self.close_start_s = float(close_start_s)
        self.release_start_s = float(release_start_s)
        self.phase_offset_s = float(phase_offset_s)
        self._ends = np.cumsum([max(0.0, w.duration_s) for w in waypoints])
        self.duration_s = float(self._ends[-1])

    def sample(self, time_s: float) -> tuple[np.ndarray, Phase]:
        t = float(np.clip(time_s, 0.0, self.duration_s))
        idx = int(np.searchsorted(self._ends, t, side="right"))
        idx = min(idx, len(self.waypoints) - 1)
        start_t = 0.0 if idx == 0 else float(self._ends[idx - 1])
        waypoint = self.waypoints[idx]
        alpha = 1.0 if waypoint.duration_s <= 0 else (t - start_t) / waypoint.duration_s
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if idx == 0:
            command = waypoint.full_joint_positions.copy()
        else:
            prev = self.waypoints[idx - 1].full_joint_positions
            command = prev + alpha * (waypoint.full_joint_positions - prev)

        # Arms follow the shared trajectory.  Only the two real gripper axes
        # are phase-shifted; both arms remain in every command and recording.
        close_duration = max(1e-3, self._phase_time(Phase.CLOSE)[1])
        left_start = self.close_start_s - max(self.phase_offset_s, 0.0)
        right_start = self.close_start_s - max(-self.phase_offset_s, 0.0)
        for side in ("left", "right"):
            start = left_start if side == "left" else right_start
            progress = np.clip((t - start) / close_duration, 0.0, 1.0)
            if t < self.release_start_s:
                value = self.gripper_open[side] + progress * (self.gripper_close[side] - self.gripper_open[side])
            else:
                release_progress = np.clip((t - self.release_start_s) / close_duration, 0.0, 1.0)
                value = self.gripper_close[side] + release_progress * (self.gripper_open[side] - self.gripper_close[side])
            command[self.gripper_indices[side]] = value
        return command.astype(np.float64), waypoint.phase

    def _phase_time(self, phase: Phase) -> tuple[float, float]:
        for i, waypoint in enumerate(self.waypoints):
            if waypoint.phase == phase:
                start = 0.0 if i == 0 else float(self._ends[i - 1])
                return start, float(waypoint.duration_s)
        raise KeyError(phase)
