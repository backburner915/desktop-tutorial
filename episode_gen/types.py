"""Shared plain-data types passed between SimAdapter, detectors, recovery
and the dataset writer. Kept dependency-free (stdlib only) so the FSM
logic in this package can be unit-tested without Isaac Sim installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from episode_gen.fsm import Phase


@dataclass
class Observation:
    """One control-step snapshot, as reported by a SimAdapter.

    Every field an anomaly detector might need lives here so
    `episode_gen/detectors.py` never has to reach back into the
    simulator directly — that is what keeps detectors testable with
    MockSimAdapter and portable to the real Isaac Lab adapter.
    """

    t: float
    qpos: list[float]  # 16D, JOINT_ORDER
    qvel: list[float]  # 16D, JOINT_ORDER
    left_ee_pos: tuple[float, float, float]
    right_ee_pos: tuple[float, float, float]
    object_pos: tuple[float, float, float]
    # currently active contact pairs, e.g. {("left_arm_link6", "table")}
    contacts: set[tuple[str, str]] = field(default_factory=set)
    dist_to_grasp_frame: dict[str, float] = field(default_factory=dict)  # arm -> meters
    grip_force: dict[str, float] = field(default_factory=dict)  # arm -> N
    joint_limit_margin: dict[str, float] = field(default_factory=dict)  # joint -> margin
    object_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    scenario_class: str  # "R11", "R12", ...
    phase: Phase
    t: float
    arm: str  # "left" | "right" | "both"
    detail: dict[str, Any] = field(default_factory=dict)
