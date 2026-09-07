"""Scenario configuration: the only thing that changes between episodes.

This is the "Scenario Config" layer from the taxonomy doc. Nothing here
is simulator-specific — it is plain data, loaded from YAML, consumed by
episode_runner.py through a SimAdapter. Adding a new episode variant
must only ever mean adding a new YAML file (or a batch of generated
ones), never a new Python file.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TASK_TEXT = (
    "Pick up the Crew Lock Bag with both arms and place it in the green target ring."
)

# 16D joint order, frozen by docs/dataset_spec_v0.1.md — verified against
# userguide-galaxea/galaxea_lab: omni/isaac/lab_assets/galaxea_robots.py
JOINT_ORDER = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
    "left_arm_joint5", "left_arm_joint6", "left_gripper_axis1", "left_gripper_axis2",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
    "right_arm_joint5", "right_arm_joint6", "right_gripper_axis1", "right_gripper_axis2",
]


@dataclass
class AnomalyConfig:
    """Parameterizes one R-family closure (see taxonomy doc §2).

    `scenario_class` selects which detector/recovery function pair runs
    (e.g. "R11"); everything else is the config axis that lets the same
    R11 detector cover "hit the table" vs "hit a cabinet" vs "hit an
    obstacle", left arm vs right arm, different contact severities, etc.
    """

    scenario_class: str
    trigger_phase: str = "APPROACH"
    contact_arm: str = "left"  # "left" | "right" | "both"
    surface: str = "table"
    # how far (m) from the intended grasp frame the arm still is when the
    # spurious contact happens — used by detectors to distinguish "too
    # early" contact from the expected end-of-approach contact
    object_distance_at_contact: float = 0.15
    retreat_distance: float = 0.10
    retry_approach_angle_delta: float = 0.15  # radians
    retry_grasp_offset_delta: tuple[float, float, float] = (0.0, 0.0, 0.02)
    max_recovery_attempts: int = 3
    # anything detector/recovery-specific that doesn't fit the common fields
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    scenario_id: str
    family: str  # "N" | "R" | "F" | "P"
    seed: int = 0
    task_text: str = DEFAULT_TASK_TEXT
    control_hz: float = 30.0

    # object pose (world frame, meters / radians)
    object_x: float = 0.40
    object_y: float = 0.00
    object_z: float = 1.00
    object_roll: float = 0.0
    object_pitch: float = 0.0
    object_yaw: float = 0.0

    # arm initial pose overrides (joint_name -> value); None = use robot default
    left_start_pose: dict[str, float] | None = None
    right_start_pose: dict[str, float] | None = None

    # grasp frame axes, derived nominally from object bounding box + these offsets
    left_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_approach_angle: float = 0.0
    right_approach_angle: float = 0.0
    pregrasp_distance: float = 0.15

    lift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lift_height: float = 0.15
    hold_duration_s: float = 1.0

    # TRA-01: episode must end by placing into the green target ring
    target_ring_pose: tuple[float, float, float] = (0.55, 0.35, 1.00)

    anomaly: AnomalyConfig | None = None

    def __post_init__(self) -> None:
        if self.family not in ("N", "R", "F", "P"):
            raise ValueError(f"unknown family {self.family!r} for {self.scenario_id}")
        if self.family == "R" and self.anomaly is None:
            raise ValueError(f"{self.scenario_id} is family R but has no anomaly config")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        data = yaml.safe_load(Path(path).read_text())
        anomaly_data = data.pop("anomaly", None)
        anomaly = AnomalyConfig(**anomaly_data) if anomaly_data else None
        return cls(anomaly=anomaly, **data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def clone(self) -> "ScenarioConfig":
        return copy.deepcopy(self)


def sample_variant(
    base: ScenarioConfig,
    rng: random.Random,
    episode_id: str,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> ScenarioConfig:
    """Perturb a subset of a base scenario's fields within given ranges.

    Used by batch_generate.py to turn one hand-authored config (e.g.
    R11's base) into many episodes without writing new detector/recovery
    code — only the numbers change. `ranges` maps a field name (top level
    or "anomaly.<field>") to a (low, high) uniform sampling range;
    unspecified fields are copied from `base` unchanged.
    """
    cfg = base.clone()
    cfg.scenario_id = episode_id
    cfg.seed = rng.randint(0, 2**31 - 1)
    for key, (lo, hi) in (ranges or {}).items():
        value = rng.uniform(lo, hi)
        if key.startswith("anomaly."):
            if cfg.anomaly is None:
                raise ValueError(f"cannot sample {key!r}: base scenario has no anomaly")
            setattr(cfg.anomaly, key.split(".", 1)[1], value)
        else:
            setattr(cfg, key, value)
    return cfg
