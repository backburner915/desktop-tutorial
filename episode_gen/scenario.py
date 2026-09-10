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
    "Grasp the object with both hands, lift it, hold it steadily, and place it down."
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

    # N08/P24 axis: how far the two arms' phase transitions drift apart.
    # Field names match sim's own scenario schema (reference/scenario_t03.json,
    # metadata.json) rather than inventing a parallel convention.
    # coordination_mode: "synchronous" | "left_leads" | "right_leads"
    coordination_mode: str = "synchronous"
    # phase_offset: seconds the lagging arm's phase transitions are delayed
    # by when coordination_mode != "synchronous". 0.0 under "synchronous".
    phase_offset: float = 0.0

    # Where the object should end up after PLACE/RELEASE. Originally framed
    # as "the green target ring" in TRA-01's task text; that ring does not
    # exist in the scene (confirmed against a full prim scan — nothing named
    # ring/target_ring), so this is just a coordinate + radius/z-tolerance
    # proximity check (see IsaacLabR1Adapter._inside_target), not a literal
    # marked target. See docs/scene_grounding_t03.md.
    place_target_pose: tuple[float, float, float] = (-2.633, 3.506, 1.429)

    anomaly: AnomalyConfig | None = None
    # F02 only: a second, distinct anomaly that fires while already
    # recovering from `anomaly` (taxonomy §3). episode_runner only checks
    # this once recovery_count > 0, and a hit goes straight to FAILURE —
    # it is never itself retried.
    secondary_anomaly: AnomalyConfig | None = None

    def __post_init__(self) -> None:
        if self.family not in ("N", "R", "F", "P"):
            raise ValueError(f"unknown family {self.family!r} for {self.scenario_id}")
        if self.family == "R" and self.anomaly is None:
            raise ValueError(f"{self.scenario_id} is family R but has no anomaly config")
        if self.secondary_anomaly is not None and self.anomaly is None:
            raise ValueError(f"{self.scenario_id} has secondary_anomaly but no primary anomaly")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        data = yaml.safe_load(Path(path).read_text())
        anomaly_data = data.pop("anomaly", None)
        anomaly = AnomalyConfig(**anomaly_data) if anomaly_data else None
        secondary_data = data.pop("secondary_anomaly", None)
        secondary = AnomalyConfig(**secondary_data) if secondary_data else None
        return cls(anomaly=anomaly, secondary_anomaly=secondary, **data)

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
