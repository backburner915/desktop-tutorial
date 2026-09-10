"""Per-episode writer implementing docs/dataset_spec_v0.1.md.

Two output streams, sharing `frame_index` as the join key:
  - episode.jsonl    -> exactly the LeRobot v3 policy fields (§4)
  - diagnostics.jsonl -> sim ground-truth / taxonomy bookkeeping (§5),
                          never fed to the policy

This module intentionally does not depend on the real `lerobot` package
(not installed in this sandbox) or on numpy/pyarrow. `images` are stored
as whatever the SimAdapter hands back (a real adapter can pass real
arrays; MockSimAdapter passes a shape/dtype descriptor). Swapping this
for a real `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`
writer is a drop-in replacement behind the same `EpisodeWriter` API —
do that once running against real Isaac Sim frames.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONTROL_HZ = 30.0
FRAME_PERIOD_S = 1.0 / CONTROL_HZ


class DatasetContractError(ValueError):
    """A frame or episode violates docs/dataset_spec_v0.1.md."""


@dataclass
class EpisodeResult:
    episode_index: int
    scenario_id: str
    num_frames: int
    success: bool
    failure_reason: str | None
    recovery_count: int
    out_dir: str


def _validate_vector16(name: str, values: list[float]) -> None:
    if len(values) != 16:
        raise DatasetContractError(f"{name} must be float32[16], got length {len(values)}")
    for v in values:
        if math.isnan(v) or math.isinf(v):
            raise DatasetContractError(f"{name} contains NaN/Inf: {values}")


@dataclass
class EpisodeWriter:
    out_dir: str
    episode_index: int
    task_index: int
    task_text: str
    scenario_id: str
    control_hz: float = CONTROL_HZ
    timestamp_tolerance_s: float = 0.003  # slack around the 33.333ms nominal period

    _frames: list[dict[str, Any]] = field(default_factory=list, init=False)
    _diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False)
    _index_cursor: int = field(default=0, init=False)

    def append_frame(
        self,
        images: dict[str, Any],
        state: list[float],
        action: list[float],
        timestamp: float,
        diagnostics: dict[str, Any],
    ) -> None:
        _validate_vector16("observation.state", state)
        _validate_vector16("action", action)
        missing = {"front", "left_wrist", "right_wrist"} - set(images)
        if missing:
            raise DatasetContractError(f"missing camera(s) in frame: {missing}")

        frame_index = len(self._frames)
        policy_frame = {
            "observation.images.front": images["front"],
            "observation.images.left_wrist": images["left_wrist"],
            "observation.images.right_wrist": images["right_wrist"],
            "observation.state": list(state),
            "action": list(action),
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": self.episode_index,
            "index": self._index_cursor,
            "task_index": self.task_index,
        }
        diag_frame = {"frame_index": frame_index, "episode_index": self.episode_index, **diagnostics}

        self._frames.append(policy_frame)
        self._diagnostics.append(diag_frame)
        self._index_cursor += 1

    def _validate_timing(self) -> None:
        for prev, cur in zip(self._frames, self._frames[1:]):
            dt = cur["timestamp"] - prev["timestamp"]
            if abs(dt - FRAME_PERIOD_S) > self.timestamp_tolerance_s:
                raise DatasetContractError(
                    f"frame interval {dt * 1000:.2f}ms at frame_index={cur['frame_index']} "
                    f"deviates from the required 33.333ms (30Hz) by more than "
                    f"{self.timestamp_tolerance_s * 1000:.1f}ms"
                )

    def finalize(self, success: bool, failure_reason: str | None, recovery_count: int) -> EpisodeResult:
        if not self._frames:
            raise DatasetContractError("episode has zero frames")
        if not success and failure_reason is None:
            raise DatasetContractError("failure_reason is required when success=False")
        self._validate_timing()

        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "episode.jsonl").open("w") as f:
            for frame in self._frames:
                f.write(json.dumps(frame, default=str) + "\n")
        with (out / "diagnostics.jsonl").open("w") as f:
            for frame in self._diagnostics:
                f.write(json.dumps(frame, default=str) + "\n")

        meta = {
            "episode_index": self.episode_index,
            "scenario_id": self.scenario_id,
            "task_index": self.task_index,
            "task_text": self.task_text,
            "num_frames": len(self._frames),
            "fps": self.control_hz,
            "episode_success": success,
            "failure_reason": failure_reason,
            "recovery_count": recovery_count,
        }
        (out / "episode_meta.json").write_text(json.dumps(meta, indent=2))

        return EpisodeResult(
            episode_index=self.episode_index,
            scenario_id=self.scenario_id,
            num_frames=len(self._frames),
            success=success,
            failure_reason=failure_reason,
            recovery_count=recovery_count,
            out_dir=str(out),
        )
