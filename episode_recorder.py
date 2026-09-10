from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import EpisodeResult


class EpisodeRecorder:
    """Write one self-contained RAW episode and keep image/state alignment."""

    def __init__(self, dataset_dir: str | Path, scenario: Any, scene_cfg: dict[str, Any], nominal: Any, fps: float):
        self.dataset_dir = Path(dataset_dir)
        self.scenario = scenario
        self.scene_cfg = scene_cfg
        self.nominal = nominal
        self.fps = float(fps)
        self.episode_dir = self.dataset_dir / f"episode_{int(scenario.scenario_id):06d}"
        self.camera_dirs = {name: self.episode_dir / "cameras" / name for name in scene_cfg["cameras"]}
        self.frames: list[dict[str, Any]] = []
        self.missing_camera_frames = 0

    def start(self, runtime_metadata: dict[str, Any] | None = None) -> None:
        from .reference_policy import require_physical_dataset_mode
        require_physical_dataset_mode(self.scene_cfg)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        for path in self.camera_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset_version": self.scene_cfg.get("dataset_version"),
            "scenario_id": self.scenario.scenario_id,
            "seed": self.scenario.seed,
            "stage": self.scene_cfg.get("stage_url", self.scene_cfg.get("stage_path")),
            "robot": self.scene_cfg.get("robot"),
            "robot_prim": self.scene_cfg.get("robot_prim"),
            "target_object": self.scene_cfg.get("target_object"),
            "action_joint_names": self.scene_cfg["action_joint_names"],
            "instruction": self.scenario.instruction,
            "scenario": self.scenario.to_dict(),
            "nominal_grasp": self.nominal.as_dict() if self.nominal is not None else None,
            "fps": self.fps,
            "camera_paths": self.scene_cfg["cameras"],
            "camera_alignment_rule": "one_simulation_step_then_all_three_reads",
            "rollout_adapter": {
                "physical_rollout": bool(self.scene_cfg.get("physical_rollout", False)),
                "scripted_grasp_attachment": bool(self.scene_cfg.get("scripted_grasp_attachment", False)),
            },
            "runtime": runtime_metadata or {},
        }
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Write a checkpoint immediately.  Isaac Sim can terminate in native
        # GPU/Vulkan code, where Python finally/except blocks cannot run.  A
        # checkpoint makes that condition distinguishable from a missing
        # episode instead of leaving only a metadata directory behind.
        self._write_checkpoint_result("episode_started", completed=False)

    def _write_checkpoint_result(self, reason: str, completed: bool) -> None:
        np.savez_compressed(
            self.episode_dir / "trajectory.npz",
            timestamp=np.empty((0,), dtype=np.float64),
            frame_index=np.empty((0,), dtype=np.int64),
            **{
                "observation.state": np.empty((0, 16), dtype=np.float32),
                "action": np.empty((0, 16), dtype=np.float32),
                "phase": np.empty((0,), dtype="U16"),
            },
        )
        result = {
            "actual_outcome": "INVALID",
            "success_flags": {},
            "failure_reason": None,
            "failure_phase": None,
            "invalid_reason": reason,
            "frame_count": 0,
            "duration_s": 0.0,
            "camera_missing_frames": 0,
            "controller_failures": 0,
            "ik_failures": 0,
            "reset_failures": 0,
            "completed": bool(completed),
        }
        (self.episode_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def append(
        self,
        camera_sample: dict[str, Any],
        state16: np.ndarray,
        velocity16: np.ndarray,
        action16: np.ndarray,
        all_joint_positions: np.ndarray,
        all_joint_velocities: np.ndarray,
        left_eef_pose: np.ndarray,
        right_eef_pose: np.ndarray,
        object_pose: np.ndarray,
        object_velocity: np.ndarray,
        phase: str,
        contact_state: dict[str, Any],
    ) -> None:
        frame_index = len(self.frames)
        image_paths: dict[str, str] = {}
        for name, image in camera_sample["frames"].items():
            relative = Path("cameras") / name / f"{frame_index:06d}.png"
            path = self.episode_dir / relative
            try:
                from PIL import Image

                Image.fromarray(image).save(path)
            except ImportError:
                # Keep a deterministic raw fallback when Pillow is unavailable.
                relative = Path("cameras") / name / f"{frame_index:06d}.npy"
                np.save(self.episode_dir / relative, image)
            image_paths[name] = relative.as_posix()
        if camera_sample["missing"]:
            self.missing_camera_frames += 1
        self.frames.append(
            {
                "timestamp": float(camera_sample["timestamp"]),
                "frame_index": frame_index,
                "image_paths": image_paths,
                "joint_position_16d": np.asarray(state16, dtype=np.float32),
                "joint_velocity_16d": np.asarray(velocity16, dtype=np.float32),
                "action_16d": np.asarray(action16, dtype=np.float32),
                "all_joint_positions": np.asarray(all_joint_positions, dtype=np.float32),
                "all_joint_velocities": np.asarray(all_joint_velocities, dtype=np.float32),
                "left_eef_pose": np.asarray(left_eef_pose, dtype=np.float32),
                "right_eef_pose": np.asarray(right_eef_pose, dtype=np.float32),
                "object_pose": np.asarray(object_pose, dtype=np.float32),
                "object_velocity": np.asarray(object_velocity, dtype=np.float32),
                "phase": str(phase),
                "contact_state": contact_state,
            }
        )

    def save_result(self, result: EpisodeResult, actual_outcome: str | None = None) -> None:
        outcome = actual_outcome or result.actual_outcome
        numeric = {}
        if self.frames:
            numeric = {
                "timestamp": np.asarray([x["timestamp"] for x in self.frames], dtype=np.float64),
                "frame_index": np.asarray([x["frame_index"] for x in self.frames], dtype=np.int64),
                "joint_position_16d": np.stack([x["joint_position_16d"] for x in self.frames]),
                "joint_velocity_16d": np.stack([x["joint_velocity_16d"] for x in self.frames]),
                "action_16d": np.stack([x["action_16d"] for x in self.frames]),
                "all_joint_positions": np.stack([x["all_joint_positions"] for x in self.frames]),
                "all_joint_velocities": np.stack([x["all_joint_velocities"] for x in self.frames]),
                "left_eef_pose": np.stack([x["left_eef_pose"] for x in self.frames]),
                "right_eef_pose": np.stack([x["right_eef_pose"] for x in self.frames]),
                "object_pose": np.stack([x["object_pose"] for x in self.frames]),
                "object_velocity": np.stack([x["object_velocity"] for x in self.frames]),
                "phase": np.asarray([x["phase"] for x in self.frames], dtype="U16"),
                "left_contact_proxy": np.asarray([x["contact_state"].get("left_contact_proxy", False) for x in self.frames], dtype=bool),
                "right_contact_proxy": np.asarray([x["contact_state"].get("right_contact_proxy", False) for x in self.frames], dtype=bool),
                "left_contact_distance_m": np.asarray([x["contact_state"].get("left_distance_m", np.nan) for x in self.frames], dtype=np.float32),
                "right_contact_distance_m": np.asarray([x["contact_state"].get("right_distance_m", np.nan) for x in self.frames], dtype=np.float32),
                # Stable names for a later LeRobot/SmolVLA converter.
                "observation.state": np.stack([x["joint_position_16d"] for x in self.frames]),
                "action": np.stack([x["action_16d"] for x in self.frames]),
                "task": np.asarray([self.scenario.instruction] * len(self.frames), dtype="U256"),
            }
        np.savez_compressed(self.episode_dir / "trajectory.npz", **numeric)
        result_dict = result.as_dict()
        result_dict["actual_outcome"] = outcome
        result_dict["frame_count"] = len(self.frames)
        result_dict["camera_missing_frames"] = self.missing_camera_frames
        result_dict["completed"] = True
        (self.episode_dir / "result.json").write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metadata_path = self.episode_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "actual_outcome": outcome,
                    "frame_count": len(self.frames),
                    "intended_outcome": self.scenario.intended_outcome,
                    "failure_mode": self.scenario.failure_mode,
                    "completed": True,
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
