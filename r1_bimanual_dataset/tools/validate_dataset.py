from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate(dataset_dir: Path) -> dict:
    errors = []
    checked = 0
    for episode in sorted(dataset_dir.glob("episode_*")):
        metadata_path = episode / "metadata.json"
        result_path = episode / "result.json"
        trajectory_path = episode / "trajectory.npz"
        if not metadata_path.exists() or not result_path.exists() or not trajectory_path.exists():
            errors.append({"episode": episode.name, "error": "missing metadata/result/trajectory"})
            continue
        checked += 1
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        # Older completed episodes predate the checkpoint field.  Only an
        # explicit false value is an aborted/incomplete run.
        if "completed" in result and not bool(result.get("completed")):
            errors.append({"episode": episode.name, "error": "episode checkpoint is incomplete (process aborted before save_result)"})
        for key in ("scenario_id", "seed", "action_joint_names", "instruction", "scenario", "fps", "frame_count", "actual_outcome"):
            if key not in metadata:
                errors.append({"episode": episode.name, "error": f"metadata missing {key}"})
        if len(metadata.get("action_joint_names", [])) != 16:
            errors.append({"episode": episode.name, "error": "action_joint_names is not 16D"})
        if "actual_outcome" not in result:
            errors.append({"episode": episode.name, "error": "result missing actual_outcome"})
        try:
            with np.load(trajectory_path, allow_pickle=False) as data:
                zero_frame_invalid = result.get("actual_outcome") == "INVALID" and int(result.get("frame_count", 0)) == 0
                if zero_frame_invalid:
                    # INVALID_CONFIGURATION is intentionally allowed to have
                    # no simulation frames; it is not a physical failure.
                    continue
                required = {"timestamp", "frame_index", "observation.state", "action", "all_joint_positions", "all_joint_velocities", "phase", "left_contact_proxy", "right_contact_proxy"}
                missing = sorted(required - set(data.files))
                if missing:
                    errors.append({"episode": episode.name, "error": f"trajectory missing {missing}"})
                if "observation.state" in data and data["observation.state"].ndim == 2 and data["observation.state"].shape[1] != 16:
                    errors.append({"episode": episode.name, "error": "observation.state is not 16D"})
                if "action" in data and data["action"].ndim == 2 and data["action"].shape[1] != 16:
                    errors.append({"episode": episode.name, "error": "action is not 16D"})
                frame_count = len(data["timestamp"]) if "timestamp" in data else 0
                for key in ("frame_index", "observation.state", "action", "phase", "left_contact_proxy", "right_contact_proxy"):
                    if key in data and len(data[key]) != frame_count:
                        errors.append({"episode": episode.name, "error": f"frame alignment mismatch for {key}"})
                if "timestamp" in data and len(data["timestamp"]) > 1 and np.any(np.diff(data["timestamp"]) <= 0):
                    errors.append({"episode": episode.name, "error": "timestamp is not strictly increasing"})
                for key in data.files:
                    if np.issubdtype(data[key].dtype, np.number) and not np.isfinite(data[key]).all():
                        errors.append({"episode": episode.name, "error": f"NaN or infinity in {key}"})
        except Exception as exc:
            errors.append({"episode": episode.name, "error": f"cannot inspect trajectory: {exc}"})
        for camera in ("front", "left_wrist", "right_wrist"):
            if not (episode / "cameras" / camera).exists() and int(result.get("frame_count", 0)) > 0:
                errors.append({"episode": episode.name, "error": f"missing camera directory {camera}"})
    report = {"dataset": str(dataset_dir.resolve()), "episodes_checked": checked, "valid": not errors, "errors": errors}
    (dataset_dir / "dataset_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
