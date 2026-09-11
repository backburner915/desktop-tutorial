"""Validate the RAW episode layout needed before a LeRobot/SmolVLA export."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def check_dataset(dataset: Path) -> dict:
    errors: list[dict[str, object]] = []
    checked = 0
    frame_total = 0
    outcome_counts: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    fps_values: list[float] = []
    for episode in sorted(dataset.glob("episode_*")):
        metadata_path = episode / "metadata.json"
        result_path = episode / "result.json"
        trajectory_path = episode / "trajectory.npz"
        if not (metadata_path.exists() and result_path.exists() and trajectory_path.exists()):
            errors.append({"episode": episode.name, "error": "missing metadata/result/trajectory"})
            continue
        checked += 1
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        outcome_counts[str(result.get("actual_outcome", "UNKNOWN"))] += 1
        fps_values.append(float(metadata.get("fps", 0.0)))
        with np.load(trajectory_path, allow_pickle=False) as data:
            required = {"observation.state", "action", "task", "timestamp", "frame_index", "phase"}
            missing = sorted(required - set(data.files))
            if missing:
                errors.append({"episode": episode.name, "error": f"missing fields: {missing}"})
                continue
            state = data["observation.state"]
            action = data["action"]
            timestamps = data["timestamp"]
            frames = data["frame_index"]
            n = int(state.shape[0])
            frame_total += n
            if state.shape != (n, 16):
                errors.append({"episode": episode.name, "error": f"observation.state shape {state.shape}"})
            if action.shape != (n, 16):
                errors.append({"episode": episode.name, "error": f"action shape {action.shape}"})
            if data["task"].shape != (n,):
                errors.append({"episode": episode.name, "error": f"task shape {data['task'].shape}"})
            if frames.shape != (n,) or not np.array_equal(frames, np.arange(n)):
                errors.append({"episode": episode.name, "error": "frame_index is not contiguous"})
            if timestamps.shape != (n,) or (n > 1 and np.any(np.diff(timestamps) <= 0)):
                errors.append({"episode": episode.name, "error": "timestamps are not strictly increasing"})
            numeric = [data[key] for key in data.files if np.issubdtype(data[key].dtype, np.number)]
            if any(not np.isfinite(array).all() for array in numeric):
                errors.append({"episode": episode.name, "error": "NaN or infinity in numeric trajectory"})
            expected = {"front": n, "left_wrist": n, "right_wrist": n}
            for camera, expected_count in expected.items():
                files = sorted((episode / "cameras" / camera).glob("*.png"))
                if len(files) != expected_count:
                    errors.append({"episode": episode.name, "error": f"{camera} image count {len(files)} != {expected_count}"})
                if files:
                    try:
                        from PIL import Image

                        with Image.open(files[0]) as image:
                            size = f"{image.width}x{image.height} {image.mode}"
                            image_sizes[f"{camera}:{size}"] += 1
                            if (image.width, image.height, image.mode) != (640, 480, "RGB"):
                                errors.append({"episode": episode.name, "error": f"{camera} first image is {size}, expected 640x480 RGB"})
                    except Exception as exc:
                        errors.append({"episode": episode.name, "error": f"cannot inspect {camera}: {exc}"})
        action_names = metadata.get("action_joint_names")
        if not isinstance(action_names, list) or len(action_names) != 16:
            errors.append({"episode": episode.name, "error": "metadata action_joint_names is not 16D"})
        if metadata.get("target_object") != "/World/TaskSetup/MovablePayloads/T03":
            errors.append({"episode": episode.name, "error": "target_object is not T03"})
    report = {
        "dataset": str(dataset.resolve()),
        "episodes_checked": checked,
        "total_frames": frame_total,
        "outcomes": dict(outcome_counts),
        "fps_values": sorted(set(fps_values)),
        "image_layouts": dict(image_sizes),
        "errors": errors,
        "valid": not errors and checked > 0,
        "lerobot_installed": _lerobot_installed(),
        "ready_for_lerobot_export": not errors and checked > 0,
        "note": "RAW is export-ready; an explicit LeRobot conversion step is still required.",
    }
    (dataset / "smolvla_compatibility.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _lerobot_installed() -> bool:
    try:
        import lerobot  # noqa: F401

        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    report = check_dataset(args.dataset)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
