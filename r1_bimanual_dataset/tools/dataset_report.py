from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _range(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return {"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values))}


def build_report(dataset_dir: Path) -> dict:
    rows = []
    phases = Counter()
    failure_modes = Counter()
    failure_reasons = Counter()
    object_positions = []
    grasp_offsets = []
    approach_angles = []
    phase_offsets = []
    durations = []
    camera_missing = controller_failures = ik_failures = reset_failures = 0
    valid_episodes = 0
    nan_failures = 0
    for episode_dir in sorted(dataset_dir.glob("episode_*")):
        metadata_path = episode_dir / "metadata.json"
        result_path = episode_dir / "result.json"
        if not metadata_path.exists() or not result_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        scenario = metadata.get("scenario", {})
        outcome = result.get("actual_outcome", metadata.get("actual_outcome", "UNKNOWN"))
        row = {
            "episode_id": episode_dir.name,
            "scenario_id": metadata.get("scenario_id"),
            "intended_outcome": scenario.get("intended_outcome"),
            "actual_outcome": outcome,
            "failure_mode": scenario.get("failure_mode"),
            "failure_reason": result.get("failure_reason"),
            "failure_phase": result.get("failure_phase"),
            "frame_count": result.get("frame_count", 0),
            "duration_s": result.get("duration_s", 0.0),
            "camera_missing_frames": result.get("camera_missing_frames", 0),
            "controller_failures": result.get("controller_failures", 0),
            "ik_failures": result.get("ik_failures", 0),
            "reset_failures": result.get("reset_failures", 0),
        }
        rows.append(row)
        failure_modes[scenario.get("failure_mode", "unknown")] += 1
        if result.get("failure_reason"):
            failure_reasons[result["failure_reason"]] += 1
        camera_missing += int(row["camera_missing_frames"] or 0)
        controller_failures += int(row["controller_failures"] or 0)
        ik_failures += int(row["ik_failures"] or 0)
        reset_failures += int(row["reset_failures"] or 0)
        durations.append(float(row["duration_s"] or 0.0))
        if scenario.get("object_position"):
            object_positions.append(scenario["object_position"])
        if scenario.get("left_grasp_offset") and scenario.get("right_grasp_offset"):
            grasp_offsets.extend([scenario["left_grasp_offset"], scenario["right_grasp_offset"]])
        if scenario.get("left_approach_angle") and scenario.get("right_approach_angle"):
            approach_angles.extend([scenario["left_approach_angle"], scenario["right_approach_angle"]])
        if scenario.get("phase_offset") is not None:
            phase_offsets.append(scenario["phase_offset"])
        npz_path = episode_dir / "trajectory.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as data:
                if "phase" in data:
                    phases.update(data["phase"].tolist())
                if "observation.state" in data and data["observation.state"].shape[1] == 16:
                    valid_episodes += 1
                if any(np.isnan(data[key]).any() for key in data.files if np.issubdtype(data[key].dtype, np.number)):
                    nan_failures += 1
    counts = Counter(row["actual_outcome"] for row in rows)
    intended_counts = Counter(row["intended_outcome"] for row in rows)
    total = len(rows)
    report = {
        "total": total,
        "intended_success": intended_counts.get("success", 0),
        "intended_failure": intended_counts.get("failure", 0),
        "actual_success": counts.get("SUCCESS", 0),
        "actual_failure": counts.get("FAILURE", 0),
        "invalid": counts.get("INVALID", 0),
        "actual_success_rate": counts.get("SUCCESS", 0) / total if total else 0.0,
        "failure_mode_counts": dict(failure_modes),
        "actual_failure_reason_counts": dict(failure_reasons),
        "phase_counts": dict(phases),
        "object_position_distribution": _range(object_positions),
        "grasp_offset_distribution": _range(grasp_offsets),
        "approach_angle_distribution_rad": _range(approach_angles),
        "phase_offset_distribution_s": _range(phase_offsets),
        "episode_duration_s": _range(durations),
        "camera_missing_frames": camera_missing,
        "controller_failures": controller_failures,
        "ik_failures": ik_failures,
        "reset_failures": reset_failures,
        "nan_episodes": nan_failures,
        "episodes_with_16d_observation_state": valid_episodes,
    }
    manifest_path = dataset_dir / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["episode_id"])
        writer.writeheader()
        writer.writerows(rows)
    (dataset_dir / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.dataset_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
