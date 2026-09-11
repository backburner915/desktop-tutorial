from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import ScenarioConfig, load_scene_config


DEFAULT_FAILURE_TEMPLATES = [
    "left_grasp_position_error",
    "right_grasp_position_error",
    "left_gripper_underclose",
    "phase_mismatch",
    "insufficient_lift",
]


def _target_center(scene_cfg: dict, preflight_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not preflight_path.exists():
        raise FileNotFoundError(f"preflight report is required: {preflight_path}")
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    for item in report.get("object_candidates", []):
        if item.get("path") == scene_cfg["target_object"]:
            bounds = item["bbox_world"]
            minimum = np.asarray(bounds["min"], dtype=np.float64)
            maximum = np.asarray(bounds["max"], dtype=np.float64)
            return (minimum + maximum) / 2.0, maximum - minimum
    raise ValueError(f"target object not present in preflight report: {scene_cfg['target_object']}")


def _home(scene_cfg: dict, side: str) -> np.ndarray:
    return np.asarray([scene_cfg["home_joint_positions"][f"{side}_arm_joint{i}"] for i in range(1, 7)], dtype=np.float64)


def _level(rng: random.Random, values: dict[str, float]) -> tuple[str, float]:
    name = rng.choices(list(values), weights=[0.50, 0.35, 0.15], k=1)[0]
    return name, values[name]


def _failure_magnitude(rng: random.Random, mode: str) -> tuple[str, float]:
    table = {
        "position": {"small": 0.012, "medium": 0.025, "large": 0.045},
        "angle": {"small": 0.10, "medium": 0.25, "large": 0.45},
        "gripper": {"small": 0.004, "medium": 0.010, "large": 0.018},
        "phase": {"small": 0.08, "medium": 0.18, "large": 0.35},
        "lift": {"small": 0.20, "medium": 0.35, "large": 0.55},
    }
    if "angle" in mode or "approach" in mode:
        kind = "angle"
    elif "gripper" in mode or mode == "slip":
        kind = "gripper"
    elif "phase" in mode or "early" in mode or "late" in mode:
        kind = "phase"
    elif mode == "insufficient_lift":
        kind = "lift"
    else:
        kind = "position"
    return _level(rng, table[kind])


def generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene_cfg = load_scene_config(args.scene_config)
    if args.target_object:
        scene_cfg = dict(scene_cfg)
        scene_cfg["target_object"] = args.target_object
    center, dimensions = _target_center(scene_cfg, Path(args.preflight))
    master = random.Random(args.master_seed)
    success_count = int(round(args.num_scenarios * args.success_ratio))
    failure_count = args.num_scenarios - success_count
    if success_count + failure_count != args.num_scenarios:
        raise ValueError("scenario counts do not sum to NUM_SCENARIOS")
    failure_modes = args.failure_modes or DEFAULT_FAILURE_TEMPLATES
    files = []

    for scenario_id in range(1, args.num_scenarios + 1):
        intended = "success" if scenario_id <= success_count else "failure"
        seed = master.randrange(0, 2**31 - 1)
        rng = random.Random(seed)
        object_position = center.copy()
        # Position/orientation variation is layered and remains close to the
        # verified support pose.  z remains on the support surface.
        object_position[:2] += np.asarray([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)])
        object_orientation = [0.0, 0.0, rng.uniform(-0.20, 0.20)]
        # T03 is the small object used by the calibrated physical reference.
        # Keep start perturbations conservative enough to remain in the same
        # reachable basin while still making every scenario distinguishable.
        start_jitter = 0.045 if str(scene_cfg["target_object"]).rstrip("/").endswith("T03") else 0.08
        left_start = _home(scene_cfg, "left") + np.asarray([rng.uniform(-start_jitter, start_jitter) for _ in range(6)])
        right_start = _home(scene_cfg, "right") + np.asarray([rng.uniform(-start_jitter, start_jitter) for _ in range(6)])
        left_offset = np.asarray([rng.uniform(-0.006, 0.006), rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004)])
        right_offset = np.asarray([rng.uniform(-0.006, 0.006), rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004)])
        left_angle = np.asarray([rng.uniform(-0.08, 0.08) for _ in range(3)])
        right_angle = np.asarray([rng.uniform(-0.08, 0.08) for _ in range(3)])
        lift = np.asarray([rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015), rng.uniform(0.09, 0.14)])
        coordination = rng.choices(["synchronous", "left_leads", "right_leads"], weights=[0.8, 0.1, 0.1], k=1)[0]
        phase_offset = rng.uniform(0.0, 0.06) if coordination != "synchronous" else 0.0
        failure_mode = "none"
        failure_magnitude = 0.0
        sampling_level = "small_or_medium"
        if intended == "failure":
            failure_mode = failure_modes[(scenario_id - success_count - 1) % len(failure_modes)]
            sampling_level, failure_magnitude = _failure_magnitude(rng, failure_mode)
            if failure_mode == "phase_mismatch":
                coordination = "left_leads" if rng.random() < 0.5 else "right_leads"
                phase_offset = failure_magnitude

        scenario = ScenarioConfig(
            scenario_id=scenario_id,
            intended_outcome=intended,
            object_position=object_position.tolist(),
            object_orientation=object_orientation,
            left_start_pose=left_start.tolist(),
            right_start_pose=right_start.tolist(),
            left_grasp_offset=left_offset.tolist(),
            right_grasp_offset=right_offset.tolist(),
            left_approach_angle=left_angle.tolist(),
            right_approach_angle=right_angle.tolist(),
            pregrasp_distance=0.10 + rng.uniform(-0.01, 0.015),
            lift_vector=lift.tolist(),
            coordination_mode=coordination,
            phase_offset=phase_offset,
            gripper_target={"left": list(scene_cfg["gripper_closed"]), "right": list(scene_cfg["gripper_closed"])},
            failure_mode=failure_mode,
            failure_magnitude=failure_magnitude,
            seed=seed,
            target_object=scene_cfg["target_object"],
            metadata={
                "sampling_level": sampling_level,
                "object_bbox_dimensions_m": dimensions.tolist(),
                "master_seed": args.master_seed,
                "design_ratio": "6:4",
            },
        )
        path = output / f"scenario_{scenario_id:06d}.json"
        scenario.to_json(path)
        files.append(path)
    manifest = {
        "master_seed": args.master_seed,
        "num_scenarios": args.num_scenarios,
        "intended_success": success_count,
        "intended_failure": failure_count,
        "success_ratio": args.success_ratio,
        "failure_modes": failure_modes,
        "files": [p.name for p in files],
    }
    (output / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate constrained R1 bimanual scenario JSON files")
    parser.add_argument("--num-scenarios", type=int, required=True)
    parser.add_argument("--success-ratio", type=float, default=0.6)
    parser.add_argument("--master-seed", type=int, default=20260904)
    parser.add_argument("--scene-config", type=Path, default=Path("scene_config.json"))
    parser.add_argument("--preflight", type=Path, default=Path("preflight_report.json"))
    parser.add_argument("--output", type=Path, default=Path("scenarios"))
    parser.add_argument("--target-object", type=str, default=None)
    parser.add_argument("--failure-modes", nargs="*")
    args = parser.parse_args()
    if not 0.0 <= args.success_ratio <= 1.0:
        raise ValueError("success ratio must be in [0, 1]")
    if args.num_scenarios <= 0:
        raise ValueError("num-scenarios must be positive")
    generate(args)


if __name__ == "__main__":
    main()
