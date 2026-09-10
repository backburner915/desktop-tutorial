"""N02-N10 sampling: turns N01 (nominal) into batches of episodes that
vary one axis of the taxonomy's Normal family at a time.

Every range here is the **provisional fallback convention** from
docs/sim_data_request_v1.md §3, not a measured safety boundary — sim
confirmed no real "still-graspable" workspace/tolerance data exists yet
(scan_r1_reachability.py doesn't do what its name suggests; the one
real non-nominal episode on record is a *failure*, not a boundary
sample). Treat every episode this module produces as provisional until
that data lands, and keep every actual outcome recorded — don't let
"we sampled it" quietly turn into "we validated it".

Each function takes the same shape: (base, rng, n) -> list[ScenarioConfig].
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from episode_gen.scenario import ScenarioConfig

# T03-appropriate arm-joint jitter magnitude (sim_data_request_v1.md §3
# N04 gives ±0.045 rad for T03, ±0.08 rad for T01/T02 — this module
# targets T03, the object our grounded configs use).
_N04_JOINT_JITTER_RAD = 0.045
_LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 7)]
_RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 7)]


def _labeled(base: ScenarioConfig, rng: random.Random, family_id: str, i: int) -> ScenarioConfig:
    cfg = base.clone()
    cfg.scenario_id = f"{family_id}_{i:04d}"
    cfg.family = "N"
    cfg.anomaly = None
    cfg.secondary_anomaly = None
    cfg.seed = rng.randint(0, 2**31 - 1)
    return cfg


def generate_n02_object_xy_offset(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """Object x/y offset from nominal. z fixed (not independently jittered)."""
    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N02", i)
        cfg.object_x = base.object_x + rng.uniform(-0.025, 0.025)
        cfg.object_y = base.object_y + rng.uniform(-0.025, 0.025)
        out.append(cfg)
    return out


def generate_n03_object_yaw(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """Object yaw perturbation. roll/pitch stay 0 (matches what's actually
    been exercised so far); extend with roll,pitch ~ U(-0.05,+0.05) rad
    later if needed — see sim_data_request_v1.md §3 N03."""
    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N03", i)
        cfg.object_yaw = rng.uniform(-0.20, 0.20)
        out.append(cfg)
    return out


def generate_n04_arm_start_jitter(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """Per-joint jitter on both arms' initial pose. Gripper axes untouched."""
    from episode_gen.sim_adapter import DEFAULT_QPOS

    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N04", i)
        cfg.left_start_pose = {
            j: DEFAULT_QPOS[j] + rng.uniform(-_N04_JOINT_JITTER_RAD, _N04_JOINT_JITTER_RAD)
            for j in _LEFT_ARM_JOINTS
        }
        cfg.right_start_pose = {
            j: DEFAULT_QPOS[j] + rng.uniform(-_N04_JOINT_JITTER_RAD, _N04_JOINT_JITTER_RAD)
            for j in _RIGHT_ARM_JOINTS
        }
        out.append(cfg)
    return out


def generate_n05_grasp_offset(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """Per-side grasp point offset. Ranges are asymmetric per axis
    (x tighter than y/z) per the convention table, not a cube."""
    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N05", i)
        cfg.left_grasp_offset = (
            rng.uniform(-0.006, 0.006), rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004),
        )
        cfg.right_grasp_offset = (
            rng.uniform(-0.006, 0.006), rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004),
        )
        out.append(cfg)
    return out


def generate_n06_approach_angle(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """Per-side approach angle perturbation.

    Simplification: sim's real scenario schema uses a 3-component
    (roll/pitch/yaw) approach_angle per side; ScenarioConfig.left/
    right_approach_angle is a single scalar. This samples that scalar
    within the same +-0.08 rad convention rather than the full 3-axis
    range — flagged here rather than silently treated as equivalent.
    """
    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N06", i)
        cfg.left_approach_angle = rng.uniform(-0.08, 0.08)
        cfg.right_approach_angle = rng.uniform(-0.08, 0.08)
        out.append(cfg)
    return out


def generate_n07_pregrasp_distance(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N07", i)
        cfg.pregrasp_distance = rng.uniform(0.090, 0.115)
        out.append(cfg)
    return out


def generate_n08_coordination(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """coordination_mode distribution: 80% synchronous, 10% left_leads,
    10% right_leads; phase_offset magnitude ~ U(0, 0.060)s when async."""
    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N08", i)
        roll = rng.random()
        if roll < 0.8:
            cfg.coordination_mode = "synchronous"
            cfg.phase_offset = 0.0
        else:
            cfg.coordination_mode = "left_leads" if roll < 0.9 else "right_leads"
            cfg.phase_offset = rng.uniform(0.0, 0.060)
        out.append(cfg)
    return out


def generate_n09_workspace_edge(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """N02's box perimeter, treated as an *assumed* edge (not a measured
    reachability boundary) per sim_data_request_v1.md §3 N09: points where
    max(|dx|,|dy|) == 0.025, stepped every 0.0125. Deterministic — cycles
    through the 16 perimeter points if n > 16, truncates if n < 16."""
    r, s = 0.025, 0.0125
    steps = [-r, -r + s, 0.0, r - s, r]
    points: list[tuple[float, float]] = []
    for dx in steps:
        points.append((dx, r))
        points.append((dx, -r))
    for dy in steps[1:-1]:
        points.append((r, dy))
        points.append((-r, dy))
    points = sorted(set(points))

    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N09", i)
        dx, dy = points[i % len(points)]
        cfg.object_x = base.object_x + dx
        cfg.object_y = base.object_y + dy
        out.append(cfg)
    return out


def generate_n10_table_clearance(base: ScenarioConfig, rng: random.Random, n: int) -> list[ScenarioConfig]:
    """approach_clearance_m at the suggested 5/7.5/10/12.5/15mm points,
    never below 3mm. See ScenarioConfig.approach_clearance_m docstring —
    this is sampling metadata, not yet consumed by any adapter's actual
    trajectory shaping."""
    candidates = [0.005, 0.0075, 0.010, 0.0125, 0.015]
    out = []
    for i in range(n):
        cfg = _labeled(base, rng, "N10", i)
        cfg.approach_clearance_m = candidates[i % len(candidates)]
        out.append(cfg)
    return out


GENERATORS = {
    "N02": generate_n02_object_xy_offset,
    "N03": generate_n03_object_yaw,
    "N04": generate_n04_arm_start_jitter,
    "N05": generate_n05_grasp_offset,
    "N06": generate_n06_approach_angle,
    "N07": generate_n07_pregrasp_distance,
    "N08": generate_n08_coordination,
    "N09": generate_n09_workspace_edge,
    "N10": generate_n10_table_clearance,
}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-id", required=True, choices=sorted(GENERATORS), help="which N-axis to sample")
    parser.add_argument("--base", default="configs/scenarios/n01_nominal.yaml")
    parser.add_argument("--out", required=True, help="output directory for generated YAMLs")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = ScenarioConfig.from_yaml(args.base)
    rng = random.Random(args.seed)
    configs = GENERATORS[args.n_id](base, rng, args.n)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        cfg.to_yaml(out / f"{cfg.scenario_id}.yaml")
    print(f"wrote {len(configs)} {args.n_id} configs to {args.out} (PROVISIONAL ranges — see sim_data_request_v1.md)")


if __name__ == "__main__":
    _main()
