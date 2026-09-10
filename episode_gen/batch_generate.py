"""Turn one hand-authored scenario config into many episode configs.

This is the "closure" mechanism the taxonomy doc asks for: R11 stays
one detector/recovery pair in code, while dozens of concrete episodes
(different contact arm, contact distance, retreat distance, retry
angle...) come out of sampling a base YAML — no new Python per episode.

Usage:
    python -m episode_gen.batch_generate \
        --base configs/scenarios/r11_premature_table_contact.yaml \
        --out configs/scenarios/generated/r11 \
        --n 40 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from episode_gen.scenario import ScenarioConfig, sample_variant

# Default sampling ranges for the R11 closure. Keys are ScenarioConfig
# field names, or "anomaly.<field>" for AnomalyConfig fields.
R11_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "object_x": (0.32, 0.48),
    "object_y": (-0.08, 0.08),
    "object_yaw": (-0.3, 0.3),
    "anomaly.object_distance_at_contact": (0.10, 0.25),
    "anomaly.retreat_distance": (0.06, 0.16),
    "anomaly.retry_approach_angle_delta": (0.08, 0.25),
}


def generate_batch(
    base_path: str,
    out_dir: str,
    n: int,
    seed: int = 0,
    ranges: dict[str, tuple[float, float]] | None = None,
    alternate_contact_arm: bool = True,
) -> list[str]:
    base = ScenarioConfig.from_yaml(base_path)
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for i in range(n):
        episode_id = f"{base.scenario_id}_{i:04d}"
        cfg = sample_variant(base, rng, episode_id, ranges or R11_DEFAULT_RANGES)
        if alternate_contact_arm and cfg.anomaly is not None and cfg.anomaly.contact_arm != "both":
            cfg.anomaly.contact_arm = "left" if i % 2 == 0 else "right"
        path = out / f"{episode_id}.yaml"
        cfg.to_yaml(path)
        written.append(str(path))
    return written


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base scenario YAML, e.g. configs/scenarios/r11_premature_table_contact.yaml")
    parser.add_argument("--out", required=True, help="output directory for generated YAMLs")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    written = generate_batch(args.base, args.out, args.n, args.seed)
    print(f"wrote {len(written)} scenario configs to {args.out}")


if __name__ == "__main__":
    _main()
