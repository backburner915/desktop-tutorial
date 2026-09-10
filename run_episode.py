#!/usr/bin/env python3
"""Run ONE scenario config through the FSM and write one episode.

    python scripts/run_episode.py \
        --scenario configs/scenarios/r11_premature_table_contact.yaml \
        --out-dir out/episodes --episode-index 0 --adapter mock

--adapter mock   : MockSimAdapter, runs anywhere (no Isaac Sim) — control-flow
                   dry-run only, does NOT produce physically real trajectories
                   or real images. Useful to sanity-check a config before
                   handing it to Isaac Sim.
--adapter isaaclab : IsaacLabR1Adapter — NOT YET IMPLEMENTED (see
                   episode_gen/sim_adapter.py TODOs). Exits with an explicit
                   error rather than silently falling back to mock.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from episode_gen.episode_runner import EpisodeRunConfig, EpisodeRunner
from episode_gen.scenario import ScenarioConfig


def build_adapter(name: str):
    if name == "mock":
        from episode_gen.sim_adapter import MockSimAdapter
        return MockSimAdapter()
    if name == "isaaclab":
        raise SystemExit(
            "IsaacLabR1Adapter is a skeleton (see episode_gen/sim_adapter.py "
            "TODOs) — it is not runnable yet. Finish wiring it against a live "
            "Isaac Lab scene + the scene USD before using --adapter isaaclab."
        )
    raise SystemExit(f"unknown adapter {name!r}, expected 'mock' or 'isaaclab'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", required=True, help="path to a scenario YAML")
    p.add_argument("--out-dir", required=True, help="parent dir for this episode's output")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--adapter", choices=["mock", "isaaclab"], default="mock")
    args = p.parse_args()

    scenario = ScenarioConfig.from_yaml(args.scenario)
    adapter = build_adapter(args.adapter)
    runner = EpisodeRunner(adapter)
    run_cfg = EpisodeRunConfig(
        out_dir=f"{args.out_dir}/{scenario.scenario_id}_{args.episode_index:05d}",
        episode_index=args.episode_index,
        task_index=args.task_index,
    )
    trace = runner.run(scenario, run_cfg)

    print(f"scenario_id={trace.result.scenario_id} success={trace.result.success} "
          f"recovery_count={trace.result.recovery_count} "
          f"failure_reason={trace.result.failure_reason} "
          f"frames={trace.result.num_frames} out_dir={trace.result.out_dir}")
    if args.adapter == "mock":
        print("NOTE: --adapter mock is a control-flow dry-run, not physically real data.", file=sys.stderr)


if __name__ == "__main__":
    main()
