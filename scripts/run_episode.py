#!/usr/bin/env python3
"""Run ONE scenario config through the FSM and write one episode.

    python scripts/run_episode.py \
        --scenario configs/scenarios/r11_premature_table_contact.yaml \
        --out-dir out/episodes --episode-index 0 --adapter mock

--adapter mock   : MockSimAdapter, runs anywhere (no Isaac Sim) — control-flow
                   dry-run only, does NOT produce physically real trajectories
                   or real images. Useful to sanity-check a config before
                   handing it to Isaac Sim.
--adapter isaaclab : IsaacLabR1Adapter, the real Isaac Lab bridge. Requires
                   --env-factory module:function — a function YOU write (in
                   your own bootstrap module, alongside AppLauncher/scene
                   setup, which is Isaac-Sim-specific and can't live in this
                   repo) that returns the env/scene handle to pass in. e.g.:

                     python scripts/run_episode.py \
                         --scenario configs/scenarios/n01_nominal.yaml \
                         --out-dir out/episodes --adapter isaaclab \
                         --env-factory my_isaac_bootstrap:build_env

                   See docs/sim_adapter_handoff.md for what build_env() needs
                   to expose.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from episode_gen.episode_runner import EpisodeRunConfig, EpisodeRunner
from episode_gen.scenario import ScenarioConfig


def build_adapter(name: str, env_factory: str | None):
    if name == "mock":
        from episode_gen.sim_adapter import MockSimAdapter
        return MockSimAdapter()
    if name == "isaaclab":
        if not env_factory:
            raise SystemExit(
                "--adapter isaaclab requires --env-factory module:function — see "
                "this script's --help / docs/sim_adapter_handoff.md."
            )
        module_name, _, func_name = env_factory.partition(":")
        if not module_name or not func_name:
            raise SystemExit(f"--env-factory must be 'module:function', got {env_factory!r}")
        module = importlib.import_module(module_name)
        env = getattr(module, func_name)()
        from episode_gen.sim_adapter import IsaacLabR1Adapter
        return IsaacLabR1Adapter(env)
    raise SystemExit(f"unknown adapter {name!r}, expected 'mock' or 'isaaclab'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, help="path to a scenario YAML")
    p.add_argument("--out-dir", required=True, help="parent dir for this episode's output")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--adapter", choices=["mock", "isaaclab"], default="mock")
    p.add_argument("--env-factory", default=None, help="module:function returning the Isaac Lab env (isaaclab only)")
    args = p.parse_args()

    scenario = ScenarioConfig.from_yaml(args.scenario)
    adapter = build_adapter(args.adapter, args.env_factory)
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
