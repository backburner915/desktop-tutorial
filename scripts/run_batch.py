#!/usr/bin/env python3
"""Run every scenario YAML in a directory (e.g. one produced by
`python -m episode_gen.batch_generate`) through the FSM and write a
summary report for human spot-checking.

    python -m episode_gen.batch_generate \
        --base configs/scenarios/r11_premature_table_contact.yaml \
        --out configs/scenarios/generated/r11 --n 200 --seed 0

    python scripts/run_batch.py \
        --scenarios configs/scenarios/generated/r11 \
        --out-dir out/episodes --adapter mock

This is the one script hundreds/thousands of generated configs get run
through — there is no such thing as "200 generated scripts", only 200
generated YAML files fed through this single runner.

--adapter isaaclab requires --env-factory module:function, same as
scripts/run_episode.py — see that script's --help or
docs/sim_adapter_handoff.md for what the factory function needs to
return. The env is built ONCE and reused across the whole batch (Isaac
Sim startup is expensive); each episode calls adapter.reset(scenario)
to move to the next one.
"""

from __future__ import annotations

import argparse
import importlib
import json
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
    p.add_argument("--scenarios", required=True, help="directory of scenario YAMLs")
    p.add_argument("--out-dir", required=True, help="parent dir for episode outputs")
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--adapter", choices=["mock", "isaaclab"], default="mock")
    p.add_argument("--env-factory", default=None, help="module:function returning the Isaac Lab env (isaaclab only)")
    args = p.parse_args()

    paths = sorted(Path(args.scenarios).glob("*.yaml"))
    if not paths:
        raise SystemExit(f"no *.yaml files found under {args.scenarios}")

    adapter = build_adapter(args.adapter, args.env_factory)
    runner = EpisodeRunner(adapter)

    rows = []
    for i, path in enumerate(paths):
        scenario = ScenarioConfig.from_yaml(path)
        run_cfg = EpisodeRunConfig(
            out_dir=f"{args.out_dir}/{scenario.scenario_id}",
            episode_index=i,
            task_index=args.task_index,
        )
        trace = runner.run(scenario, run_cfg)
        rows.append({
            "source_config": str(path),
            "scenario_id": trace.result.scenario_id,
            "success": trace.result.success,
            "failure_reason": trace.result.failure_reason,
            "recovery_count": trace.result.recovery_count,
            "num_frames": trace.result.num_frames,
            "out_dir": trace.result.out_dir,
        })

    n = len(rows)
    n_success = sum(r["success"] for r in rows)
    n_recovered = sum(r["recovery_count"] > 0 for r in rows)
    summary = {
        "adapter": args.adapter,
        "total_episodes": n,
        "succeeded": n_success,
        "failed": n - n_success,
        "recovered_at_least_once": n_recovered,
        "failure_reasons": sorted({r["failure_reason"] for r in rows if r["failure_reason"]}),
        "episodes": rows,
    }
    out_path = Path(args.out_dir) / "batch_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"{n_success}/{n} succeeded, {n_recovered} needed recovery, "
          f"{n - n_success} failed. Report: {out_path}")
    if args.adapter == "mock":
        print("NOTE: --adapter mock is a control-flow dry-run, not physically real data.")


if __name__ == "__main__":
    main()
