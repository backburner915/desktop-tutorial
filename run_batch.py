from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import ScenarioConfig, load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config


def main() -> int:
    from r1_bimanual_dataset.core.reference_policy import reject_legacy_execution
    reject_legacy_execution()
    parser = argparse.ArgumentParser(description="Run a scenario directory and stop after three consecutive INVALID episodes")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "scenarios")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_raw")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record-cameras-in-gui", action="store_true")
    args = parser.parse_args()
    files = sorted(args.scenarios.glob("scenario_*.json"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no scenario JSON files under {args.scenarios}")
    scene_cfg = load_scene_config(args.scene_config)
    gui_preview = bool(args.gui and not args.record_cameras_in_gui)

    prepare_isaac_environment()
    print("R1 batch: starting Isaac Sim; startup may take several minutes...", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=not args.gui))
    results = []
    invalid_streak = 0
    try:
        from r1_bimanual_dataset.core.runner import run_episode

        for path in files:
            scenario = ScenarioConfig.from_json(path)
            result = run_episode(
                app,
                scenario,
                scene_cfg,
                args.output,
                record_cameras=not gui_preview,
                save_data=not gui_preview,
                gui_preview=gui_preview,
            )
            data = result.as_dict()
            data["scenario_file"] = path.name
            results.append(data)
            print(json.dumps(data, ensure_ascii=False), flush=True)
            invalid_streak = invalid_streak + 1 if result.actual_outcome == "INVALID" else 0
            if invalid_streak >= 3:
                print("BATCH STOP: three consecutive INVALID episodes", flush=True)
                break
    finally:
        app.close()
    summary = {
        "total_attempted": len(results),
        "success": sum(x["actual_outcome"] == "SUCCESS" for x in results),
        "failure": sum(x["actual_outcome"] == "FAILURE" for x in results),
        "invalid": sum(x["actual_outcome"] == "INVALID" for x in results),
        "stopped_after_three_invalid": invalid_streak >= 3,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "batch_summary.json").write_text(json.dumps({**summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["invalid"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
