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
    parser = argparse.ArgumentParser(description="Repeat the corrected-scene reference scenario")
    parser.add_argument("--scenario", type=Path, default=ROOT / "reference" / "scenario_reference.json")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--scene-config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_reference")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record-cameras-in-gui", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    scenario = ScenarioConfig.from_json(args.scenario)
    scene_cfg = load_scene_config(args.scene_config)
    gui_preview = bool(args.gui and not args.record_cameras_in_gui)

    prepare_isaac_environment()
    print("R1 reference: starting Isaac Sim; startup may take several minutes...", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=not args.gui))
    results = []
    try:
        from r1_bimanual_dataset.core.runner import run_episode

        for repetition in range(args.repetitions):
            # Keep the same verified reference design while changing only the
            # output episode id to avoid overwriting a previous run.
            if repetition:
                scenario = ScenarioConfig.from_dict({**scenario.to_dict(), "scenario_id": 900000 + repetition, "seed": scenario.seed + repetition})
            result = run_episode(
                app,
                scenario,
                scene_cfg,
                args.output,
                record_cameras=not gui_preview,
                save_data=not gui_preview,
                gui_preview=gui_preview,
            )
            results.append(result.as_dict())
            if sum(x["actual_outcome"] == "INVALID" for x in results) >= 3:
                break
    finally:
        app.close()
    summary = {
        "success": sum(x["actual_outcome"] == "SUCCESS" for x in results),
        "failure": sum(x["actual_outcome"] == "FAILURE" for x in results),
        "invalid": sum(x["actual_outcome"] == "INVALID" for x in results),
        "results": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["success"] >= 9 and len(results) == args.repetitions else 2


if __name__ == "__main__":
    raise SystemExit(main())
