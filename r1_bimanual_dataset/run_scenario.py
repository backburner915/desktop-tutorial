from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import ScenarioConfig, load_scene_config
from r1_bimanual_dataset.core.launch import (
    available_system_memory_mb,
    prepare_isaac_environment,
    simulation_app_config,
)


def main() -> int:
    from r1_bimanual_dataset.core.reference_policy import reject_legacy_execution
    reject_legacy_execution()
    parser = argparse.ArgumentParser(description="Run one R1 bimanual scenario")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--scene-config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_raw")
    parser.add_argument("--gui", action="store_true", help="show Isaac Sim UI")
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="with --gui, keep the Isaac Sim window open after the episode",
    )
    parser.add_argument(
        "--record-cameras-in-gui",
        action="store_true",
        help="opt into three-camera GPU recording in GUI mode; normally GUI is a lightweight preview",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="skip camera products and dataset writes for fast geometry/controller debugging",
    )
    parser.add_argument(
        "--base-offset",
        nargs=2,
        type=float,
        metavar=("DX", "DY"),
        help="override scene_config base_target_offset_xy for this run only",
    )
    args = parser.parse_args()
    scenario = ScenarioConfig.from_json(args.scenario)
    scene_cfg = load_scene_config(args.scene_config)
    if args.base_offset is not None:
        scene_cfg["base_target_offset_xy"] = [float(args.base_offset[0]), float(args.base_offset[1])]
    gui_preview = bool(args.gui and not args.record_cameras_in_gui)
    if scenario.target_object and scenario.target_object != scene_cfg["target_object"]:
        raise ValueError("scenario target_object differs from the fixed scene target")

    prepare_isaac_environment()
    if gui_preview:
        available_mb = available_system_memory_mb()
        # GUI preview does not create camera render products, so it needs
        # less RAM than the old viewport-plus-three-camera mode.  Keep a
        # safety margin above the 253 MB level that caused the Vulkan crash.
        if 0 <= available_mb < 1536:
            print(
                f"R1 runner: GUI preview not started; only {available_mb} MB physical RAM is free. "
                "At least 1536 MB is required; close other applications and retry, or run headless for data generation.",
                flush=True,
            )
            return 3
        print("R1 runner: GUI preview selected; no camera render products or dataset files will be created", flush=True)
    print("R1 runner: starting Isaac Sim; startup may take several minutes on a memory-constrained machine...", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=not args.gui))
    print("R1 runner: Isaac Sim started", flush=True)
    try:
        from r1_bimanual_dataset.core.runner import run_episode

        try:
            if gui_preview:
                print("R1 runner: GUI preview mode; camera recording is disabled to avoid Vulkan/GPU crashes", flush=True)
            result = run_episode(
                app,
                scenario,
                scene_cfg,
                args.output,
                record_cameras=not gui_preview and not args.no_cameras,
                save_data=not gui_preview and not args.no_cameras,
                gui_preview=gui_preview,
            )
        except Exception as exc:
            print(f"R1 runner: unhandled exception: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            return 1
        if gui_preview:
            # Reapply the view after the episode too: Kit may restore its
            # authored stage camera when the timeline stops. This leaves the
            # completed grasp visible in the task workspace while --keep-open
            # is waiting for the user.
            try:
                import numpy as np
                from isaacsim.core.utils.viewports import set_active_viewport_camera
                from r1_bimanual_dataset.core.lighting import set_gui_task_view

                set_active_viewport_camera("/OmniverseKit_Persp")
                center = np.asarray(scenario.object_position, dtype=np.float64)
                set_gui_task_view(
                    center + np.asarray([1.35, -1.55, 0.95]),
                    center + np.asarray([0.0, 0.0, 0.30]),
                )
                app.update()
                print("R1 runner: GUI viewport focused on robot and T01", flush=True)
            except Exception as exc:
                print(f"R1 runner: unable to focus GUI viewport: {type(exc).__name__}: {exc}", flush=True)
        print(f"R1 runner: episode finished with {result.actual_outcome}", flush=True)
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        if args.keep_open and args.gui:
            print("R1 runner: keeping GUI open; close the Isaac Sim window to exit.", flush=True)
            while app.is_running():
                app.update()
        return 0 if result.actual_outcome != "INVALID" else 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
