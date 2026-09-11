"""Preview the non-destructive Sugar Box replacement in spacerobot.usd."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.scene_overlay import add_sugar_box_overlay, settle_target_on_support
from r1_bimanual_dataset.tools.physical_reference_probe import _open_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Session-only Sugar Box replacement preview")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--gui", action="store_true", help="show the prepared scene")
    parser.add_argument("--keep-open", action="store_true", help="keep GUI open after successful preparation")
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "sugar_scene_precheck.json")
    args = parser.parse_args()
    scene_cfg = load_scene_config(args.scene_config)
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(
        simulation_app_config(headless=not args.gui, renderer="RayTracedLighting" if args.gui else "None")
    )
    try:
        stage, timeline = _open_stage(app, scene_cfg["stage_path"])
        overlay = add_sugar_box_overlay(stage, scene_cfg)
        # Let the resolver compose the remote USD before computing its bound.
        for _ in range(120):
            app.update()
        report = settle_target_on_support(stage, overlay)
        for _ in range(10):
            app.update()
        report["source_target_hidden_session_only"] = overlay.source_target_path
        report["table_moved"] = False
        report["source_usd_modified"] = False
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SUGAR_SCENE_PRECHECK=" + json.dumps(report, ensure_ascii=False), flush=True)
        timeline.stop()
        if args.gui and args.keep_open:
            while app.is_running():
                app.update()
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
